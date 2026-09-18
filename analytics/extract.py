"""Turn the chatbot and backend databases into flat question/thread rows and a summary."""

import uuid
from collections import Counter
from datetime import date, datetime, timezone
from itertools import groupby

from sqlalchemy import func
from sqlmodel import Session, select

from app.db.models import (
    Feedback,
    FeedbackRating,
    Message,
    MessageRole,
    QueryHandle,
    Thread,
)

# Messages are streamed in batches of this many rows. An assistant message's
# `events` JSON is by far the largest column in the database, so holding every
# message at once is what made this export's memory grow with usage.
_MESSAGE_BATCH = 500

# Chatbot Thread.user_id is the JWT `uuid` claim, which is the backend
# Account.uuid column. Accounts whose email is on this domain are BD
# team members, not real users (see auth.py: payload.get("uuid")).
TEAM_EMAIL_DOMAIN = "@basedosdados.org"

# File formats the agent can export. Mirror of app.exports.OFFERED_EXPORT_FORMATS,
# hardcoded so this script needn't import app.exports (which pulls in the full app
# Settings). If the app adds a format, add it here to get its own column/metric.
EXPORT_FORMATS = ("AVRO", "CSV", "JSONL", "PARQUET")


def _dataviz_actions(events) -> tuple[int, Counter]:
    """Count chart renders and exports-by-format from an assistant message's events.

    Charts and exports aren't tables — each is a successful tool-output artifact in
    the message's `events` JSON: {"type": "chart", ...} and
    {"type": "export", "format": "CSV", ...}. Returns (chart_count, {format: count}),
    de-duplicating by tool_call_id so an output counted once even if it recurs.
    """
    charts = 0
    formats: Counter = Counter()
    seen: set = set()
    for event in events or []:
        for out in (event.get("data") or {}).get("tool_outputs") or []:
            call_id = out.get("tool_call_id")
            if call_id in seen or out.get("status") != "success":
                continue
            seen.add(call_id)
            artifact = out.get("artifact")
            if not isinstance(artifact, dict):
                continue
            if artifact.get("type") == "chart":
                charts += 1
            elif artifact.get("type") == "export" and artifact.get("format"):
                formats[artifact["format"]] += 1
    return charts, formats


def _naive_utc(dt: datetime | None):
    """Excel has no concept of tz-aware datetimes; drop the (already-UTC) tzinfo."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _thread_index(session: Session, include_deleted: bool) -> dict:
    """Map thread id -> the thread columns the export needs.

    Columns, not ORM objects: plain rows are a fraction of the size and never
    enter the Session's identity map, so nothing accumulates as the stream runs.
    """
    statement = select(
        Thread.id,
        Thread.user_id,
        Thread.title,
        Thread.language,
        Thread.created_at,
        Thread.deleted,
    )
    if not include_deleted:
        statement = statement.where(Thread.deleted.is_(False))
    return {row.id: row for row in session.exec(statement)}


def _downloads_by_message(session: Session) -> dict[uuid.UUID, int]:
    """Map message id -> number of query handles, counted in SQL.

    The handles themselves are never loaded: each one carries a
    `destination_table` JSON blob and only the count reaches the export.
    """
    statement = select(QueryHandle.message_id, func.count()).group_by(
        QueryHandle.message_id
    )
    return {message_id: count for message_id, count in session.exec(statement)}


def _feedback_by_message(session: Session) -> dict[uuid.UUID, tuple[int, str | None]]:
    """Map message id -> (1 positive / 0 negative, comment) for rated messages.

    A message missing from this map writes a genuinely blank cell, so Excel's
    AVERAGE/COUNT ignore unrated messages: AVERAGE(feedback) is then the positive
    share among rated ones.
    """
    statement = select(Feedback.message_id, Feedback.rating, Feedback.comments)
    return {
        message_id: (1 if rating == FeedbackRating.POSITIVE else 0, comments)
        for message_id, rating, comments in session.exec(statement)
    }


def fetch_rows(
    session: Session,
    email_by_user: dict[str, str],
    include_deleted: bool,
    since: date | None,
    until: date | None,
):
    """Return (question_rows, thread_rows, empty_conversations) from the database.

    A question row is one USER message joined to the ASSISTANT message that
    answered it (linked via Message.user_message_id) plus that answer's feedback
    and query-handle count. Every row carries an `is_team` flag, derived from
    `email_by_user` (the backend account uuid -> email map).

    Messages stream ordered by thread, so only one conversation is held at a time
    on the read side; the small per-message aggregates are queried up front.
    """
    threads = _thread_index(session, include_deleted)
    downloads_by_message = _downloads_by_message(session)
    feedback_by_message = _feedback_by_message(session)

    # Ordered by thread so each conversation arrives as one contiguous run, and
    # by time within it so turn_index follows insertion order. `structured_response`
    # is deliberately absent: the export never reads it, and it is a JSON column.
    statement = (
        select(
            Message.id,
            Message.thread_id,
            Message.user_message_id,
            Message.role,
            Message.content,
            Message.events,
            Message.status,
            Message.model_uri,
            Message.created_at,
        )
        .order_by(Message.thread_id, Message.created_at)
        .execution_options(yield_per=_MESSAGE_BATCH)
    )

    question_rows = []
    thread_stats: dict = {}
    threads_with_questions: set = set()

    def _is_team(user_id: str) -> int:
        """1 if the account's email is on the team domain, else 0.

        Only the derived flag is kept — the email itself is personal data and is
        deliberately dropped. Int, not a Python bool, so a pt-BR Excel shows 1/0,
        not VERDADEIRO/FALSO.
        """
        email = email_by_user.get(user_id, "")
        return int(email.lower().endswith(TEAM_EMAIL_DOMAIN))

    for thread_id, group in groupby(session.exec(statement), key=lambda m: m.thread_id):
        # One conversation at a time, bounded by its own length.
        messages = list(group)

        thread = threads.get(thread_id)
        if thread is None:  # a deleted thread, excluded by the caller
            continue

        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        if not user_msgs:
            continue
        threads_with_questions.add(thread_id)

        # An assistant reply always belongs to the thread of the message it
        # answers, so this index only has to cover the current conversation.
        answer_by_user_msg = {
            m.user_message_id: m
            for m in messages
            if m.role == MessageRole.ASSISTANT and m.user_message_id is not None
        }

        counted_for_thread = 0
        thread_charts = 0
        thread_exports: Counter = Counter()
        for turn_index, msg in enumerate(user_msgs, start=1):
            # Filter on the same UTC value that is exported: Message.created_at is
            # tz-aware, so `.date()` on it would follow the connection's timezone
            # and disagree with the documented "(UTC)" filter by up to a day.
            created = _naive_utc(msg.created_at)
            if since is not None and created.date() < since:
                continue
            if until is not None and created.date() > until:
                continue

            answer = answer_by_user_msg.get(msg.id)
            counted_for_thread += 1

            charts, export_formats = _dataviz_actions(answer.events if answer else None)
            thread_charts += charts
            thread_exports += export_formats

            rating, comment = (
                feedback_by_message.get(answer.id, (None, None))
                if answer
                else (None, None)
            )

            question_rows.append(
                {
                    "user_id": str(thread.user_id),
                    "is_team": _is_team(str(thread.user_id)),
                    "thread_id": str(thread_id),
                    "thread_title": thread.title,
                    "language": thread.language,
                    "thread_deleted": int(thread.deleted),
                    "turn_index": turn_index,
                    "asked_at_utc": created,
                    "question": msg.content,
                    "answer": answer.content if answer else None,
                    "answer_status": answer.status.value if answer else "NO_ANSWER",
                    "model_uri": (answer or msg).model_uri,
                    "downloads_offered": (
                        downloads_by_message.get(answer.id, 0) if answer else 0
                    ),
                    "charts_rendered": charts,
                    "exports_made": sum(export_formats.values()),
                    **{
                        f"exports_{fmt.lower()}": export_formats.get(fmt, 0)
                        for fmt in EXPORT_FORMATS
                    },
                    "feedback": rating,
                    "feedback_comment": comment,
                    "question_id": str(msg.id),
                }
            )

        if counted_for_thread:
            thread_stats[thread_id] = {
                "user_id": str(thread.user_id),
                "is_team": _is_team(str(thread.user_id)),
                "thread_id": str(thread_id),
                "thread_title": thread.title,
                "language": thread.language,
                "deleted": int(thread.deleted),
                "created_at_utc": _naive_utc(thread.created_at),
                "num_questions": counted_for_thread,
                "charts_rendered": thread_charts,
                "exports_made": sum(thread_exports.values()),
                **{
                    f"exports_{fmt.lower()}": thread_exports.get(fmt, 0)
                    for fmt in EXPORT_FORMATS
                },
            }

    question_rows.sort(key=lambda r: r["asked_at_utc"])
    thread_rows = sorted(thread_stats.values(), key=lambda t: t["created_at_utc"])

    # Conversations that exist but hold no user message at all (created then
    # abandoned). Independent of the date filters — a thread has questions
    # or it doesn't. `threads` already respects the deleted filter.
    empty_conversations = len(threads) - len(threads_with_questions)

    return question_rows, thread_rows, empty_conversations


def build_summary(
    question_rows: list[dict], thread_rows: list[dict], empty_conversations: int = 0
) -> list[tuple]:
    """Headline aggregates as (label, value) pairs."""
    users = {r["user_id"] for r in question_rows}
    dates = [r["asked_at_utc"] for r in question_rows if r["asked_at_utc"]]
    languages = Counter(r["language"] for r in question_rows)
    statuses = Counter(r["answer_status"] for r in question_rows)
    feedback = Counter(
        r["feedback"] for r in question_rows if r["feedback"] is not None
    )
    with_download = sum(1 for r in question_rows if r["downloads_offered"] > 0)
    q_with_chart = sum(1 for r in question_rows if r["charts_rendered"])
    q_with_export = sum(1 for r in question_rows if r["exports_made"])
    charts_total = sum(r["charts_rendered"] for r in question_rows)
    exports_by_format = {
        fmt: sum(r[f"exports_{fmt.lower()}"] for r in question_rows)
        for fmt in EXPORT_FORMATS
    }
    exports_total = sum(exports_by_format.values())
    n_questions = len(question_rows)

    rows: list[tuple] = [
        ("Gerado em (UTC)", _naive_utc(datetime.now(timezone.utc))),
        ("Total de perguntas", n_questions),
        ("Usuários distintos", len(users)),
        ("Total de conversas", len(thread_rows)),
        ("Conversas vazias (sem perguntas, desde sempre)", empty_conversations),
        (
            "Média de perguntas por usuário",
            round(n_questions / len(users), 2) if users else 0,
        ),
        (
            "Média de perguntas por conversa",
            round(n_questions / len(thread_rows), 2) if thread_rows else 0,
        ),
        ("Primeira pergunta (UTC)", min(dates) if dates else None),
        ("Última pergunta (UTC)", max(dates) if dates else None),
        (
            "Perguntas que ofereceram download de dados",
            f"{with_download} ({with_download / n_questions:.0%})"
            if n_questions
            else 0,
        ),
    ]

    team_users = {r["user_id"] for r in question_rows if r["is_team"]}
    team_questions = sum(1 for r in question_rows if r["is_team"])
    rows += [
        ("", ""),
        ("===== Equipe vs. Usuários reais =====", ""),
        ("Usuários reais distintos", len(users) - len(team_users)),
        ("Membros da equipe distintos", len(team_users)),
        ("Perguntas de usuários reais", n_questions - team_questions),
        ("Perguntas da equipe", team_questions),
    ]

    feedback_pt = {1: "positivo", 0: "negativo"}
    rows += [("", ""), ("===== Perguntas por idioma =====", "")]
    rows += [(lang or "(desconhecido)", n) for lang, n in languages.most_common()]
    rows += [("", ""), ("===== Respostas por status =====", "")]
    rows += [(status, n) for status, n in statuses.most_common()]

    rows += [("", ""), ("===== Gráficos e exportações =====", "")]
    rows += [
        (
            "Perguntas que geraram gráfico",
            f"{q_with_chart} ({q_with_chart / n_questions:.0%})" if n_questions else 0,
        ),
        ("Gráficos gerados", charts_total),
        (
            "Perguntas que exportaram arquivo",
            f"{q_with_export} ({q_with_export / n_questions:.0%})"
            if n_questions
            else 0,
        ),
        ("Arquivos exportados", exports_total),
    ]
    rows += [(f"Exportações em {fmt}", n) for fmt, n in exports_by_format.items()]

    rated = sum(feedback.values())
    positive = feedback.get(1, 0)
    rows += [("", ""), ("===== Feedback recebido =====", "")]
    rows += [
        (feedback_pt.get(label, label), n) for label, n in feedback.most_common()
    ] or [("(nenhum)", 0)]
    rows += [
        ("Total avaliadas", rated),
        ("% positivo (entre avaliadas)", f"{positive / rated:.0%}" if rated else "—"),
    ]
    return rows

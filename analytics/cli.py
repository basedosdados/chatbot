"""Export chatbot usage (questions, threads, feedback, charts/exports) for the team.

Connections come from env vars or a `.env.analytics` file (see analytics/settings.py);
point them at production, e.g. via the Cloud SQL Auth Proxy. GCP access uses ADC.

Sinks: `--bigquery-dataset` writes the BigQuery tables; `--out` writes an .xlsx
(default chatbot_usage.xlsx when neither is given). xlsx output needs `--with openpyxl`.

Examples (run from the repo root):
    # preview only, nothing written
    uv run python -m analytics.cli --dry-run

    # xlsx for a one-off hand-off
    uv run --with openpyxl python -m analytics.cli --out usage.xlsx

    # load BigQuery (always the full history; date filters are xlsx-only)
    uv run python -m analytics.cli --bigquery-dataset chatbot_analytics
"""

import argparse
import sys
from datetime import date, datetime

from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlmodel import Session, create_engine

from analytics.bigquery import load_to_bigquery
from analytics.extract import TEAM_EMAIL_DOMAIN, build_summary, fetch_rows
from analytics.settings import (
    BackendDatabaseSettings,
    ChatbotDatabaseSettings,
    Settings,
)
from analytics.xlsx import write_workbook

# Each DatabaseSettings validates on its own, so a missing field arrives with a
# bare name ("host"); this maps the failing model back to the env var to set.
_ENV_PREFIXES = {
    cls.__name__: cls.model_config.get("env_prefix", "")
    for cls in (ChatbotDatabaseSettings, BackendDatabaseSettings)
}


def _valid_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        help=(
            "output .xlsx path. If neither --out nor --bigquery-dataset is given, "
            "an .xlsx is written to chatbot_usage.xlsx; with only --bigquery-dataset, "
            "xlsx file is skipped."
        ),
    )
    parser.add_argument(
        "--bigquery-dataset",
        help=(
            "Load the questions and threads into this BigQuery dataset (tables "
            "`questions` and `threads`, fully replaced each run). The dataset "
            "must already exist, in GCP_PROJECT. Can be combined with --out, but "
            "not with --since/--until: the load replaces the tables, so a filtered "
            "run would drop every row outside the window."
        ),
    )
    parser.add_argument(
        "--since",
        type=_valid_date,
        metavar="YYYY-MM-DD",
        help="only questions asked on/after this date (UTC). xlsx output only.",
    )
    parser.add_argument(
        "--until",
        type=_valid_date,
        metavar="YYYY-MM-DD",
        help="only questions asked on/before this date (UTC). xlsx output only.",
    )
    parser.add_argument(
        "--exclude-deleted",
        action="store_true",
        help="skip conversations the user soft-deleted (default: include them)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "connect and run the query, print a preview of what would be "
            "exported, but do not write anything (no sink credentials needed)"
        ),
    )
    return parser


def _load_settings() -> Settings:
    """Build Settings from the environment, exiting cleanly on missing config."""
    try:
        return Settings()
    except ValidationError as exc:
        prefix = _ENV_PREFIXES.get(exc.title, "")
        problems = "\n".join(
            f"  {prefix}{'_'.join(str(part) for part in err['loc']).upper()}: "
            f"{err['msg']}"
            for err in exc.errors()
        )
        sys.exit(
            f"Could not read the analytics config:\n{problems}\n"
            f"Set the CHATBOT_DB_* and BACKEND_DB_* variables in the environment "
            f"or in a .env.analytics file in the working directory."
        )


def _connect(url: URL, label: str):
    """Log a masked connection line and return a read-only engine for `url`.

    `default_transaction_read_only=on` is a psycopg startup option, so PostgreSQL
    itself rejects any INSERT/UPDATE/DELETE/DDL — a hard guarantee independent of
    this script's code.
    """
    print(f"Connecting to {label} DB at {url.render_as_string(hide_password=True)}")
    return create_engine(
        url, connect_args={"options": "-c default_transaction_read_only=on"}
    )


def _load_account_emails(url: URL) -> dict[str, str]:
    """Load the backend Account.uuid -> email map.

    Raw SQL against the `account` table, so this script needs no dependency on the
    backend repo's models. Keyed by str(uuid) to match the stringified user_id.
    """
    engine = _connect(url, "backend")

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT uuid, email FROM account"))
        email_by_user: dict[str, str] = {str(uuid): email for uuid, email in rows}

    n_team = sum(
        1 for e in email_by_user.values() if e.lower().endswith(TEAM_EMAIL_DOMAIN)
    )

    print(f"Loaded {len(email_by_user)} accounts ({n_team} on {TEAM_EMAIL_DOMAIN})\n")

    return email_by_user


def _print_dry_run(args, question_rows, thread_rows, summary_rows) -> None:
    targets = []
    if args.bigquery_dataset:
        targets.append(f"BigQuery dataset {args.bigquery_dataset}")
    if args.out or not args.bigquery_dataset:
        targets.append(args.out or "chatbot_usage.xlsx")
    print(
        f"\n[dry run] would write to {' and '.join(targets)}: "
        f"{len(question_rows)} questions across {len(thread_rows)} conversations.\n"
    )

    # Align data rows on one column; headers (empty value) and blanks (empty label)
    # print as-is. Width is driven by the longest data label only.
    print("Summary preview:")
    width = max(
        (len(label) for label, value in summary_rows if label and value != ""),
        default=0,
    )
    for label, value in summary_rows:
        if not label:
            print()
        elif value == "":
            print(f"  {label}")
        else:
            print(f"  {label:<{width}}  {value}")

    print("\nFirst questions that would be exported:")
    for r in question_rows[:5]:
        tag = " [TEAM]" if r["is_team"] else ""
        print(f"  [{r['asked_at_utc']}] ({r['language']}){tag} {r['question'][:100]!r}")
    print("\n[dry run] nothing written.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # The BigQuery load truncates both tables, so a date-filtered run would
    # replace the whole history with just that window. Refuse the combination
    # rather than silently destroying rows; --dry-run writes nothing, so it is fine.
    if args.bigquery_dataset and (args.since or args.until) and not args.dry_run:
        parser.error(
            "--since/--until cannot be combined with --bigquery-dataset: the load "
            "replaces the tables in full, so every row outside the date range would "
            "be deleted. Load the full history, and use the date filters for --out."
        )

    settings = _load_settings()
    chatbot_url = settings.chatbot_db.url
    accounts_url = settings.backend_db.url

    email_by_user = _load_account_emails(accounts_url)

    with Session(_connect(chatbot_url, "chatbot")) as session:
        question_rows, thread_rows, empty_conversations = fetch_rows(
            session,
            email_by_user,
            include_deleted=not args.exclude_deleted,
            since=args.since,
            until=args.until,
        )

    if not question_rows:
        # Not an error: an empty window is a normal outcome for a scheduled run,
        # and a non-zero exit would be reported as a job failure.
        print("No questions found for the given filters — nothing written.")
        return

    summary_rows = build_summary(question_rows, thread_rows, empty_conversations)

    if args.dry_run:
        _print_dry_run(args, question_rows, thread_rows, summary_rows)
        return

    if args.bigquery_dataset:
        load_to_bigquery(
            args.bigquery_dataset,
            question_rows,
            thread_rows,
            project=settings.gcp_project,
        )

    # Write xlsx when asked for explicitly, or when no other sink was given.
    out_path = args.out or (None if args.bigquery_dataset else "chatbot_usage.xlsx")
    if out_path:
        write_workbook(out_path, question_rows, thread_rows, summary_rows)
        print(
            f"\nWrote {out_path}: {len(question_rows)} questions "
            f"across {len(thread_rows)} conversations."
        )


if __name__ == "__main__":
    main()

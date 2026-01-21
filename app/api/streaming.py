import json
import uuid
from typing import Any, AsyncIterator, Literal

from google.api_core import exceptions as google_api_exceptions
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, JsonValue

from app.api.schemas import ConfigDict
from app.db.database import AsyncDatabase
from app.db.models import Message, MessageCreate, MessageRole, MessageStatus
from app.settings import settings


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


class ToolOutput(BaseModel):
    status: Literal["error", "success"]
    tool_call_id: str
    tool_name: str
    content: str
    artifact: JsonValue | None = None
    metadata: JsonValue | None = None


EventType = Literal[
    "tool_call",
    "tool_output",
    "final_answer",
    "error",
    "complete",
]


class EventData(BaseModel):
    run_id: uuid.UUID | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_outputs: list[ToolOutput] | None = None
    error_details: dict[str, Any] | None = None


class StreamEvent(BaseModel):
    type: EventType
    data: EventData

    def to_sse(self) -> str:
        return self.model_dump_json() + "\n\n"


class ErrorMessage:
    UNEXPECTED = (
        "Ops, algo deu errado! Ocorreu um erro inesperado. Por favor, tente novamente. "
        "Se o problema persistir, avise-nos. Obrigado pela paciência!"
    )

    CONTEXT_OVERFLOW = (
        "Sua última mensagem ultrapassou o limite de tamanho para esta conversa. "
        "Por favor, tente dividir sua solicitação em partes menores "
        "ou inicie uma nova conversa."
    )

    GRAPH_RECURSION_LIMIT_REACHED = (
        "Desculpe, não consegui encontrar uma resposta para a sua pergunta. "
        "Por favor, tente reformulá-la ou pergunte algo diferente."
    )


def _truncate_json(
    json_string: str, max_list_len: int = 10, max_str_len: int = 300
) -> str:
    """Iteratively truncates a serialized JSON object by shortening lists and strings
    and adding human-readable placeholders.

    Note:
        This function only processes JSON objects (dictionaries). If the serialized JSON
        represents any other type, the original JSON string will be returned unchanged.

    Args:
        json_string (str): The serialized JSON to process.
        max_list_len (int, optional): The max number of items to keep in a list. Defaults to 10.
        max_str_len (int, optional): The max length for any single string. Defaults to 300.

    Returns:
        str: The truncated, formatted, and serialized JSON object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        return json_string

    if not isinstance(data, dict):
        return json_string

    stack = [data]

    while stack:
        current_node = stack.pop()

        if isinstance(current_node, dict):
            items_to_process = current_node.items()
        else:
            items_to_process = enumerate(current_node)

        for key_or_idx, item in items_to_process:
            if isinstance(item, str):
                if len(item) > max_str_len:
                    truncated_str = (
                        item[:max_str_len]
                        + f"... ({len(item) - max_str_len} more characters)"
                    )
                    current_node[key_or_idx] = truncated_str

            elif isinstance(item, list):
                if len(item) > max_list_len:
                    original_len = len(item)
                    del item[max_list_len:]
                    item.append(f"... ({original_len - max_list_len} more items)")
                stack.append(item)

            elif isinstance(item, dict):
                stack.append(item)

    return json.dumps(data, ensure_ascii=False, indent=2)


def _process_chunk(chunk: dict[str, Any]) -> StreamEvent | None:
    """Process a streaming chunk from a react agent workflow into a standardized StreamEvent.

    Args:
        chunk (dict[str, Any]): Raw chunk from agent workflow.
            Only processes "agent" and "tools" nodes.

    Returns:
        StreamEvent | None: Structured event or None if the chunk is ignored:
            - "tool_call" for agent messages with tool calls
            - "tool_output" for tool execution results
            - "final_answer" for agent messages without tool calls
            - None for ignored chunks
    """
    if "agent" in chunk:
        ai_messages: list[AIMessage] = chunk["agent"]["messages"]

        # If no messages are returned, the model returned an empty response
        # with no tool calls. This also counts as a final (but empty) answer.
        if not ai_messages:
            return StreamEvent(type="final_answer", data=EventData(content=""))

        message = ai_messages[0]

        if message.tool_calls:
            event_type = "tool_call"
            tool_calls = [
                ToolCall(
                    id=tool_call["id"], name=tool_call["name"], args=tool_call["args"]
                )
                for tool_call in message.tool_calls
            ]
        else:
            event_type = "final_answer"
            tool_calls = None

        event_data = EventData(content=message.text, tool_calls=tool_calls)

        return StreamEvent(type=event_type, data=event_data)
    elif "tools" in chunk:
        updates = chunk["tools"]

        # single tool call
        if isinstance(updates, dict):
            tool_messages: list[ToolMessage] = updates["messages"]

        # multiple parallel tool calls
        elif isinstance(updates, list):
            tool_messages: list[ToolMessage] = [
                update["messages"][0] for update in updates if "messages" in update
            ]

        # defensive handling (langgraph should only output dicts and lists)
        else:
            tool_messages = []

        tool_outputs = [
            ToolOutput(
                status=message.status,
                tool_call_id=message.tool_call_id,
                tool_name=message.name,
                content=_truncate_json(message.content),
                artifact=message.artifact,
            )
            for message in tool_messages
        ]

        return StreamEvent(
            type="tool_output", data=EventData(tool_outputs=tool_outputs)
        )
    return None


async def stream_response(
    database: AsyncDatabase,
    agent: CompiledStateGraph,
    user_message: Message,
    config: ConfigDict,
    thread_id: str,
    model_uri: str,
) -> AsyncIterator[str]:
    """Stream ReAct Agent's execution progress.

    Args:
        message (str): User's input message.
        config (ConfigDict): Configuration for the agent's execution.
        thread_id (str): Unique identifier for the conversation thread.

    Yields:
        Iterator[str]: JSON string containing the streaming status and the current step data.
    """
    events = []
    artifacts = []
    agent_state = None
    assistant_message = ""
    status = MessageStatus.SUCCESS

    try:
        async for mode, chunk in agent.astream(  # pragma: no cover
            input={"messages": [{"role": "user", "content": user_message.content}]},
            config=config,
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                agent_state = chunk
                continue

            event = _process_chunk(chunk)

            if event is not None:
                if event.type == "tool_output":
                    for output in event.data.tool_outputs:
                        if output.artifact:
                            artifacts.append(output.artifact)

                elif event.type == "final_answer":
                    assistant_message = event.data.content
                    status = MessageStatus.SUCCESS

                events.append(event.model_dump())
                yield event.to_sse()

    except GraphRecursionError:
        logger.warning(f"Graph recursion limit reached for message {config['run_id']}")

        assistant_message = ErrorMessage.GRAPH_RECURSION_LIMIT_REACHED

        status = MessageStatus.SUCCESS

        yield StreamEvent(
            type="final_answer", data=EventData(content=assistant_message)
        ).to_sse()

    except google_api_exceptions.InvalidArgument:
        logger.exception(
            "Agent execution failed with Google API InvalidArgument error:"
        )

        assistant_message = ErrorMessage.UNEXPECTED

        status = MessageStatus.ERROR

        if agent_state is not None:
            model = init_chat_model(settings.MODEL_URI)
            total_tokens = model.get_num_tokens_from_messages(agent_state["messages"])

            if total_tokens >= model.profile.get(
                "max_input_tokens", settings.MAX_TOKENS
            ):
                assistant_message = ErrorMessage.CONTEXT_OVERFLOW

        yield StreamEvent(
            type="error", data=EventData(error_details={"message": assistant_message})
        ).to_sse()

    except Exception:
        logger.exception(f"Unexpected error responding message {config['run_id']}:")

        assistant_message = ErrorMessage.UNEXPECTED

        status = MessageStatus.ERROR

        yield StreamEvent(
            type="error", data=EventData(error_details={"message": assistant_message})
        ).to_sse()

    message_create = MessageCreate(
        id=config["run_id"],
        thread_id=thread_id,
        user_message_id=user_message.id,
        model_uri=model_uri,
        role=MessageRole.ASSISTANT,
        content=assistant_message,
        artifacts=artifacts or None,
        events=events or None,
        status=status,
    )

    message_pair = await database.create_message(message_create)

    yield StreamEvent(type="complete", data=EventData(run_id=message_pair.id)).to_sse()

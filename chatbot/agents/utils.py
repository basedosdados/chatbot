from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def prune_messages(messages: list[BaseMessage], question_limit: int) -> list[RemoveMessage]:
    """Prunes a message list to a limited number of HumanMessages
    and their corresponding AI messages and Tool messages.

    Args:
        messages (list[BaseMessage]): A list of messages

    Returns:
        list[RemoveMessage]: The pruned message list
    """
    n_questions = 0

    # if the question limit is 1, just delete all the messages
    if question_limit == 1:
        return [RemoveMessage(msg.id) for msg in messages]

    for i in range(len(messages)-1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            n_questions += 1
        # if the question limit is reached
        if n_questions == question_limit-1:
            # and the last message is a HumanMessage, delete all messages
            # or if there are two consecutive HumanMessages, delete both
            if i == len(messages)-1 \
            or i < len(messages)-1 and isinstance(messages[i+1], HumanMessage):
                i += 1
            return [RemoveMessage(msg.id) for msg in messages[:i]]

    return []

def delete_checkpoints(checkpointer: PostgresSaver, thread_id: str):
    """Deletes all checkpoints for a given thread id

    Args:
        thread_id (str): The thread id
    """
    # Unfortunately, there is no clean way to delete an agent's memory
    # except by deleting its checkpoints, as noted in this github discussion:
    # https://github.com/langchain-ai/langgraph/discussions/912
    with checkpointer._cursor() as cursor:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))

async def async_delete_checkpoints(checkpointer: AsyncPostgresSaver, thread_id: str):
    """Asynchronously deletes all checkpoints for a given thread id

    Args:
        thread_id (str): The thread id
    """
    # Unfortunately, there is no clean way to delete an agent's memory
    # except by deleting its checkpoints, as noted in this github discussion:
    # https://github.com/langchain-ai/langgraph/discussions/912
    async with checkpointer._cursor() as cursor:
        await cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        await cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        await cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Item(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    content: Any = Field(default=None)

class ItemRemove(BaseModel):
    id: uuid.UUID

class ChatTurn(BaseModel):
    id: int # easier for the LLMs to parse
    user_question: str
    ai_response: str
    data: list[Item]

class ChatTurnRemove(BaseModel):
    id: int # easier for the LLMs to parse

ItemUpdate = Item | ItemRemove | list[Item|ItemRemove]

def add_item(existing: Item | list[Item], update: ItemUpdate) -> list[Item]:
    """Merge Items into an existing list, replacing or removing entries by ID.

    Args:
        existing (Item | list[Item]):
            The current `Item` or list of `Item` objects to be merged into.
        update (ItemUpdate):
            An `Item`, `ItemRemove`, or a list containing a mix of `Item` and
            `ItemRemove` objects that specify changes to apply.

    Returns:
        list[Item]:
            The merged list of `Item` objects after applying the updates and deletions.

    Raises:
        ValueError:
            If an `ItemRemove` references an ID that does not exist in `existing`.
    """
    if not isinstance(existing, list):
        existing = [existing]

    if not isinstance(update, list):
        update = [update]

    # copying to avoid modifying the existing list
    merged: list[Item] = existing.copy()

    ids_to_idx = {item.id: idx for idx, item in enumerate(existing)}

    ids_to_remove = set()

    for item in update:
        # if the Item already exists, prepare it for removal or update it
        if (idx := ids_to_idx.get(item.id)) is not None:
            if isinstance(item, ItemRemove):
                ids_to_remove.add(item.id)
            else:
                merged[idx] = item
        # else add it to the list
        else:
            if isinstance(item, ItemRemove):
                raise ValueError(
                    f"Attempted to delete an item with an ID that does not exist ('{item.id}')"
                )
            merged.append(item)

    merged = [item for item in merged if item.id not in ids_to_remove]

    return merged

def add_chat_turn(existing: dict[int, ChatTurn], update: dict[int, ChatTurn|ChatTurnRemove]) -> dict[int, ChatTurn]:
    """Merge chat turns into an existing dictionary, replacing or removing entries by key.

    Args:
        existing (dict[int, ChatTurn]):
            The current dictionary of chat turns to be updated.
        update (dict[int, ChatTurn | ChatTurnRemove]):
            The dictionary containing additions, replacements, or deletions to apply.

    Returns:
        dict[int, ChatTurn]:
            The updated dictionary of chat turns.

    Raises:
        ValueError:
            If a `ChatTurnRemove` references a key not present in `existing`.
    """
    merged = existing.copy()

    for k, v in update.items():
        if isinstance(v, ChatTurnRemove):
            if k not in merged:
                raise ValueError(
                    f"Attempted to delete non-existent chat turn '{k}'"
                )
            del merged[k]
        else:
            merged[k] = v

    return merged

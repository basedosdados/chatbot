import uuid
from typing import Any

from pydantic import BaseModel, Field


class BaseItem(BaseModel):
    content: Any = Field(default=None)
    id: uuid.UUID

class Item(BaseItem):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)

class ItemRemove(BaseItem):
    ...

class BaseChatTurn(BaseModel):
    id: int # easier for the LLMs to parse

class ChatTurn(BaseChatTurn):
    user_question: str
    ai_response: str
    data: list[Item]

class ChatTurnRemove(BaseChatTurn):
    ...

def add_item(existing: Item|list[Item], update: BaseItem|list[BaseItem]) -> list[Item]:
    """Merges two lists of Items, deleting or updating Items by ID.

    Args:
        existing (Item | list[Item]): An Item or a list of Item objects.
        update (BaseItem | list[BaseItem]): A BaseItem or a list of BaseItem objects.

    Returns:
        list[Item]: The merged list.
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

def add_chat_turn(existing: dict[int, ChatTurn], update: dict[int, BaseChatTurn]) -> dict[int, ChatTurn]:
    """Merges two dictionaries of Chat Turns, deleting or updating Chat Turns by key.

    Args:
        existing (dict[int, ChatTurn]): A dictionary of ChatTurn objects.
        update (dict[int, BaseChatTurn]): A dictionary of BaseChatTurn objects.

    Returns:
        dict[int, ChatTurn]: The merged dictionary.
    """
    merged = existing.copy()

    for k, v in update.items():
        if isinstance(v, ChatTurnRemove):
            if k not in merged:
                raise ValueError(
                    f"Attempted to delete non-existent chat turn '{k}'"
                )
            del merged[k]
        elif isinstance(v, ChatTurn):
            merged[k] = v
        else:
            raise TypeError(
                f"Expected `ChatTurn` or `ChatTurnRemove`, got '{type(v)}'"
            )

    return merged

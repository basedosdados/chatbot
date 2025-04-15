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

def add_item(existing: Item|list[Item], update: BaseItem|list[BaseItem]) -> list[Item]:
    """Merges two lists of Items, deleting or updating Items by ID.

    Args:
        existing (list[Item]): A base list of Item objects
        update (list[BaseItem]): A list of Item or ItemRemove objects

    Returns:
        list[Item]: The merged list
    """
    if not isinstance(existing, list):
        existing = [existing]

    if not isinstance(update, list):
        update = [update]

    # copying to avoid modifying the existing list
    merged: list[Item] = existing.copy() or []

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

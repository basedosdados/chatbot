import uuid

import pytest

from chatbot.agents.reducers import Item, ItemRemove, add_item


@pytest.fixture
def item_id() -> uuid.UUID:
    return uuid.uuid4()


# add tests
def test_add_one_item_to_empty():
    existing = []
    update = Item(content=1)
    merged = add_item(existing, update)
    assert merged == [update]

def test_add_one_item_list_to_empty():
    existing = []
    update = [Item(content=1)]
    merged = add_item(existing, update)
    assert merged == update

def test_add_multiple_items_to_empty():
    existing = []
    update = [Item(content=i) for i in range(5)]
    merged = add_item(existing, update)
    assert merged == update

def test_add_one_item_to_one_item_list():
    existing = [Item(content=1)]
    update = Item(content=2)
    merged = add_item(existing, update)
    assert merged == existing+[update]

def test_add_one_item_list_to_one_item_list():
    existing = [Item(content=1)]
    update = [Item(content=2)]
    merged = add_item(existing, update)
    assert merged == existing+update

def test_add_multiple_items_to_one_item_list():
    existing = [Item(content=1)]
    update = [Item(content=i) for i in range(5)]
    merged = add_item(existing, update)
    assert merged == existing+update

def test_add_one_item_to_multiple_items():
    existing = [Item(content=i) for i in range(5)]
    update = Item(content=1)
    merged = add_item(existing, update)
    assert merged == existing+[update]

def test_add_one_item_list_to_multiple_items():
    existing = [Item(content=i) for i in range(5)]
    update = [Item(content=1)]
    merged = add_item(existing, update)
    assert merged == existing+update

def test_add_multiple_items_to_multiple_items():
    existing = [Item(content=i) for i in range(5)]
    update = [Item(content=i) for i in range(5)]
    merged = add_item(existing, update)
    assert merged == existing+update

# update tests
def test_update_one_item_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = Item(content=2, id=item_id)
    merged = add_item(existing, update)
    assert len(merged) == 1
    assert merged[0] == update

def test_update_one_item_list_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [Item(content=2, id=item_id)]
    merged = add_item(existing, update)
    assert len(merged) == 1
    assert merged[0] == update[0]

def test_update_multiple_items_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [Item(content=i) for i in range(4)]
    update.insert(2, Item(content=2, id=item_id))
    merged = add_item(existing, update)
    assert len(merged) == 5
    assert merged[0] == update[2]

def test_update_one_item_in_multiple_items(item_id: uuid.UUID):
    existing = [Item(content=i) for i in range(4)]
    existing.insert(2, Item(content=1, id=item_id))
    update = Item(content=2, id=item_id)
    merged = add_item(existing, update)
    assert len(merged) == 5
    assert merged[2] == update

def test_update_one_item_list_in_multiple_items(item_id: uuid.UUID):
    existing = [Item(content=i) for i in range(4)]
    existing.insert(2, Item(content=1, id=item_id))
    update = [Item(content=2, id=item_id)]
    merged = add_item(existing, update)
    assert len(merged) == 5
    assert merged[2] == update[0]

def test_update_multiple_items_in_multiple_items():
    item1_id = uuid.uuid4()
    item2_id = uuid.uuid4()
    item3_id = uuid.uuid4()

    existing = [
        Item(content=1, id=item1_id),
        Item(content=2),
        Item(content=3, id=item2_id),
        Item(content=4),
        Item(content=5, id=item3_id)
    ]

    update = [
        Item(content=2, id=item1_id),
        Item(content=4, id=item2_id),
        Item(content=6, id=item3_id),
        Item(content=7)
    ]

    merged = add_item(existing, update)

    assert len(merged) == 6
    assert merged[0] == update[0]
    assert merged[2] == update[1]
    assert merged[4] == update[2]

# remove tests
def test_remove_one_item_in_empty_list(item_id: uuid.UUID):
    existing = []
    update = ItemRemove(id=item_id)
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

def test_remove_one_item_list_in_empty_list(item_id: uuid.UUID):
    existing = []
    update = [ItemRemove(id=item_id)]
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

def test_remove_multiple_items_in_empty_list():
    existing = []
    update = [ItemRemove(id=uuid.uuid4()) for _ in range(5)]
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

def test_remove_one_item_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = ItemRemove(id=item_id)
    merged = add_item(existing, update)
    assert merged == []

def test_remove_one_item_list_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [ItemRemove(id=item_id)]
    merged = add_item(existing, update)
    assert merged == []

def test_remove_multiple_items_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [ItemRemove(id=item_id)] + [ItemRemove(id=uuid.uuid4()) for _ in range(4)]
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

def test_remove_one_item_in_multiple_items_list(item_id: uuid.UUID):
    existing = [Item(content=i) for i in range(4)]
    existing.insert(2, Item(content=1, id=item_id))
    update = ItemRemove(content=2, id=item_id)
    merged = add_item(existing, update)
    assert len(merged) == 4

def test_remove_one_item_list_in_multiple_items_list(item_id: uuid.UUID):
    existing = [Item(content=i) for i in range(4)]
    existing.insert(2, Item(content=1, id=item_id))
    update = [ItemRemove(content=2, id=item_id)]
    merged = add_item(existing, update)
    assert len(merged) == 4

def test_remove_multiple_items_in_multiple_items_list():
    item1_id = uuid.uuid4()
    item2_id = uuid.uuid4()
    item3_id = uuid.uuid4()

    existing = [
        Item(content=1, id=item1_id),
        Item(content=2),
        Item(content=3, id=item2_id),
        Item(content=4),
        Item(content=5, id=item3_id)
    ]

    update = [
        ItemRemove(content=2, id=item1_id),
        ItemRemove(content=4, id=item2_id),
        ItemRemove(content=6, id=item3_id),
    ]

    merged = add_item(existing, update)

    assert len(merged) == 2

# add, update, remove at the same time
def test_add_update_remove_items_in_multiple_items_list():
    item1_id = uuid.uuid4()
    item2_id = uuid.uuid4()
    item3_id = uuid.uuid4()

    existing = [
        Item(content=1, id=item1_id),
        Item(content=2),
        Item(content=3, id=item2_id),
        Item(content=4),
        Item(content=5, id=item3_id)
    ]

    update = [
        Item(content=2, id=item1_id),
        ItemRemove(id=item2_id),
        ItemRemove(id=item3_id),
        Item(content=7),
    ]

    merged = add_item(existing, update)

    assert len(merged) == 4
    assert merged[0] == update[0]

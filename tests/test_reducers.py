import uuid

import pytest

from chatbot.agents.reducers import (ChatTurn, ChatTurnRemove, Item,
                                     ItemRemove, add_chat_turn, add_item)


# ============================================
# ==              Item Reducer              ==
# ============================================
@pytest.fixture
def item_id() -> uuid.UUID:
    """Returns an Item uuid."""
    return uuid.uuid4()

@pytest.fixture
def existing_items() -> list[Item]:
    """Returns a list of 5 Items."""
    return [Item(content=i+1) for i in range(5)]

# =========================== add tests ===========================
@pytest.mark.parametrize("update", [
    (Item(content=1)),
    ([Item(content=1)]),
    ([Item(content=i+1) for i in range(5)]),
])
def test_add_item_to_empty(update: Item | list[Item]):
    existing = []
    merged = add_item(existing, update)
    if not isinstance(update, list):
        update = [update]
    assert merged == update

@pytest.mark.parametrize("update", [
    (Item(content=1)),
    ([Item(content=1)]),
    ([Item(content=i+1) for i in range(5)]),
])
def test_add_item_to_existing(update: Item | list[Item], existing_items: list[Item]):
    merged = add_item(existing_items, update)
    if not isinstance(update, list):
        update = [update]
    assert merged == existing_items + update

@pytest.mark.parametrize("update", [
    (Item(content=1)),
    ([Item(content=1)]),
    ([Item(content=i+1) for i in range(5)]),
])
def test_add_item_to_item(update: Item | list[Item]):
    existing = Item(content=1)
    merged = add_item(existing, update)
    if not isinstance(update, list):
        update = [update]
    assert merged == [existing] + update

# =========================== update tests ===========================
@pytest.mark.parametrize("update_form", ["single", "list"])
def test_update_item_in_item(item_id: uuid.UUID, update_form: str):
    existing = Item(content=1, id=item_id)
    update_item = Item(content=2, id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing, update)
    assert len(merged) == 1
    assert merged[0] == update_item

@pytest.mark.parametrize("update_form", ["single", "list"])
def test_update_one_item_in_one_item_list(item_id: uuid.UUID, update_form: str):
    existing = [Item(content=1, id=item_id)]
    update_item = Item(content=2, id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing, update)
    assert len(merged) == 1
    assert merged[0] == update_item

def test_update_multiple_items_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [Item(content=i+1) for i in range(4)]
    update.insert(2, Item(content=2, id=item_id))
    merged = add_item(existing, update)
    assert len(merged) == 5
    assert merged[0] == update[2]

@pytest.mark.parametrize("update_form", ["single", "list"])
def test_update_one_item_in_multiple_items_list(item_id: uuid.UUID, existing_items: list[Item], update_form: str):
    existing_items.insert(2, Item(content=1, id=item_id))
    update_item = Item(content=2, id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing_items, update)
    assert len(merged) == 6
    assert merged[2] == update_item

def test_update_multiple_items_in_multiple_items_list():
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

# =========================== remove tests ===========================
@pytest.mark.parametrize("update", [
    (ItemRemove(id=uuid.uuid4())),
    ([ItemRemove(id=uuid.uuid4())]),
    ([ItemRemove(id=uuid.uuid4()) for _ in range(5)]),
])
def test_remove_item_in_empty_list(update: ItemRemove | list[ItemRemove]):
    existing = []
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

@pytest.mark.parametrize("update_form", ["single", "list"])
def test_remove_item_in_item(item_id: uuid.UUID, update_form: str):
    existing = Item(content=1, id=item_id)
    update_item = ItemRemove(id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing, update)
    assert merged == []

@pytest.mark.parametrize("update_form", ["single", "list"])
def test_remove_one_item_in_one_item_list(item_id: uuid.UUID, update_form: str):
    existing = [Item(content=1, id=item_id)]
    update_item = ItemRemove(id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing, update)
    assert merged == []

def test_remove_multiple_items_in_one_item_list(item_id: uuid.UUID):
    existing = [Item(content=1, id=item_id)]
    update = [ItemRemove(id=item_id)] + [ItemRemove(id=uuid.uuid4()) for _ in range(4)]
    with pytest.raises(ValueError):
        _ = add_item(existing, update)

@pytest.mark.parametrize("update_form", ["single", "list"])
def test_remove_one_item_in_multiple_items_list(item_id: uuid.UUID, existing_items: list[Item], update_form: str):
    existing_items.insert(2, Item(content=1, id=item_id))
    update_item = ItemRemove(id=item_id)
    update = [update_item] if update_form == "list" else update_item
    merged = add_item(existing_items, update)
    assert len(merged) == 5
    assert not any(item.id == item_id for item in merged)

def test_remove_multiple_items_in_multiple_items_list():
    item1_id = uuid.uuid4()
    item3_id = uuid.uuid4()
    item5_id = uuid.uuid4()

    existing = [
        Item(content=1, id=item1_id),
        Item(content=2),
        Item(content=3, id=item3_id),
        Item(content=4),
        Item(content=5, id=item5_id)
    ]

    update = [
        ItemRemove(id=item1_id),
        ItemRemove(id=item3_id),
        ItemRemove(id=item5_id),
    ]

    merged = add_item(existing, update)

    assert len(merged) == 2

# ================== add, update, remove at the same time ==================
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

# ================================================
# ==              ChatTurn Reducer              ==
# ================================================
@pytest.fixture
def mock_chat_turn() -> ChatTurn:
    """Returns a ChatTurn with id=1."""
    return ChatTurn(
        id=1,
        user_question="mock question",
        ai_response="mock response",
        data=[],
    )

@pytest.fixture
def mock_chat_turns() -> dict[int, ChatTurn]:
    """Returns a dict of two ChatTurns with id=1 and id=2."""
    return {
        i+1: ChatTurn(
            id=i+1,
            user_question="mock question",
            ai_response="mock response",
            data=[],
        ) for i in range(2)
    }

# =========================== add tests ===========================
def test_add_chat_turn_to_empty(mock_chat_turn: ChatTurn):
    existing = {}
    update = {mock_chat_turn.id: mock_chat_turn}
    merged = add_chat_turn(existing, update)
    assert merged == update

def test_add_multiple_chat_turns_to_empty(mock_chat_turns: dict[int, ChatTurn]):
    existing = {}
    update = mock_chat_turns
    merged = add_chat_turn(existing, update)
    assert merged == update

def test_add_chat_turn_to_existing(mock_chat_turn: ChatTurn):
    existing = {
        2: ChatTurn(
            id=2,
            user_question="another question",
            ai_response="another response",
            data=[]
        )
    }
    update = {mock_chat_turn.id: mock_chat_turn}
    merged = add_chat_turn(existing, update)
    assert len(merged) == 2
    assert merged[mock_chat_turn.id] == mock_chat_turn

def test_add_and_update_chat_turns(mock_chat_turns: dict[int, ChatTurn]):
    existing = {
        1: ChatTurn(
            id=1,
            user_question="one question",
            ai_response="one response",
            data=[]
        ),
        3: ChatTurn(
            id=3,
            user_question="another question",
            ai_response="another response",
            data=[]
        ),
    }
    # mock_chat_turns has ids 1 and 2.
    # expecting to update id 1, add id 2, and keep id 3
    update = mock_chat_turns
    merged = add_chat_turn(existing, update)

    expected = existing.copy()
    expected.update(update)

    assert merged == expected
    assert len(merged) == 3


def test_add_chat_turn_with_empty_update(mock_chat_turns: dict[int, ChatTurn]):
    existing = mock_chat_turns
    update = {}
    merged = add_chat_turn(existing, update)
    assert merged == existing

# =========================== update tests ===========================
def test_update_one_chat_turn(mock_chat_turn: ChatTurn):
    existing = {mock_chat_turn.id: mock_chat_turn}

    chat_turn_copy = mock_chat_turn.model_copy()
    chat_turn_copy.user_question = "mock update"

    update = {chat_turn_copy.id: chat_turn_copy}
    merged = add_chat_turn(existing, update)

    assert merged == update

def test_update_multiple_chat_turns(mock_chat_turns: dict[int, ChatTurn]):
    existing = mock_chat_turns

    update = {}
    for _, chat_turn in existing.items():
        chat_turn_copy = chat_turn.model_copy()
        chat_turn_copy.user_question = "mock update"
        update[chat_turn_copy.id] = chat_turn_copy

    merged = add_chat_turn(existing, update)

    assert merged == update

# =========================== remove tests ===========================
def test_remove_the_only_chat_turn(mock_chat_turn: ChatTurn):
    existing = {mock_chat_turn.id: mock_chat_turn}
    update = {mock_chat_turn.id: ChatTurnRemove(id=mock_chat_turn.id)}

    merged = add_chat_turn(existing, update)

    assert merged == {}

def test_remove_all_chat_turns(mock_chat_turns: dict[int, ChatTurn]):
    existing = mock_chat_turns
    update = {k: ChatTurnRemove(id=k) for k in existing.keys()}
    merged = add_chat_turn(existing, update)
    assert merged == {}

def test_remove_one_chat_turn_in_existing(mock_chat_turns: dict[int, ChatTurn]):
    existing = mock_chat_turns
    remove_key, *_ = existing.keys()
    update = {remove_key: ChatTurnRemove(id=remove_key)}

    merged = add_chat_turn(existing, update)

    assert len(merged) == len(existing) - 1
    assert remove_key not in merged

def test_remove_invalid():
    existing = {}
    chat_turn_remove = ChatTurnRemove(id=2)
    update = {chat_turn_remove.id: chat_turn_remove}

    with pytest.raises(ValueError):
        _ = add_chat_turn(existing, update)

# ================== add, update, remove at the same time ==================
def test_add_update_remove_chat_turns(mock_chat_turns: dict[int, ChatTurn]):
    # mock_chat_turns has ids 1 and 2.
    # expecting to remove id 1, update id 2, and add id 3
    existing = mock_chat_turns
    remove_key, update_key = existing.keys()

    update_chat_turn = ChatTurn(
        id=update_key,
        user_question="mock update",
        ai_response="mock response",
        data=[]
    )

    new_chat_turn = ChatTurn(
        id=3,
        user_question="mock question",
        ai_response="mock response",
        data=[]
    )

    update = {
        new_chat_turn.id: new_chat_turn,
        update_key: update_chat_turn,
        remove_key: ChatTurnRemove(id=remove_key)
    }

    merged = add_chat_turn(existing, update)

    assert new_chat_turn.id in merged
    assert update_key in merged
    assert remove_key not in merged
    assert merged[new_chat_turn.id] == new_chat_turn
    assert merged[update_key] == update_chat_turn
    assert len(merged) == 2

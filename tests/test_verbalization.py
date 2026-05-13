import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_recsys.data.datasets.base import UserHistory
from llm_recsys.data.verbalization.templates import (
    RatingHistoryVerbalizer, MinimalVerbalizer, get_verbalizer
)


ITEM_META = {
    "m1": {"title": "The Matrix", "genres": "Action|Sci-Fi"},
    "m2": {"title": "Inception", "genres": "Sci-Fi|Thriller"},
    "m3": {"title": "Interstellar", "genres": "Drama|Sci-Fi"},
}


def make_history(item_ids=None, ratings=None, review_texts=None):
    ids = item_ids or ["m1", "m2"]
    extra = {"review_texts": review_texts} if review_texts else None
    return UserHistory(
        user_id="u1",
        item_ids=ids,
        ratings=ratings,
        timestamps=list(range(len(ids))),
        item_metadata=ITEM_META,
        extra=extra,
    )


# ── RatingHistoryVerbalizer ──────────────────────────────────────────────────

def test_rating_verbalizer_contains_title():
    v = RatingHistoryVerbalizer()
    text = v.verbalize(make_history(["m1", "m2"]))
    assert "The Matrix" in text
    assert "Inception" in text


def test_rating_verbalizer_contains_genre():
    v = RatingHistoryVerbalizer()
    text = v.verbalize(make_history(["m1"]))
    assert "Action" in text or "Sci-Fi" in text


def test_rating_verbalizer_contains_rating():
    v = RatingHistoryVerbalizer()
    text = v.verbalize(make_history(["m1", "m2"], ratings=[5.0, 3.0]))
    assert "5/5" in text
    assert "3/5" in text


def test_rating_verbalizer_appends_target():
    v = RatingHistoryVerbalizer()
    text_no_target = v.verbalize(make_history(["m1"]))
    text_with_target = v.verbalize(make_history(["m1"]), target_item_id="m2")
    assert "Inception" not in text_no_target
    assert "Inception" in text_with_target


def test_rating_verbalizer_prompt_ends_with_next():
    v = RatingHistoryVerbalizer()
    text = v.verbalize(make_history(["m1"]))
    assert "Next:" in text


def test_rating_verbalizer_review_text():
    v = RatingHistoryVerbalizer()
    text = v.verbalize(make_history(["m1"], review_texts=["Great film!"]))
    assert "Great film!" in text


def test_rating_verbalizer_unknown_item_uses_id():
    v = RatingHistoryVerbalizer()
    h = make_history(["unknown_id"])
    text = v.verbalize(h)
    assert "unknown_id" in text


# ── MinimalVerbalizer ────────────────────────────────────────────────────────

def test_minimal_verbalizer_arrow_chain():
    v = MinimalVerbalizer()
    text = v.verbalize(make_history(["m1", "m2", "m3"]))
    assert "The Matrix → Inception → Interstellar" in text


def test_minimal_verbalizer_no_ratings():
    v = MinimalVerbalizer()
    text = v.verbalize(make_history(["m1"], ratings=[5.0]))
    assert "5/5" not in text


def test_minimal_verbalizer_appends_target():
    v = MinimalVerbalizer()
    text = v.verbalize(make_history(["m1"]), target_item_id="m3")
    assert "Interstellar" in text


# ── Registry ─────────────────────────────────────────────────────────────────

def test_get_verbalizer_rating_history():
    v = get_verbalizer("rating_history")
    assert isinstance(v, RatingHistoryVerbalizer)


def test_get_verbalizer_minimal():
    v = get_verbalizer("minimal")
    assert isinstance(v, MinimalVerbalizer)


def test_get_verbalizer_unknown_raises():
    try:
        get_verbalizer("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

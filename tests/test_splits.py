import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from types import SimpleNamespace
from llm_recsys.data.datasets.base import UserHistory
from llm_recsys.data.splits import split_users, _leave_one_out, _temporal_split


def make_user(item_ids, timestamps=None, ratings=None):
    meta = {iid: {"title": f"Movie {iid}"} for iid in item_ids}
    return UserHistory(
        user_id="u1",
        item_ids=item_ids,
        ratings=ratings,
        timestamps=timestamps or list(range(len(item_ids))),
        item_metadata=meta,
    )


def make_users(n=5):
    return {
        f"u{i}": make_user(
            item_ids=[f"item_{i}_{j}" for j in range(5)],
            timestamps=list(range(i * 10, i * 10 + 5)),
            ratings=[4.0, 3.0, 5.0, 2.0, 4.0],
        )
        for i in range(n)
    }


# ── leave-one-out ────────────────────────────────────────────────────────────

def test_leave_one_out_split_sizes():
    users = make_users(5)
    split = _leave_one_out(users)
    assert len(split.train) == len(split.val) == len(split.test) == 5


def test_leave_one_out_targets():
    users = {"u1": make_user(["a", "b", "c", "d", "e"])}
    split = _leave_one_out(users)
    assert split.train["u1"].item_ids == ["a", "b", "c"]
    assert split.val["u1"].item_ids == ["a", "b", "c", "d"]
    assert split.test["u1"].item_ids == ["a", "b", "c", "d", "e"]


def test_leave_one_out_ratings_sliced():
    users = {"u1": make_user(["a", "b", "c", "d"], ratings=[1.0, 2.0, 3.0, 4.0])}
    split = _leave_one_out(users)
    assert split.train["u1"].ratings == [1.0, 2.0]
    assert split.val["u1"].ratings == [1.0, 2.0, 3.0]


def test_leave_one_out_short_user_excluded():
    users = {
        "u_short": make_user(["a", "b"]),  # needs >= 3
        "u_ok": make_user(["a", "b", "c"]),
    }
    split = _leave_one_out(users)
    assert "u_short" not in split.train
    assert "u_ok" in split.train


def test_leave_one_out_no_data_leakage():
    users = {"u1": make_user(["a", "b", "c", "d", "e"])}
    split = _leave_one_out(users)
    # test target must not appear in training context
    test_target = split.test["u1"].item_ids[-1]
    assert test_target not in split.train["u1"].item_ids


# ── temporal split ───────────────────────────────────────────────────────────

def test_temporal_split_no_overlap():
    users = make_users(10)
    cfg = SimpleNamespace(strategy="temporal", val_ratio=0.1, test_ratio=0.1)
    split = split_users(users, cfg)
    # Check that timestamps in train < val < test (no overlap) for any user that appears in all
    for uid in split.train:
        if uid in split.val and uid in split.test:
            assert max(split.train[uid].timestamps) <= min(split.val[uid].timestamps)
            assert max(split.val[uid].timestamps) <= min(split.test[uid].timestamps)


def test_temporal_split_covers_all_items():
    users = {"u1": make_user(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
                              timestamps=list(range(10)))}
    cfg = SimpleNamespace(strategy="temporal", val_ratio=0.1, test_ratio=0.1)
    split = split_users(users, cfg)
    all_ids = set()
    for part in [split.train, split.val, split.test]:
        if "u1" in part:
            all_ids.update(part["u1"].item_ids)
    assert all_ids == {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}


# ── split_users dispatch ─────────────────────────────────────────────────────

def test_split_users_unknown_strategy():
    import pytest
    users = make_users(3)
    cfg = SimpleNamespace(strategy="bad_strategy", val_ratio=0.1, test_ratio=0.1)
    try:
        split_users(users, cfg)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

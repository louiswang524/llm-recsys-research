from typing import NamedTuple
from .datasets.base import UserHistory


class SplitResult(NamedTuple):
    train: dict[str, UserHistory]
    val: dict[str, UserHistory]
    test: dict[str, UserHistory]


def split_users(users: dict[str, UserHistory], cfg) -> SplitResult:
    if cfg.strategy == "leave_one_out":
        return _leave_one_out(users)
    elif cfg.strategy == "temporal":
        return _temporal_split(users, cfg.val_ratio, cfg.test_ratio)
    raise ValueError(f"Unknown split strategy: {cfg.strategy}")


def _leave_one_out(users: dict[str, UserHistory]) -> SplitResult:
    train, val, test = {}, {}, {}
    for uid, h in users.items():
        if len(h.item_ids) < 3:
            continue
        train[uid] = _slice_history(h, end=-2)
        val[uid] = _slice_history(h, end=-1)
        test[uid] = h  # full history; evaluator will use last item as target
    return SplitResult(train=train, val=val, test=test)


def _temporal_split(users: dict[str, UserHistory], val_ratio: float, test_ratio: float) -> SplitResult:
    all_ts = sorted(
        ts
        for h in users.values()
        for ts in (h.timestamps or [])
    )
    n = len(all_ts)
    val_cut = all_ts[int(n * (1 - val_ratio - test_ratio))]
    test_cut = all_ts[int(n * (1 - test_ratio))]

    train, val, test = {}, {}, {}
    for uid, h in users.items():
        ts_list = h.timestamps or [0] * len(h.item_ids)
        t = _filter_by_mask(h, [t < val_cut for t in ts_list])
        v = _filter_by_mask(h, [val_cut <= t < test_cut for t in ts_list])
        te = _filter_by_mask(h, [t >= test_cut for t in ts_list])
        if t:
            train[uid] = t
        if v:
            val[uid] = v
        if te:
            test[uid] = te

    return SplitResult(train=train, val=val, test=test)


def _slice_history(h: UserHistory, end: int) -> UserHistory:
    return UserHistory(
        user_id=h.user_id,
        item_ids=h.item_ids[:end],
        ratings=h.ratings[:end] if h.ratings else None,
        timestamps=h.timestamps[:end] if h.timestamps else None,
        item_metadata=h.item_metadata,
        extra=h.extra,
    )


def _filter_by_mask(h: UserHistory, mask: list[bool]) -> UserHistory | None:
    idxs = [i for i, m in enumerate(mask) if m]
    if not idxs:
        return None
    return UserHistory(
        user_id=h.user_id,
        item_ids=[h.item_ids[i] for i in idxs],
        ratings=[h.ratings[i] for i in idxs] if h.ratings else None,
        timestamps=[h.timestamps[i] for i in idxs] if h.timestamps else None,
        item_metadata=h.item_metadata,
        extra=None,
    )

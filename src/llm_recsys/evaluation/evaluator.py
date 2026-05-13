import random
from tqdm import tqdm
from omegaconf import DictConfig
from .metrics import compute_metrics
from .candidate_scoring import CandidateScorer
from ..data.datasets.base import UserHistory
from ..data.verbalization.base import BaseVerbalizer
from ..data.splits import _slice_history


class RecEvaluator:
    def __init__(self, cfg: DictConfig, test_users: dict[str, UserHistory],
                 all_item_ids: list[str], verbalizer: BaseVerbalizer, scorer: CandidateScorer):
        self.cfg = cfg
        self.test_users = test_users
        self.all_item_ids = all_item_ids
        self.verbalizer = verbalizer
        self.scorer = scorer
        self.k_values = list(cfg.eval.k_values)
        self.num_neg = cfg.eval.num_neg_samples

    def evaluate(self, model) -> dict[str, float]:
        ranked_lists, targets = [], []

        for uid, history in tqdm(self.test_users.items(), desc="Evaluating"):
            if len(history.item_ids) < 2:
                continue
            target_item = history.item_ids[-1]
            context = _slice_history(history, end=-1)
            context_text = self.verbalizer.verbalize(context)

            negatives = random.sample(
                [i for i in self.all_item_ids if i != target_item],
                k=min(self.num_neg, len(self.all_item_ids) - 1),
            )
            candidates = [target_item] + negatives
            random.shuffle(candidates)

            ranked = self.scorer.rank_candidates(model, context_text, candidates)
            ranked_lists.append(ranked)
            targets.append(target_item)

        return compute_metrics(ranked_lists, targets, self.k_values)

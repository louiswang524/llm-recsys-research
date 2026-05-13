from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class RecMetricsCallback(TrainerCallback):
    """Injects full RecSys metrics (HR@K, NDCG@K) into eval results."""

    def __init__(self, evaluator, eval_k: list[int] | None = None):
        self.evaluator = evaluator
        self.eval_k = eval_k or [10, 20]

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        if model is None:
            return
        metrics = self.evaluator.evaluate(model)
        if state.log_history:
            state.log_history[-1].update({f"eval_{k}": v for k, v in metrics.items()})
        else:
            state.log_history.append({f"eval_{k}": v for k, v in metrics.items()})

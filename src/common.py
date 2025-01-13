import numpy as np
from transformers.trainer_utils import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, any]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}

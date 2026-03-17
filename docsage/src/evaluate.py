import json
import os
from pathlib import Path
from typing import Dict, List, Literal

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .pipeline import chat_once


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


EvalCollection = Literal["naive", "tuned"]


def load_eval_dataset() -> List[Dict]:
    here = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(here, "data", "eval_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_eval_for_collection(collection: EvalCollection) -> Dict:
    raw = load_eval_dataset()

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for row in raw:
        q = row["question"]
        gt = row["ground_truth"]
        result = chat_once(q, collection)
        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in result["citations"]])
        ground_truths.append(gt)

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    ragas_result = evaluate(ds, metrics=metrics)
    return ragas_result.to_pandas().to_dict(orient="list")


if __name__ == "__main__":
    naive_scores = run_eval_for_collection("naive")
    tuned_scores = run_eval_for_collection("tuned")
    print("Naive metrics:", naive_scores)
    print("Tuned metrics:", tuned_scores)


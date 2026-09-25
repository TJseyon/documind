"""
Runs the hand-built Q&A test set through the live pipeline, logs every
input/output pair, scores automatically what can be scored automatically,
and leaves a manual_score column for the judgment calls that can't be
automated ("is this summary actually accurate?").

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --testset eval/qa_testset.json

Requires documents to already be ingested (see scripts/ingest_sample_docs.py)
and ANTHROPIC_API_KEY set in the environment / .env file.

Outputs:
    eval/results/run_<timestamp>.json  -- full input/output log, machine-readable
    eval/results/run_<timestamp>.csv   -- the "spreadsheet of rows you scored
                                           yourself" -- open this in Excel/
                                           Sheets and fill in manual_score.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import EvalCaseResult, aggregate_metrics  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.dependencies import get_query_pipeline  # noqa: E402


def run(testset_path: str) -> None:
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)

    pipeline = get_query_pipeline()
    results = []

    for case in testset["cases"]:
        response = pipeline.answer(case["question"])
        retrieved_docs = [c.source_filename for c in response.citations]

        results.append(
            EvalCaseResult(
                case_id=case["id"],
                question=case["question"],
                expected_source_document=case["expected_source_document"],
                expected_answer_substring=case["expected_answer_substring"],
                retrieved_documents=retrieved_docs,
                answer=response.answer,
                grounded=response.grounded,
                unsupported_claims=response.unsupported_claims,
                latency_ms=response.latency_ms,
            )
        )

    metrics = aggregate_metrics(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / f"run_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "cases": [vars(r) for r in results]},
            f,
            indent=2,
        )

    csv_path = results_dir / f"run_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id", "question", "expected_source_document", "retrieved_documents",
                "retrieval_hit", "expected_answer_substring", "answer", "contains_answer",
                "grounded", "unsupported_claims", "latency_ms", "manual_score (fill in: correct/partial/wrong)",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.case_id, r.question, r.expected_source_document, ";".join(r.retrieved_documents),
                    r.retrieval_hit, r.expected_answer_substring, r.answer, r.contains_answer,
                    r.grounded, ";".join(r.unsupported_claims), r.latency_ms, "",
                ]
            )

    print(json.dumps(metrics, indent=2))
    print(f"\nFull log:  {json_path}")
    print(f"Spreadsheet: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default=str(Path(__file__).resolve().parent / "qa_testset.json"))
    args = parser.parse_args()
    run(args.testset)

"""
This is deliberately NOT in tests/unit/. That separation is the point:

  - Unit tests check deterministic code: given fixed input, is the output
    exactly what's expected? They run in milliseconds, need no API key, and
    a failure means "you broke something."
  - This eval test checks a fuzzier property: has overall pipeline quality
    regressed against a baseline, on a held-out question set, when the real
    LLM and real embedding model are in the loop? It costs real API calls,
    takes real time, and a failure means "something got worse" -- it might
    be a code bug, a prompt change, a model swap, or a flaky LLM response,
    and figuring out which is a judgment call a unit test can't make for you.

Both are useful. Neither is a substitute for the other. Run this one
explicitly (`pytest tests/eval/ -m eval`), not as part of the normal
fast unit-test loop -- it needs ANTHROPIC_API_KEY set and downloaded models,
so it's skipped automatically when those aren't available (e.g. in CI
without secrets configured).
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval

BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "baseline_score.json"


def _has_llm_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.skipif(not _has_llm_credentials(), reason="ANTHROPIC_API_KEY not set; skipping live eval regression test")
def test_eval_score_has_not_regressed():
    from eval.metrics import aggregate_metrics
    from eval.run_eval import run as run_eval_and_write_results

    # Running this pulls in the real embedder/reranker/LLM via app.dependencies,
    # so it also requires documents to have been ingested first (see
    # scripts/ingest_sample_docs.py) and the models to be downloaded.
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    testset_path = Path(__file__).resolve().parent.parent.parent / "eval" / "qa_testset.json"
    run_eval_and_write_results(str(testset_path))

    results_dir = Path(__file__).resolve().parent.parent.parent / "eval" / "results"
    latest = max(results_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    with open(latest, "r", encoding="utf-8") as f:
        current = json.load(f)["metrics"]

    tolerance = baseline.get("tolerance", 0.05)
    for metric in ("retrieval_hit_rate", "contains_answer_rate", "groundedness_rate"):
        assert current[metric] >= baseline[metric] - tolerance, (
            f"{metric} regressed: {current[metric]} is more than {tolerance} below "
            f"baseline {baseline[metric]}. See {latest} for full detail."
        )

"""
Convenience script: POSTs every file in sample_docs/ to a running instance's
/ingest endpoint. Run this after `docker compose up` (or a local uvicorn run)
so the eval harness and manual /query testing have something to work with.

Usage:
    python scripts/ingest_sample_docs.py
    python scripts/ingest_sample_docs.py --base-url http://localhost:8000
"""
import argparse
import sys
from pathlib import Path

import httpx

SAMPLE_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

# Skip the intentionally-broken edge-case files in the default demo run --
# ingest them explicitly if you want to see the error responses.
SKIP = {"empty_document.txt"}


def main(base_url: str) -> None:
    files = [p for p in SAMPLE_DOCS_DIR.iterdir() if p.is_file() and p.name not in SKIP]
    if not files:
        print(f"No sample docs found in {SAMPLE_DOCS_DIR}", file=sys.stderr)
        sys.exit(1)

    with httpx.Client(timeout=120) as client:
        for path in files:
            with open(path, "rb") as f:
                resp = client.post(f"{base_url}/ingest", files={"file": (path.name, f)})
            if resp.status_code == 201:
                body = resp.json()
                print(f"OK  {path.name}: {body['chunk_count']} chunks, warnings={body['warnings']}")
            else:
                print(f"FAIL {path.name}: [{resp.status_code}] {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    main(args.base_url)

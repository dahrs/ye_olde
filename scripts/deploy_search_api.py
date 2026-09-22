"""Pushes search_api/ to the Hugging Face Space that runs it (spec §10).

Run by .github/workflows/deploy-search-api.yml on every push to main that
touches search_api/. Needs HF_TOKEN (a write-scoped access token, stored as
a GitHub Actions secret) and HF_SPACE_REPO_ID (e.g. "dahrs/ye-olde-api",
stored as a GitHub Actions repo *variable* — not a secret, it isn't
sensitive) in the environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_REPO_ID")
    if not token or not repo_id:
        print("HF_TOKEN and HF_SPACE_REPO_ID must both be set", file=sys.stderr)
        raise SystemExit(1)

    HfApi(token=token).upload_folder(
        folder_path=str(Path(__file__).resolve().parent.parent / "search_api"),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy search_api from GitHub",
    )


if __name__ == "__main__":
    main()

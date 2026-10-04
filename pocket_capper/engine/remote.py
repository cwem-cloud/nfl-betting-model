"""Start runs on GitHub Actions from the dashboard (fast and free), instead of on a tiny web host.

Needs a fine-grained GitHub token with "Actions: Read and write" on this repo, set as
GH_DISPATCH_TOKEN where the dashboard runs. GH_REPO defaults to this repo.
"""
from __future__ import annotations

import requests

from pocket_capper.settings import secret

WORKFLOW = "pocket_capper.yml"
API = "https://api.github.com"


def enabled() -> bool:
    return bool(secret("GH_DISPATCH_TOKEN"))


def _repo() -> str:
    return secret("GH_REPO", "cwem-cloud/nfl-betting-model")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {secret('GH_DISPATCH_TOKEN')}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def dispatch(sports: list[str], include_props: bool, task: str = "slate") -> None:
    r = requests.post(
        f"{API}/repos/{_repo()}/actions/workflows/{WORKFLOW}/dispatches",
        headers=_headers(),
        json={"ref": "main", "inputs": {"task": task, "sports": ",".join(sports), "props": str(include_props).lower()}},
        timeout=20,
    )
    if r.status_code != 204:
        raise RuntimeError(f"GitHub refused the run ({r.status_code}): {r.text[:300]}")


def latest_runs(n: int = 3) -> list[dict]:
    r = requests.get(
        f"{API}/repos/{_repo()}/actions/workflows/{WORKFLOW}/runs",
        headers=_headers(), params={"per_page": n}, timeout=20,
    )
    r.raise_for_status()
    return [
        {"status": x["status"], "conclusion": x.get("conclusion"), "event": x["event"],
         "created_at": x["created_at"], "url": x["html_url"]}
        for x in r.json().get("workflow_runs", [])
    ]

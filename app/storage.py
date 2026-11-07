"""Private, opt-in storage of analysis results for the project owner.

Two backends with the same interface:
  * GitHubStore - a PRIVATE GitHub repository (e.g. NowfaL1N/LungLocAI-data), written through the
                  GitHub REST API with a fine-grained token that can only touch that one repo.
  * LocalStore  - a folder on disk (development / tests).

Configured with environment variables:
  DATA_REPO=owner/LungLocAI-data   DATA_REPO_TOKEN=github_pat_...   (GitHub backend)
  DATA_DIR=/path/to/folder                                          (local backend)

Layout inside the store:
  cases/<case_id>/result.json        anonymised result (no file names, no DICOM UIDs)
  cases/<case_id>/table.csv          every candidate + 33 features + XGBoost output
  cases/<case_id>/report.pdf         anonymised PDF report
  cases/<case_id>/nodule_<n>.png     3-plane close-ups (for labelling)
  index.csv                          one line per stored case
  labels.csv                         owner's confirmed labels (case_id, candidate_id, label, ...)
  training/base_table.csv            optional: the LungLocAI / LUNA16 feature table used for retraining
  models/                            retrained models (+ models/history/)
Only the analysis results are stored - never the uploaded CT scan itself.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

INDEX_FIELDS = ["case_id", "stored_at", "scan_type", "n_candidates", "n_nodules", "n_suspicious", "max_p_malignant",
                "slice_mm", "model"]
LABEL_FIELDS = ["case_id", "candidate_id", "label", "labelled_at"]
VALID_LABELS = ("malignant", "benign", "not_nodule", "unsure")

_LOCK = threading.RLock()


def anon_id(text: str) -> str:
    return hashlib.sha256(("lunglocai:" + str(text)).encode()).hexdigest()[:12]


class BaseStore:
    name = "base"

    # --- primitives (implemented by subclasses) -------------------------------------------
    def read(self, path: str) -> bytes | None: ...
    def write(self, path: str, data: bytes, message: str) -> None: ...
    def write_many(self, files: dict[str, bytes], message: str) -> None:
        for p, d in files.items():
            self.write(p, d, message)
    def list(self, prefix: str) -> list[str]: ...

    # --- helpers -----------------------------------------------------------------------------
    def read_csv(self, path: str) -> list[dict]:
        raw = self.read(path)
        if not raw:
            return []
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))

    def _csv_bytes(self, rows: list[dict], fields: list[str]) -> bytes:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue().encode("utf-8")

    def save_case(self, case_id: str, files: dict[str, bytes], index_row: dict) -> None:
        with _LOCK:
            payload = {f"cases/{case_id}/{k}": v for k, v in files.items()}
            idx = [r for r in self.read_csv("index.csv") if r.get("case_id") != case_id]
            idx.append(index_row)
            payload["index.csv"] = self._csv_bytes(idx, INDEX_FIELDS)
            self.write_many(payload, f"Add case {case_id}")

    def cases(self) -> list[dict]:
        return self.read_csv("index.csv")

    def labels(self) -> list[dict]:
        return self.read_csv("labels.csv")

    def set_label(self, case_id: str, candidate_id: int, label: str) -> None:
        if label not in VALID_LABELS and label != "":
            raise ValueError(f"label must be one of {VALID_LABELS}")
        with _LOCK:
            rows = [r for r in self.labels()
                    if not (r["case_id"] == case_id and str(r["candidate_id"]) == str(candidate_id))]
            if label:
                rows.append({"case_id": case_id, "candidate_id": candidate_id, "label": label,
                             "labelled_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            self.write("labels.csv", self._csv_bytes(rows, LABEL_FIELDS), f"Label {case_id}#{candidate_id}: {label or 'cleared'}")


class LocalStore(BaseStore):
    name = "local"

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, path):
        p = self.root / path
        return p.read_bytes() if p.exists() else None

    def write(self, path, data, message=""):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, prefix):
        base = self.root / prefix
        if base.is_file():
            return [prefix]
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)).replace("\\", "/") for p in base.rglob("*") if p.is_file())


class GitHubStore(BaseStore):
    """Private GitHub repository via the REST API (git data API for multi-file atomic commits)."""
    name = "github"
    API = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    def __init__(self, repo: str, token: str, branch: str = "main"):
        self.repo, self.token, self.branch = repo, token, branch

    def _req(self, method, url, body=None, accept="application/vnd.github+json", raw=False):
        if not url.startswith("http"):
            url = f"{self.API}/repos/{self.repo}/{url.lstrip('/')}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", accept)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "LungLocAI")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    payload = r.read()
                    return payload if raw else (json.loads(payload) if payload else {})
            except urllib.error.HTTPError as e:
                if e.code == 404 and method == "GET":
                    return None
                if e.code in (409, 422, 502, 503) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"GitHub API {method} {url}: {e.code} {e.read()[:300]!r}") from None

    def check(self) -> dict:
        info = self._req("GET", f"{self.API}/repos/{self.repo}")
        if info is None:
            raise RuntimeError(f"Repository {self.repo} not found or token has no access")
        self.branch = info.get("default_branch", self.branch)
        return {"repo": self.repo, "private": info.get("private"), "branch": self.branch}

    def read(self, path):
        return self._req("GET", f"contents/{path}?ref={self.branch}", accept="application/vnd.github.raw", raw=True)

    def write(self, path, data, message=""):
        self.write_many({path: data}, message)

    def write_many(self, files, message):
        """One commit containing all files (blobs -> tree -> commit -> move branch).
        Falls back to the per-file contents API if the git data API is unavailable."""
        try:
            return self._write_tree(files, message)
        except RuntimeError as e:
            if " 403 " not in str(e) and " 404 " not in str(e):
                raise
        for path, data in files.items():
            self._put_contents(path, data, message)

    def _put_contents(self, path, data, message):
        for attempt in range(3):
            meta = self._req("GET", f"contents/{path}?ref={self.branch}")
            body = {"message": message, "content": base64.b64encode(data).decode(), "branch": self.branch}
            if isinstance(meta, dict) and meta.get("sha"):
                body["sha"] = meta["sha"]
            try:
                self._req("PUT", f"contents/{path}", body)
                return
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(1)

    def _write_tree(self, files, message):
        for attempt in range(3):
            ref = self._req("GET", f"git/ref/heads/{self.branch}")
            head = ref["object"]["sha"]
            base_tree = self._req("GET", f"git/commits/{head}")["tree"]["sha"]
            tree = []
            for path, data in files.items():
                blob = self._req("POST", "git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
                tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            new_tree = self._req("POST", "git/trees", {"base_tree": base_tree, "tree": tree})["sha"]
            commit = self._req("POST", "git/commits", {"message": message, "tree": new_tree, "parents": [head]})["sha"]
            try:
                self._req("PATCH", f"git/refs/heads/{self.branch}", {"sha": commit, "force": False})
                return
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(1)

    def list(self, prefix):
        tree = self._req("GET", f"git/trees/{self.branch}?recursive=1") or {}
        return sorted(t["path"] for t in tree.get("tree", []) if t["type"] == "blob" and t["path"].startswith(prefix))


def get_store() -> BaseStore | None:
    repo, token = os.environ.get("DATA_REPO", ""), os.environ.get("DATA_REPO_TOKEN", "")
    if repo and token:
        return GitHubStore(repo, token)
    if os.environ.get("DATA_DIR"):
        return LocalStore(os.environ["DATA_DIR"])
    return None

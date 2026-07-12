"""
src/experiment_tracking.py
-------------------------
Lightweight, dependency-free experiment tracking.

Why this exists:
- The repo already has `configs/` + `outputs/` conventions, but no integrated
  experiment tracker (MLflow/W&B are optional).
- This module provides a simple local run directory with:
    - run metadata (who/when/what)
    - config snapshots (YAML copies + hashes)
    - params (YAML)
    - metrics (JSONL stream)
    - artifacts (files copied into the run dir)

If you later adopt MLflow, you can still keep these run folders as an auditable
paper-trail.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_slug(s: str) -> str:
    keep = []
    for ch in s.strip().replace(" ", "-"):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
    out = "".join(keep)
    return out or "run"


def create_run_id(name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = os.urandom(4).hex()
    return f"{ts}_{_safe_slug(name)}_{rand}"


@dataclass
class ExperimentRun:
    """
    One experiment run rooted at `run_dir`.

    Files created:
      - meta.yaml
      - params.yaml
      - metrics.jsonl
      - artifacts/ (directory)
      - configs/ (directory, optional)
    """

    run_dir: Path
    run_id: str
    name: str
    meta: dict[str, Any] = field(default_factory=dict)
    _metrics_path: Path = field(init=False)
    _params_path: Path = field(init=False)
    _meta_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._params_path = self.run_dir / "params.yaml"
        self._meta_path = self.run_dir / "meta.yaml"

    @classmethod
    def start(
        cls,
        name: str,
        base_dir: Path | str = "outputs/experiments",
        config_paths: Optional[list[Path | str]] = None,
        snapshot_env: bool = True,
        tags: Optional[Mapping[str, Any]] = None,
        extra_meta: Optional[Mapping[str, Any]] = None,
    ) -> "ExperimentRun":
        base_dir = Path(base_dir)
        run_id = create_run_id(name)
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (run_dir / "configs").mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "run_id": run_id,
            "name": name,
            "created_utc": _utc_now_iso(),
            "cwd": os.getcwd(),
            "argv": list(sys.argv),
            "platform": {
                "python": sys.version.replace("\n", " "),
                "executable": sys.executable,
                "os": platform.platform(),
                "machine": platform.machine(),
                "hostname": platform.node(),
            },
            "tags": dict(tags or {}),
        }
        if extra_meta:
            meta.update(dict(extra_meta))

        run = cls(run_dir=run_dir, run_id=run_id, name=name, meta=meta)
        run._write_yaml(run._meta_path, meta)
        run._write_yaml(run._params_path, {})

        if config_paths:
            run.snapshot_configs(config_paths)

        if snapshot_env:
            run.snapshot_environment()

        return run

    def _write_yaml(self, path: Path, obj: Any) -> None:
        path.write_text(yaml.safe_dump(obj, sort_keys=False))

    def log_params(self, params: Mapping[str, Any]) -> None:
        existing = {}
        if self._params_path.exists():
            existing = yaml.safe_load(self._params_path.read_text()) or {}
        existing.update(dict(params))
        self._write_yaml(self._params_path, existing)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        record = {
            "time_utc": _utc_now_iso(),
            "step": step,
            **dict(metrics),
        }
        with self._metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_artifact(self, path: Path | str, name: Optional[str] = None) -> Path:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(src)
        dst_name = name or src.name
        dst = self.run_dir / "artifacts" / dst_name
        if src.is_dir():
            if dst.exists():
                raise FileExistsError(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return dst

    def snapshot_configs(self, config_paths: list[Path | str]) -> dict[str, str]:
        """
        Copy config files into `run_dir/configs/` and record sha256 hashes.
        Returns {relative_name: sha256}.
        """
        out: dict[str, str] = {}
        configs_dir = self.run_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        for p in config_paths:
            src = Path(p)
            if not src.exists():
                raise FileNotFoundError(src)
            dst = configs_dir / src.name
            shutil.copy2(src, dst)
            out[src.name] = _sha256_file(dst)

        # Store hashes in meta for reproducibility
        meta = yaml.safe_load(self._meta_path.read_text()) or {}
        meta["config_hashes"] = out
        self._write_yaml(self._meta_path, meta)
        return out

    def snapshot_environment(self) -> dict[str, str]:
        """
        Capture basic environment information to support reproducibility.

        Creates:
          - pip_freeze.txt (best-effort)

        Returns {filename: sha256}.
        """
        out: dict[str, str] = {}
        freeze_path = self.run_dir / "pip_freeze.txt"

        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=False,
                capture_output=True,
                text=True,
            )
            content = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            freeze_path.write_text(content, encoding="utf-8")
        except Exception as e:  # pragma: no cover
            freeze_path.write_text(f"pip freeze failed: {type(e).__name__}: {e}\n", encoding="utf-8")

        out[freeze_path.name] = _sha256_file(freeze_path)

        meta = yaml.safe_load(self._meta_path.read_text()) or {}
        meta["environment_hashes"] = out
        self._write_yaml(self._meta_path, meta)
        return out


if __name__ == "__main__":
    # Tiny smoke test: create a run folder and log one metric.
    run = ExperimentRun.start(
        name="smoke-test",
        config_paths=["configs/pinn_config.yaml", "configs/sweep_config.yaml"],
        tags={"purpose": "smoke"},
    )
    run.log_params({"seed": 456})
    run.log_metrics({"loss": 1.23}, step=0)
    print(f"Created run: {run.run_dir}")

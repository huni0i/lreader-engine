"""Read-only research dashboard + a small training job launcher.

Everything here reads artifacts that scripts/*.py already produce (no new
data format is invented) and starts the same scripts as subprocesses so the
dashboard is a thin window onto the existing CLI workflow, not a parallel
implementation of it.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
JOBS_DIR = DATA / "dashboard_jobs"

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _read_json(path: Path):
    # Bind-mounted dataset files (data/** is gitignored) may not exist on a
    # given machine yet -- Docker/Compose then creates an empty directory in
    # their place. Guard on is_file(), not just exists(), so a missing
    # dataset file degrades to "no data" instead of a 500.
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@router.get("/summary")
def summary() -> dict:
    """Everything the research tab needs, in one call."""
    return {
        "manga109s_cascade": _read_json(DATA / "benchmarks" / "manga109s.json"),
        "legacy_benchmark": _read_json(DATA / "benchmarks" / "report.json"),
        "eval_summary": _read_json(DATA / "eval" / "summary.json"),
        "eval_benchmark": _read_json(DATA / "eval" / "benchmark-report.json"),
        "ocr_router": _read_json(DATA / "experiments" / "ocr-router" / "metrics.json"),
        "detector_train_report": _read_json(DATA / "det-text" / "train_report.json"),
        "detector_curve": _read_csv_rows(
            DATA / "det-text" / "runs" / "text-detector-m" / "results.csv"
        ),
        "catalog": _read_json(DATA / "catalog.json"),
    }


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# Whitelisted so the dashboard can only ever run these known scripts with
# arbitrary flags -- never an arbitrary shell command.
SCRIPTS = {
    "text-detector": ROOT / "scripts" / "train_text_detector.py",
    "ocr-router": ROOT / "scripts" / "train_ocr_router.py",
    "benchmark-manga109s": ROOT / "scripts" / "benchmark_manga109s.py",
    "benchmark-pipeline": ROOT / "scripts" / "benchmark_pipeline.py",
    "benchmark-yolo-ocr": ROOT / "scripts" / "benchmark_yolo_ocr.py",
}


class TrainRequest(BaseModel):
    script: str
    args: list[str] = []


def _run_job(job_id: str, script_path: Path, args: list[str]) -> None:
    log_path = JOBS_DIR / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", str(script_path), *args],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
            )
        except OSError as error:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(error)
                JOBS[job_id]["finished_at"] = time.time()
            return
        with JOBS_LOCK:
            JOBS[job_id]["pid"] = process.pid
        process.wait()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done" if process.returncode == 0 else "failed"
            JOBS[job_id]["returncode"] = process.returncode
            JOBS[job_id]["finished_at"] = time.time()


@router.get("/scripts")
def list_scripts() -> dict:
    return {name: path.exists() for name, path in SCRIPTS.items()}


@router.post("/train")
def start_training(request: TrainRequest) -> dict:
    if request.script not in SCRIPTS:
        raise HTTPException(status_code=400, detail=f"Unknown script: {request.script}")
    script_path = SCRIPTS[request.script]
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    with JOBS_LOCK:
        running = [
            job
            for job in JOBS.values()
            if job["status"] == "running"
        ]
    if running:
        raise HTTPException(
            status_code=409,
            detail=(
                "A job is already running "
                f"({running[0]['script']}, id={running[0]['id']}). "
                "The GPU is shared -- wait for it to finish."
            ),
        )

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "script": request.script,
            "args": request.args,
            "status": "running",
            "started_at": time.time(),
            "finished_at": None,
            "pid": None,
            "returncode": None,
        }
    thread = threading.Thread(
        target=_run_job, args=(job_id, script_path, request.args), daemon=True
    )
    thread.start()
    return {"job_id": job_id}


@router.get("/train")
def list_jobs() -> list[dict]:
    with JOBS_LOCK:
        return sorted(JOBS.values(), key=lambda job: job["started_at"], reverse=True)


@router.get("/train/{job_id}")
def job_status(job_id: str, tail: int = 6000) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        job = dict(job)
    log_path = JOBS_DIR / f"{job_id}.log"
    log_text = ""
    if log_path.exists():
        content = log_path.read_text(encoding="utf-8", errors="replace")
        log_text = content[-tail:]
    job["log"] = log_text
    return job

"""Local web laboratory. The hosted site itself needs no application server."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kenken.pipeline import solve_image

CACHE = ROOT / "scratch" / "web_cache"
CACHE.mkdir(parents=True, exist_ok=True)
ENGINE_ID = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "kenken").glob("*.py"))) +
                           (ROOT / "models" / "ocr_cnn_finetuned.pt").read_bytes()).hexdigest()[:16]
app = FastAPI(title="KenKen Lab", docs_url=None, redoc_url=None)
solve_lock = threading.Lock()
test_lock = threading.Lock()
test_run: dict = {"state": "idle", "log": "", "exit_code": None}


def compute(image_bytes: bytes, method="auto", mode="clean", redundant=False) -> dict:
    """Run the original pipeline; cache only successful executions, including infeasibility."""
    if method not in {"auto", "arithmetic", "table", "joint"}:
        raise ValueError("Elige un método válido.")
    if mode not in {"clean", "composite", "original", "rectified"}:
        raise ValueError("Elige una vista válida.")
    key = hashlib.sha256(image_bytes + f"{ENGINE_ID}:{method}:{mode}:{redundant}".encode()).hexdigest()
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return {**json.loads(cached.read_text("utf-8")), "cached": True}
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen. Usa PNG, JPG o WebP.")
    if image.shape[0] * image.shape[1] > 24_000_000:
        raise ValueError("La imagen supera 24 megapíxeles. Reduce su tamaño.")
    start = time.perf_counter()
    with solve_lock:
        result = solve_image(image, method=method, redundant=redundant, time_limit=10)
        rendered = None
        if result.solved:
            rendered = CACHE / f"{key}.png"
            result.render_solution(out_path=rendered, mode=mode)
    solver = result.solve_result
    payload = {"status": result.status, "grid": result.grid,
               "instance": result.instance.to_dict(), "fallback_used": result.fallback_used,
               "num_changed": result.num_changed, "fidelity": result.fidelity,
               "confidence_level": result.confidence_level, "divergent": result.divergent,
               "solver_ms": round(solver.wall_time * 1000, 2),
               "total_ms": round((time.perf_counter() - start) * 1000, 2),
               "branches": solver.branches, "conflicts": solver.conflicts,
               "image": f"cache/{rendered.name}" if rendered else None,
               "cached": False, "engine": "Python · OpenCV + GlyphCNN + CP-SAT"}
    cached.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    return payload


@app.get("/api/health")
def health():
    return {"mode": "local", "engine_version": ENGINE_ID}


@app.post("/api/solve")
def solve_photo(image: UploadFile = File(...), method: str = Form("auto"),
                mode: str = Form("clean"), redundant: bool = Form(False)):
    data = image.file.read(12 * 1024 * 1024 + 1)
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, "La imagen supera 12 MB. Reduce su tamaño.")
    try:
        return compute(data, method, mode, redundant)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(422, str(error)) from error


def run_tests():
    global test_run
    log_path = CACHE / "pytest.log"
    try:
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"}
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short"],
                                       cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -1
                output.write("\nLímite de 5 minutos alcanzado.\n")
        test_run = {"state": "passed" if code == 0 else "failed", "exit_code": code}
    except Exception as error:
        test_run = {"state": "failed", "exit_code": -1, "log": str(error)}


@app.post("/api/tests")
def start_tests():
    global test_run
    with test_lock:
        if test_run["state"] != "running":
            (CACHE / "pytest.log").write_text("", "utf-8")
            test_run = {"state": "running", "exit_code": None}
            threading.Thread(target=run_tests, daemon=True).start()
    return test_run


@app.get("/api/tests")
def test_status():
    path = CACHE / "pytest.log"
    return {**test_run, "log": path.read_text("utf-8", errors="replace")[-30000:] if path.exists() else ""}


app.mount("/cache", StaticFiles(directory=CACHE), name="cache")
app.mount("/", StaticFiles(directory=ROOT / "web" / "public", html=True), name="site")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

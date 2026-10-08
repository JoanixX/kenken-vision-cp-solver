"""Pruebas del módulo de benchmarking experimental de CP."""

import pandas as pd
import pytest

from kenken.benchmark import run_benchmark


def test_run_benchmark_smoke():
    # Ejecución rápida con n=3 y 1 repetición
    df = run_benchmark(sizes=(3,), repeats=1, verbose=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 4 variantes evaluadas
    assert set(df["status"]) == {"OPTIMAL"}
    assert df["is_valid"].all()
    assert (df["wall_time_ms"] > 0).all()

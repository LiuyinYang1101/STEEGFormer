"""Schema smoke-test for the data_preprocessing/ scripts.

Two checks, neither of which needs the raw datasets or MNE:

  1. py_compile every script under data_preprocessing/.
  2. For each dataset, synthesise a tiny output file in the exact format the script writes,
     then load it through the repo's real Dataset class and pull a sample -- confirming the
     scripts target the contract the trainer expects.

This validates the *interface* (file layout, keys, shapes, label flow). It does NOT validate
the numerical preprocessing (that needs the raw data). Run from anywhere:

    conda run -n eeg311 python data_preprocessing/tools/verify_schemas.py
"""
import os
import sys
import py_compile
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

REPO = Path(__file__).resolve().parents[2]
DP = REPO / "data_preprocessing"
sys.path.insert(0, str(REPO / "benchmark" / "neural_networks" / "util"))
import eeg_downstream_dataset as ds  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
    except Exception as e:  # noqa: BLE001
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))


# --------------------------------------------------------------------------- #
# 1) compile every script
# --------------------------------------------------------------------------- #
def compile_all():
    for py in sorted(DP.rglob("*.py")):
        if "tools" in py.parts and py.name == "verify_schemas.py":
            continue
        py_compile.compile(str(py), doraise=True)


check("py_compile all data_preprocessing/*.py", compile_all)


# --------------------------------------------------------------------------- #
# 2) synthetic round-trips through the repo Dataset classes
# --------------------------------------------------------------------------- #
def _h5_str(grp, key, values):
    grp.create_dataset(key, data=np.array(values, dtype=object),
                       dtype=h5py.string_dtype(encoding="utf-8"))


def alzheimer():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Asub-001.pkl"
        pickle.dump({"eeg": np.random.randn(10, 19, 768).astype(np.float32), "group": "C"}, open(p, "wb"))
        cl = {"C": 0, "A": 1, "F": 2}
        for train in (True, False):
            dset = ds.AlzheimerDataset(str(p), class_label=cl, train=train, data_length=768)
            x, y = dset[0]
            assert tuple(x.shape) == (19, 768) and y == 0


def error():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "S01.h5"
        n = 20
        with h5py.File(p, "w") as f5:
            f5.create_dataset("X", data=np.random.randn(n, 64, 256).astype(np.float32))
            g = f5.create_group("df")
            g.create_dataset("trial_idx", data=np.arange(n), dtype="i8")
            _h5_str(g, "class", ["error", "no_error"] * (n // 2))
            _h5_str(g, "set", ["train"] * (n // 2) + ["test"] * (n // 2))
        cl = {"no_error": 0, "error": 1}
        for train in (True, False):
            dset = ds.ErrorDataset(str(p), train=train, class_label=cl, data_length=256)
            x, y = dset[0]
            assert tuple(x.shape) == (64, 256) and y in (0, 1)


def inner_speech():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub1.h5"
        n = 20
        labels = ["Arriba/Imagined", "Abajo/Imagined", "Derecha/Imagined", "Izquierda/Imagined"] * 5
        with h5py.File(p, "w") as f5:
            f5.create_dataset("X", data=np.random.randn(n, 128, 768).astype(np.float32))
            g = f5.create_group("df")
            g.create_dataset("trial", data=np.arange(n), dtype="i8")
            _h5_str(g, "label", labels)
            folds = f5.create_group("folds")
            for k in range(5):
                fk = folds.create_group(f"fold_{k}")
                fk.create_dataset("train", data=np.arange(0, 16), dtype="i8")
                fk.create_dataset("test", data=np.arange(16, 20), dtype="i8")
        cl = {"Arriba/Imagined": 0, "Abajo/Imagined": 1, "Derecha/Imagined": 2, "Izquierda/Imagined": 3}
        for train in (True, False):
            dset = ds.InnerSpeechDataset(str(p), fold=0, train=train, class_label=cl, data_length=768)
            x, y = dset[0]
            assert tuple(x.shape) == (128, 768) and y in range(4)


def upper_limb():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub1.h5"
        n = 28
        classes = ["rest", "hand_open", "hand_close", "elbow_flexion",
                   "elbow_extension", "pronation", "supination"] * 4
        with h5py.File(p, "w") as f5:
            f5.create_dataset("X", data=np.random.randn(n, 61, 384).astype(np.float32))
            g = f5.create_group("df")
            g.create_dataset("trial_idx", data=np.arange(n), dtype="i8")
            _h5_str(g, "class", classes)
            _h5_str(g, "experiment", ["motorexecution"] * n)
            folds = f5.create_group("folds")
            for exp in ("motorexecution", "motorimagination"):
                ge = folds.create_group(exp)
                for k in range(5):
                    fk = ge.create_group(f"fold_{k}")
                    fk.create_dataset("train", data=np.arange(0, 21), dtype="i8")
                    fk.create_dataset("test", data=np.arange(21, 28), dtype="i8")
        cl = {"rest": 0, "hand_open": 1, "hand_close": 2, "elbow_flexion": 3,
              "elbow_extension": 4, "pronation": 5, "supination": 6}
        for train in (True, False):
            dset = ds.UpperLimbDataset(str(p), fold=0, classification_task="motorexecution",
                                       train=train, class_label=cl, data_length=384)
            x, y = dset[0]
            assert tuple(x.shape) == (61, 384) and y in range(7)


def binocular_ssvep():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Subject1.pkl"
        T, R, C, L = 40, 5, 64, 300
        data = np.random.randn(T, R, C, L).astype(np.float32)
        rows, ex = [], 0
        for t in range(T):
            for r in range(R):
                for s in range(0, L - 250 + 1, 25):
                    rows.append({"example_idx": ex, "target": t, "epoch": r, "start_sample": s})
                    ex += 1
        df = pd.DataFrame(rows)
        splits = []
        for r in range(R):
            test = df["epoch"] == r
            splits.append((df.loc[~test, "example_idx"].tolist(),
                           df.loc[test & (df["start_sample"] == 0), "example_idx"].tolist(),
                           df.loc[test, "example_idx"].tolist()))
        pickle.dump({"data": data, "df": df, "splits": splits}, open(p, "wb"))
        cl = {t: t for t in range(40)}
        for train, task in ((True, "sync"), (False, "sync"), (False, "async")):
            dset = ds.BinocularSSVEPDataset(str(p), fold=0, classification_task=task,
                                            train=train, class_label=cl, data_length=250)
            x, y = dset[0]
            assert tuple(x.shape) == (64, 250) and y in range(40)


for nm, fn in [("alzheimer", alzheimer), ("error", error), ("inner_speech", inner_speech),
               ("upper_limb", upper_limb), ("binocular_ssvep", binocular_ssvep)]:
    check(f"schema round-trip: {nm}", fn)


# --------------------------------------------------------------------------- #
print("\n=== verify_schemas results ===")
for status, name, msg in results:
    print(f"[{status}] {name}" + (f"  --  {msg}" if msg else ""))
n_fail = sum(1 for s, _, _ in results if s == FAIL)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
sys.exit(1 if n_fail else 0)

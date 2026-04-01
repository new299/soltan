#!/usr/bin/env python3
"""
test_accuracy_lite.py — Evaluate soltan.py on a labelled test set.

No external tools required (no USEARCH, no TMHMM).

Usage:
    python test_accuracy_lite.py

    # Custom files or threshold:
    python test_accuracy_lite.py \
        --i_fa  trainingdata/test_set.fasta \
        --i_csv trainingdata/test_set.csv \
        --threshold 0.5

    # Force model retrain before evaluating:
    python test_accuracy_lite.py --retrain
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil

import pandas as pd
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report,
)

_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def run_soltan(i_fa, tmp_dir, model=None, retrain=False):
    out_csv = os.path.join(tmp_dir, "soltan_out.csv")
    cmd = [
        sys.executable,
        os.path.join(_SCRIPT_DIR, "soltan.py"),
        "--i_fa",  i_fa,
        "--o_csv", out_csv,
    ]
    if model:
        cmd += ["--model", model]
    if retrain:
        cmd.append("--retrain")

    print("Running soltan …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            print(" ", line)
    if result.stderr:
        for line in result.stderr.splitlines():
            print(" [warn]", line, file=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"soltan.py failed with exit code {result.returncode}")
    return out_csv


def evaluate(predictions_csv, labels_csv, threshold):
    preds  = pd.read_csv(predictions_csv)   # runtime_id, fa_id, soluble
    labels = pd.read_csv(labels_csv)        # sid, solubility

    preds["fa_id"]  = preds["fa_id"].astype(str)
    labels["sid"]   = labels["sid"].astype(str)
    merged = preds.merge(labels, left_on="fa_id", right_on="sid", how="inner")

    if merged.empty:
        sys.exit("No matching IDs between predictions and label file.")

    n_total   = len(labels)
    n_matched = len(merged)
    n_missing = n_total - n_matched
    if n_missing:
        print(f"\nNote: {n_missing}/{n_total} test sequences had no prediction "
              f"and are excluded.\n")

    y_true  = merged["solubility"].values
    y_score = merged["soluble"].values
    y_pred  = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("=" * 52)
    print(f"  Test set evaluation  (threshold = {threshold})")
    print("=" * 52)
    print(f"  Sequences evaluated : {n_matched}")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  MCC                 : {mcc:.4f}")
    print(f"  ROC-AUC             : {auc:.4f}")
    print(f"  Sensitivity (TPR)   : {tp/(tp+fn):.4f}  ({tp} TP, {fn} FN)")
    print(f"  Specificity (TNR)   : {tn/(tn+fp):.4f}  ({tn} TN, {fp} FP)")
    print()
    print("  Classification report:")
    print(classification_report(y_true, y_pred,
                                target_names=["insoluble", "soluble"],
                                digits=4))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate soltan.py accuracy on a labelled test set.")
    parser.add_argument("--i_fa",  default="trainingdata/test_set.fasta",
                        help="Test sequences FASTA (default: trainingdata/test_set.fasta)")
    parser.add_argument("--i_csv", default="trainingdata/test_set.csv",
                        help="Test labels CSV with columns sid,solubility "
                             "(default: trainingdata/test_set.csv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for soluble/insoluble (default: 0.5)")
    parser.add_argument("--model", default=None,
                        help="Path to a specific model .pkl "
                             "(default: data/soltan_model.pkl)")
    parser.add_argument("--retrain", action="store_true", default=False,
                        help="Force model retrain before evaluating")
    parser.add_argument("--tmp_dir", default=None,
                        help="Temp directory (default: auto)")
    args = parser.parse_args()

    # Resolve paths relative to this script
    i_fa  = args.i_fa  if os.path.isabs(args.i_fa)  else os.path.join(_SCRIPT_DIR, args.i_fa)
    i_csv = args.i_csv if os.path.isabs(args.i_csv) else os.path.join(_SCRIPT_DIR, args.i_csv)

    for p, name in [(i_fa, "--i_fa"), (i_csv, "--i_csv")]:
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}  (set {name} explicitly if needed)")

    use_tmp = args.tmp_dir is None
    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="soltan_test_")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        out_csv = run_soltan(i_fa, tmp_dir,
                                    model=args.model,
                                    retrain=args.retrain)
        evaluate(out_csv, i_csv, args.threshold)
    finally:
        if use_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
soluprot_lite.py — Protein solubility predictor for E. coli expression.

A self-contained reimplementation of SoluProt that requires no external
tools (no USEARCH, no TMHMM). All features are derived purely from the
amino acid sequence.

Features used (242 total):
  - 20  amino acid monomer frequencies
  - 210 dipeptide (dimer) combination frequencies
  - 12  physico-chemical properties

Model: Gradient Boosting Classifier (same hyperparameters as original).
The model is trained on first run and cached to disk as a .pkl file.

Usage:
    python soluprot_lite.py --i_fa sequences.fasta --o_csv results.csv

    # Use custom training data:
    python soluprot_lite.py --i_fa sequences.fasta --o_csv results.csv \\
        --train_fa trainingdata/training_set.fasta \\
        --train_csv trainingdata/training_set.csv

    # Force model retrain:
    python soluprot_lite.py --i_fa sequences.fasta --o_csv results.csv --retrain

Requirements:
    pip install biopython scikit-learn pandas numpy joblib
"""

import argparse
import os
import sys
import warnings
from itertools import combinations_with_replacement

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import ProtParam
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AA = "ACDEFGHIKLMNPQRSTVWY"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TRAIN_FA  = os.path.join(_SCRIPT_DIR, "trainingdata", "training_set.fasta")
_DEFAULT_TRAIN_CSV = os.path.join(_SCRIPT_DIR, "trainingdata", "training_set.csv")
_DEFAULT_MODEL     = os.path.join(_SCRIPT_DIR, "data", "soluprot_lite_model.pkl")

# Pre-compute the canonical dimer list once (210 symmetric pairs)
_DIMER_COMBS = ["".join(c) for c in combinations_with_replacement(AA, 2)]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_monomers(seq: str) -> dict:
    """Relative frequency of each of the 20 standard amino acids."""
    analysis = ProtParam.ProteinAnalysis(seq)
    freq = analysis.amino_acids_percent
    return {f"mono_{aa}": freq[aa] for aa in AA}


def compute_dimers(seq: str) -> dict:
    """
    Normalised frequency of all 210 symmetric dipeptide combinations.

    For each consecutive pair (seq[i], seq[i+1]) the canonical key is the
    lexicographically smaller of the pair and its reverse, matching the
    original DimerComb logic.
    """
    counts = dict.fromkeys(_DIMER_COMBS, 0)
    prev = seq[0]
    for j in range(1, len(seq)):
        pair = prev + seq[j]
        rev  = pair[::-1]
        key  = rev if rev in counts else pair
        if key in counts:
            counts[key] += 1
        prev = seq[j]

    n = len(seq) - 1  # number of pairs
    return {f"dimer_{k}": v / n for k, v in counts.items()}


def compute_physico_chemical(seq: str) -> dict:
    """
    Twelve global physico-chemical descriptors via BioPython ProteinAnalysis.

    fracnumcharge  — fraction of charged residues (R + K + D + E)
    kr_ratio       — Lys/Arg ratio (NaN if Arg == 0)
    aa_helix       — predicted helix fraction
    aa_sheet       — predicted strand fraction
    aa_turn        — predicted turn fraction
    molecular_weight
    aromaticity
    avg_molecular_weight — MW / length
    flexibility    — mean of per-residue flexibility values
    gravy          — grand average of hydropathicity (Kyte-Doolittle)
    isoelectric_point
    instability_index
    """
    analysis = ProtParam.ProteinAnalysis(seq)
    freq = analysis.amino_acids_percent
    mw   = analysis.molecular_weight()
    h, s, t = analysis.secondary_structure_fraction()

    kr = freq["K"] / freq["R"] if freq["R"] != 0 else np.nan

    return {
        "pc_fracnumcharge":       freq["R"] + freq["K"] + freq["D"] + freq["E"],
        "pc_kr_ratio":            kr,
        "pc_aa_helix":            h,
        "pc_aa_sheet":            s,
        "pc_aa_turn":             t,
        "pc_molecular_weight":    mw,
        "pc_aromaticity":         analysis.aromaticity(),
        "pc_avg_molecular_weight": mw / analysis.length,
        "pc_flexibility":         float(np.mean(analysis.flexibility())),
        "pc_gravy":               analysis.gravy(),
        "pc_isoelectric_point":   analysis.isoelectric_point(),
        "pc_instability_index":   analysis.instability_index(),
    }


def featurize(seq: str) -> dict:
    """Compute the full 242-dimensional feature vector for a single sequence."""
    row = {}
    row.update(compute_monomers(seq))
    row.update(compute_dimers(seq))
    row.update(compute_physico_chemical(seq))
    return row


def featurize_batch(sequences: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Compute features for a dict {sid: sequence}.
    Returns a DataFrame indexed by sid.
    """
    rows = []
    ids  = list(sequences.keys())
    n    = len(ids)
    for i, sid in enumerate(ids):
        if verbose and (i % 100 == 0 or i == n - 1):
            print(f"  Features: {i + 1}/{n}", end="\r", flush=True)
        rows.append(featurize(sequences[sid]))
    if verbose:
        print()
    df = pd.DataFrame(rows, index=ids)
    df.index.name = "sid"
    return df


# ---------------------------------------------------------------------------
# Sequence loading / validation
# ---------------------------------------------------------------------------

def load_fasta(path: str, min_len: int = 20) -> dict:
    """
    Parse a FASTA file and return {id: cleaned_sequence}.
    Non-standard amino acids are silently removed; sequences shorter than
    min_len after cleaning are skipped with a warning.
    """
    seqs = {}
    seen = set()
    for record in SeqIO.parse(path, "fasta"):
        sid = record.id
        if sid in seen:
            print(f"Warning: duplicate ID '{sid}' — skipping.", file=sys.stderr)
            continue
        seen.add(sid)
        seq = "".join(aa for aa in str(record.seq).upper() if aa in AA)
        if len(seq) < min_len:
            print(
                f"Warning: '{sid}' has only {len(seq)} standard AAs after "
                f"cleaning (min {min_len}) — skipping.",
                file=sys.stderr,
            )
            continue
        seqs[sid] = seq
    return seqs


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(train_fa: str, train_csv: str, model_path: str) -> dict:
    """
    Train a GradientBoostingClassifier on the given labelled dataset and
    save it to model_path.  Returns the model bundle.
    """
    print("Loading training sequences …")
    seqs = load_fasta(train_fa)

    print("Loading training labels …")
    labels = pd.read_csv(train_csv, index_col="sid")
    labels.index = labels.index.astype(str)

    # Align labels ↔ sequences
    common = labels.index.intersection(pd.Index(list(seqs.keys())))
    dropped = len(labels) - len(common)
    if dropped:
        print(f"  Warning: {dropped} label row(s) had no matching FASTA entry — dropped.")
    seqs   = {sid: seqs[sid] for sid in common}
    labels = labels.loc[common]
    print(f"  {len(common)} sequences available for training.")

    print("Computing training features …")
    X = featurize_batch(seqs)

    feature_order = list(X.columns)
    features_mean = X.mean().to_dict()

    # Impute any NaN columns with the column mean
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(features_mean[col], inplace=True)

    y = labels.loc[list(seqs.keys()), "solubility"].values.astype(int)

    print("Training GradientBoostingClassifier …")
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X.values.astype(np.float64), y)
    train_acc = (clf.predict(X.values.astype(np.float64)) == y).mean()
    print(f"  Training accuracy: {train_acc:.4f}")

    bundle = {
        "classifier":    clf,
        "feature_order": feature_order,
        "features_mean": features_mean,
    }
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"  Model saved → {model_path}")
    return bundle


def load_or_train_model(model_path: str, train_fa: str, train_csv: str,
                         retrain: bool = False) -> dict:
    """Return a model bundle, training first if necessary."""
    if not retrain and os.path.exists(model_path):
        print(f"Loading model from {model_path} …")
        return joblib.load(model_path)

    if not os.path.exists(train_fa):
        sys.exit(
            f"Error: training FASTA not found at '{train_fa}'.\n"
            "Supply --train_fa or put training_set.fasta in trainingdata/."
        )
    if not os.path.exists(train_csv):
        sys.exit(
            f"Error: training CSV not found at '{train_csv}'.\n"
            "Supply --train_csv or put training_set.csv in trainingdata/."
        )
    return train_model(train_fa, train_csv, model_path)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model_bundle: dict, sequences: dict, round_to: int = 4) -> pd.DataFrame:
    """
    Predict solubility scores for a dict {sid: sequence}.
    Returns a DataFrame with columns [fa_id, soluble].
    """
    clf           = model_bundle["classifier"]
    feature_order = model_bundle["feature_order"]
    features_mean = model_bundle["features_mean"]

    print(f"Computing features for {len(sequences)} sequence(s) …")
    X = featurize_batch(sequences)

    # Ensure columns match training order; fill any missing with mean
    for col in feature_order:
        if col not in X.columns:
            X[col] = features_mean.get(col, 0.0)
    X = X[feature_order]

    # Impute NaNs
    for col in feature_order:
        null_mask = X[col].isnull()
        if null_mask.any():
            fill = features_mean.get(col, 0.0)
            X.loc[null_mask, col] = fill
            for sid in X.index[null_mask]:
                print(
                    f"Warning: feature '{col}' could not be computed for '{sid}'; "
                    f"using training mean ({fill:.4f}).",
                    file=sys.stderr,
                )

    pos_class_idx = list(clf.classes_).index(1)
    proba = clf.predict_proba(X.values.astype(np.float64))[:, pos_class_idx]
    proba = np.clip(np.round(proba, round_to), 0.0, 1.0)

    results = pd.DataFrame(
        {"fa_id": list(sequences.keys()), "soluble": proba},
        index=range(len(sequences)),
    )
    results.index.name = "runtime_id"
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def arguments():
    parser = argparse.ArgumentParser(
        description=(
            "SoluProt Lite — sequence-only protein solubility predictor for E. coli.\n"
            "Predicts the probability that a protein will be expressed in soluble form.\n\n"
            "No external tools required (no USEARCH, no TMHMM).\n"
            "Features: amino acid frequencies, dipeptide frequencies, physico-chemical."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--i_fa",      required=True,
                        help="Input sequences in FASTA format.")
    parser.add_argument("--o_csv",     required=True,
                        help="Output CSV file with solubility scores.")
    parser.add_argument("--model",     default=_DEFAULT_MODEL,
                        help=f"Path to trained model .pkl (default: {_DEFAULT_MODEL}).")
    parser.add_argument("--train_fa",  default=_DEFAULT_TRAIN_FA,
                        help=f"Training FASTA (default: {_DEFAULT_TRAIN_FA}).")
    parser.add_argument("--train_csv", default=_DEFAULT_TRAIN_CSV,
                        help=f"Training CSV with 'sid' and 'solubility' columns "
                             f"(default: {_DEFAULT_TRAIN_CSV}).")
    parser.add_argument("--retrain",   action="store_true", default=False,
                        help="Force retraining even if a cached model exists.")
    parser.add_argument("--min_len",   type=int, default=20,
                        help="Minimum sequence length after cleaning (default: 20).")
    return parser.parse_args()


def main():
    args = arguments()

    if not os.path.isfile(args.i_fa):
        sys.exit(f"Error: input FASTA not found: '{args.i_fa}'")

    model_bundle = load_or_train_model(
        args.model, args.train_fa, args.train_csv, retrain=args.retrain
    )

    print(f"Loading query sequences from {args.i_fa} …")
    sequences = load_fasta(args.i_fa, min_len=args.min_len)
    if not sequences:
        sys.exit("Error: no valid sequences found in input FASTA.")
    print(f"  {len(sequences)} sequence(s) loaded.")

    results = predict(model_bundle, sequences)

    results.to_csv(args.o_csv)
    print(f"\nResults written to {args.o_csv}")
    print(f"\n{'ID':<30}  {'Solubility Score':>18}")
    print("-" * 52)
    for _, row in results.iterrows():
        label = (
            "  ✓ likely soluble"   if row["soluble"] >= 0.5 else
            "  ✗ likely insoluble"
        )
        print(f"{str(row['fa_id']):<30}  {row['soluble']:>18.4f}{label}")


if __name__ == "__main__":
    main()

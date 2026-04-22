"""Aptamer QSAR Tool for RNA/DNA sequence descriptors and interaction-aware modeling."""
from __future__ import annotations

import io, math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
APP_NAME = "Aptamer QSAR Tool"
PLOT_TEMPLATE = "plotly_white"
DEVELOPER_NAME = "Ahmed G. Soliman"
DEVELOPER_PORTFOLIO = "https://sites.google.com/view/ahmed-g-soliman/home"
BASES_DNA = "ACGT"
BASES_RNA = "ACGU"
AMBIGUOUS = set("NRYKMSWBDHV")
DNA_MW = {"A": 313.21, "C": 289.18, "G": 329.21, "T": 304.20}
RNA_MW = {"A": 329.21, "C": 305.18, "G": 345.21, "U": 306.17}
EXT_260 = {"A": 15400.0, "C": 7400.0, "G": 11500.0, "T": 8700.0, "U": 9900.0}
INTERACTION_COLUMNS = ["H_bonds", "Hydrophobic_contacts", "Pi_stacking", "Electrostatic_contacts", "Salt_bridges", "Metal_coordination", "Van_der_Waals", "Water_bridges", "Base_pairing_contacts", "Base_stacking_contacts", "Target_contact_count"]
LOWER_IS_BETTER_TOKENS = ("kd", "ic50", "ec50", "ki", "mic", "docking", "vina", "bindingscore", "bindingenergy", "deltag", "delta_g")

st.set_page_config(page_title=APP_NAME, page_icon="A", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: "Manrope", "Segoe UI", sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #edf9f7 0%, #f7fbff 100%); border-right: 1px solid #cfe8e4; }
.apt-hero { border: 1px solid #cfe8e4; border-radius: 26px; padding: 1.35rem 1.55rem; margin-bottom: 1rem; background: radial-gradient(circle at 8% 12%, rgba(14,124,123,.16), transparent 16rem), linear-gradient(135deg, #f8fffc 0%, #eaf7f5 60%, #eff5ff 100%); box-shadow: 0 18px 54px rgba(16,42,45,.08); }
.apt-hero h1 { margin: 0; color: #102a2d; letter-spacing: -0.045em; font-size: 2.15rem; }
.apt-hero p { margin: .45rem 0 0; color: #486b70; }
</style>
""", unsafe_allow_html=True)

@dataclass
class DescriptorOptions:
    include_dinucleotide: bool = True
    include_trinucleotide: bool = False
    include_interactions: bool = True
    molecule_type: str = "Auto"

@dataclass
class ModelBundle:
    model_name: str
    task_type: str
    target: str
    features: list[str]
    descriptor_options: DescriptorOptions
    pipeline: Any
    label_encoder: LabelEncoder | None
    metrics: dict[str, Any]
    cv_summary: dict[str, float]
    created_at: str

def init_state() -> None:
    defaults = {"raw_df": None, "validated_df": None, "interaction_df": None, "descriptor_df": None, "descriptor_options": DescriptorOptions(), "training_result": None, "active_bundle": None, "prediction_df": None, "prediction_descriptor_df": None}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_fasta(text: str) -> pd.DataFrame:
    rows, name, seq_parts = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if name is not None: rows.append({"Name": name, "Sequence": "".join(seq_parts)})
            name, seq_parts = line[1:].strip() or f"Aptamer_{len(rows)+1}", []
        else:
            seq_parts.append(line)
    if name is not None: rows.append({"Name": name, "Sequence": "".join(seq_parts)})
    return pd.DataFrame(rows)

def parse_manual_sequences(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text: return pd.DataFrame(columns=["Name", "Sequence"])
    if text.startswith(">"): return parse_fasta(text)
    rows = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line: continue
        parts = [p.strip() for p in (line.split(",") if "," in line else line.split("\t") if "\t" in line else [line])]
        rows.append({"Name": parts[0] if len(parts) >= 2 else f"Aptamer_{i}", "Sequence": parts[1] if len(parts) >= 2 else parts[0]})
    return pd.DataFrame(rows)

def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower(); data = uploaded_file.read()
    if suffix in {".xlsx", ".xls"}: return pd.read_excel(io.BytesIO(data))
    text = data.decode("utf-8", errors="ignore")
    if suffix in {".fasta", ".fa", ".fna"} or text.lstrip().startswith(">"): return parse_fasta(text)
    if suffix == ".txt":
        try: return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception: return parse_manual_sequences(text)
    return pd.read_csv(io.BytesIO(data))

def normalize_sequence(seq: Any, molecule_type: str) -> str:
    if pd.isna(seq): return ""
    clean = "".join(ch for ch in str(seq).upper() if ch.isalpha())
    if molecule_type == "DNA": clean = clean.replace("U", "T")
    elif molecule_type == "RNA": clean = clean.replace("T", "U")
    return clean

def infer_molecule_type(seq: str, requested: str = "Auto") -> str:
    if requested in {"DNA", "RNA"}: return requested
    return "RNA" if "U" in seq and "T" not in seq else "DNA"

def sequence_validity(seq: Any, molecule_type: str) -> tuple[bool, str]:
    clean = normalize_sequence(seq, molecule_type)
    if not clean: return False, "Empty sequence"
    alphabet = set(BASES_RNA if infer_molecule_type(clean, molecule_type) == "RNA" else BASES_DNA) | AMBIGUOUS
    invalid = sorted(set(clean) - alphabet)
    return (False, "Invalid characters: " + ", ".join(invalid)) if invalid else (True, "OK")

def guess_sequence_column(df: pd.DataFrame) -> str | None:
    preferred = ["Sequence", "Aptamer", "AptamerSequence", "DNA", "RNA", "Seq", "NucleotideSequence"]
    lower_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for item in preferred:
        key = item.lower().replace(" ", "").replace("_", "")
        if key in lower_map: return lower_map[key]
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if not sample.empty and sample.map(lambda x: sequence_validity(x, "Auto")[0]).mean() >= 0.6: return col
    return None

def validate_dataframe(df: pd.DataFrame, sequence_col: str, molecule_type: str) -> pd.DataFrame:
    rows = []
    for idx, row in df.iterrows():
        seq = normalize_sequence(row.get(sequence_col, ""), molecule_type); ok, msg = sequence_validity(seq, molecule_type)
        rec = row.to_dict(); rec["Sequence"] = seq; rec["Name"] = rec.get("Name") or rec.get("ID") or f"Aptamer_{idx+1}"; rec["MoleculeType"] = infer_molecule_type(seq, molecule_type); rec["Valid"] = ok; rec["ValidationMessage"] = msg; rows.append(rec)
    return pd.DataFrame(rows)

def base_counts(seq: str, mol_type: str) -> dict[str, int]:
    return {base: seq.count(base) for base in (BASES_RNA if mol_type == "RNA" else BASES_DNA)}

def kmer_frequencies(seq: str, alphabet: str, k: int, prefix: str) -> dict[str, float]:
    if k == 2: kmers = [a + b for a in alphabet for b in alphabet]
    else: kmers = [a + b + c for a in alphabet for b in alphabet for c in alphabet]
    out = {f"{prefix}_{kmer}": 0.0 for kmer in kmers}; total = max(len(seq) - k + 1, 0)
    if total <= 0: return out
    for i in range(total):
        frag = seq[i:i+k]
        if len(frag) == k and all(ch in alphabet for ch in frag): out[f"{prefix}_{frag}"] += 1.0 / total
    return out

def shannon_entropy(seq: str, alphabet: str) -> float:
    if not seq: return 0.0
    probs = [seq.count(base) / len(seq) for base in alphabet if seq.count(base) > 0]
    return float(-sum(p * math.log2(p) for p in probs)) if probs else 0.0

def longest_run(seq: str, bases: set[str] | None = None) -> int:
    best = current = 0; last = None
    for ch in seq:
        allowed = bases is None or ch in bases
        if allowed and ch == last: current += 1
        elif allowed: current = 1
        else: current = 0
        last = ch if allowed else None; best = max(best, current)
    return int(best)

def reverse_complement(seq: str, mol_type: str) -> str:
    table = str.maketrans("ACGTU", "TGCAA" if mol_type == "DNA" else "UGCAA")
    return seq.translate(table)[::-1]

def complementarity_score(seq: str, mol_type: str) -> float:
    if len(seq) < 4: return 0.0
    rc = reverse_complement(seq, mol_type); best = 0
    for offset in range(-len(seq) + 1, len(seq)):
        matches = compared = 0
        for i, ch in enumerate(seq):
            j = i + offset
            if 0 <= j < len(rc): compared += 1; matches += int(ch == rc[j])
        if compared: best = max(best, matches)
    return round(best / len(seq), 4)

def sequence_descriptors(seq: str, mol_type: str, options: DescriptorOptions) -> dict[str, float]:
    mol_type = infer_molecule_type(seq, mol_type); alphabet = BASES_RNA if mol_type == "RNA" else BASES_DNA
    counts = base_counts(seq, mol_type); length = len(seq); gc = counts.get("G",0)+counts.get("C",0); atu = counts.get("A",0)+counts.get("T",0)+counts.get("U",0)
    purine = counts.get("A",0)+counts.get("G",0); pyr = counts.get("C",0)+counts.get("T",0)+counts.get("U",0)
    mw_map = RNA_MW if mol_type == "RNA" else DNA_MW; mw = sum(mw_map.get(ch,0.0) for ch in seq if ch in mw_map) + 79.0; ext = sum(EXT_260.get(ch,0.0) for ch in seq)
    tm_wallace = 2.0 * atu + 4.0 * gc; tm_gc = 0.0 if length == 0 else 64.9 + 41.0 * (gc - 16.4) / length
    out = {"Length": float(length), "MoleculeIsRNA": 1.0 if mol_type == "RNA" else 0.0, "MoleculeIsDNA": 1.0 if mol_type == "DNA" else 0.0, "GC_Fraction": gc/length if length else 0.0, "AT_or_AU_Fraction": atu/length if length else 0.0, "Purine_Fraction": purine/length if length else 0.0, "Pyrimidine_Fraction": pyr/length if length else 0.0, "Ambiguous_Fraction": sum(1 for ch in seq if ch in AMBIGUOUS)/length if length else 0.0, "MolecularWeight_Da": float(mw), "Extinction260_Sum": float(ext), "Tm_Wallace_C": float(max(0.0, tm_wallace)), "Tm_GC_Adjusted_C": float(max(0.0, tm_gc)), "PhosphateChargeMagnitude": float(length), "ShannonEntropy": shannon_entropy(seq, alphabet), "LongestHomopolymer": float(longest_run(seq)), "Longest_GC_Run": float(longest_run(seq, {"G","C"})), "SelfComplementarity": complementarity_score(seq, mol_type), "GQuadruplexProxy_GGG_Count": float(seq.count("GGG")), "CpG_or_CpU_Count": float(seq.count("CG") + seq.count("CU")), "PolyA_Count": float(seq.count("AAA")), "PolyG_Count": float(seq.count("GGG"))}
    for base in alphabet:
        out[f"BaseFrac_{base}"] = counts.get(base,0)/length if length else 0.0; out[f"BaseCount_{base}"] = float(counts.get(base,0))
    if options.include_dinucleotide: out.update(kmer_frequencies(seq, alphabet, 2, "Dimer"))
    if options.include_trinucleotide: out.update(kmer_frequencies(seq, alphabet, 3, "Trimer"))
    return out

def empty_interaction_df(validated_df: pd.DataFrame) -> pd.DataFrame:
    base = validated_df[["Name", "Sequence"]].copy() if isinstance(validated_df, pd.DataFrame) and not validated_df.empty else pd.DataFrame(columns=["Name", "Sequence"])
    for col in INTERACTION_COLUMNS:
        if col not in base.columns: base[col] = 0.0
    return base

def safe_float(value: Any) -> float:
    value = pd.to_numeric(value, errors="coerce")
    return 0.0 if pd.isna(value) else float(value)

def interaction_descriptors(row: pd.Series) -> dict[str, float]:
    values = {col: safe_float(row.get(col, 0.0)) for col in INTERACTION_COLUMNS}; total = float(sum(values.values()))
    polar = values["H_bonds"] + values["Electrostatic_contacts"] + values["Salt_bridges"] + values["Water_bridges"]; stacking = values["Pi_stacking"] + values["Base_stacking_contacts"]
    out = {f"INT_{k}": v for k, v in values.items()}; out.update({"INT_TotalInteractions": total, "INT_PolarFraction": polar/total if total else 0.0, "INT_StackingFraction": stacking/total if total else 0.0, "INT_MetalFlag": 1.0 if values["Metal_coordination"] > 0 else 0.0, "INT_TargetContactDensity": values["Target_contact_count"]/total if total else 0.0})
    return out

def calculate_descriptors(validated_df: pd.DataFrame, interaction_df: pd.DataFrame | None, options: DescriptorOptions) -> pd.DataFrame:
    if validated_df is None or validated_df.empty: return pd.DataFrame()
    valid = validated_df[validated_df["Valid"] == True].copy()
    if valid.empty: return pd.DataFrame()
    interaction_map = interaction_df.copy() if options.include_interactions and isinstance(interaction_df, pd.DataFrame) and not interaction_df.empty else pd.DataFrame()
    if not interaction_map.empty: interaction_map["Sequence"] = interaction_map["Sequence"].astype(str)
    rows = []
    for _, row in valid.iterrows():
        seq = str(row["Sequence"]); mol_type = str(row.get("MoleculeType", infer_molecule_type(seq, options.molecule_type)))
        rec = {"Name": row.get("Name", ""), "Sequence": seq, "MoleculeType": mol_type}
        for col in valid.columns:
            if col not in rec and col not in {"Valid", "ValidationMessage"}: rec[col] = row[col]
        rec.update(sequence_descriptors(seq, mol_type, options))
        if options.include_interactions:
            match = pd.DataFrame()
            if not interaction_map.empty:
                name_match = interaction_map.get("Name", pd.Series(index=interaction_map.index, dtype=object)).astype(str) == str(row.get("Name", ""))
                seq_match = interaction_map.get("Sequence", pd.Series(index=interaction_map.index, dtype=object)).astype(str) == seq
                match = interaction_map[name_match | seq_match]
            rec.update(interaction_descriptors(match.iloc[0] if not match.empty else pd.Series(dtype=float)))
        rows.append(rec)
    return pd.DataFrame(rows)

def descriptor_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Name", "Sequence", "MoleculeType", "Valid", "ValidationMessage"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

def target_candidates(df: pd.DataFrame) -> list[str]:
    descriptor_prefixes = ("Base", "Dimer", "Trimer", "INT_", "Length", "GC_", "AT_", "Purine", "Pyrimidine", "Tm_", "Molecular", "Extinction", "Self", "Shannon", "Longest", "Phosphate", "Molecule", "GQuad", "CpG", "Poly", "Ambiguous")
    out = []
    for col in df.columns:
        if col in {"Name", "Sequence", "MoleculeType", "Valid", "ValidationMessage"}: continue
        if str(col).startswith(descriptor_prefixes): continue
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].astype(str).nunique() <= max(20, len(df)//2): out.append(col)
    return out

def normalize_task(task_type: str) -> str:
    return "classification" if "class" in task_type.lower() else "regression"

def infer_direction(target: str) -> str:
    key = (target or "").lower().replace("_", "").replace(" ", "")
    return "minimize" if any(tok in key for tok in LOWER_IS_BETTER_TOKENS) else "maximize"

def positive_lower_score(values: Any) -> pd.Series:
    raw = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float); out = pd.Series(np.nan, index=raw.index, dtype=float); valid = raw.dropna()
    if valid.empty: return out
    vmin, vmax = float(valid.min()), float(valid.max())
    if not np.isclose(vmin, vmax): out.loc[valid.index] = 100.0 * (vmax - valid) / (vmax - vmin)
    else: out.loc[valid.index] = np.where(valid <= 0, -valid, 100.0/(1.0+valid))
    return out.clip(lower=0, upper=100)

def normalized_higher_score(values: Any) -> pd.Series:
    raw = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float); out = pd.Series(np.nan, index=raw.index, dtype=float); valid = raw.dropna()
    if valid.empty: return out
    vmin, vmax = float(valid.min()), float(valid.max())
    out.loc[valid.index] = 100.0 if np.isclose(vmin, vmax) else 100.0 * (valid - vmin) / (vmax - vmin)
    return out.clip(lower=0, upper=100)

def regression_bins(y: pd.Series) -> pd.Series | None:
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().sum() < 12 or y_num.nunique(dropna=True) < 4: return None
    try: bins = pd.qcut(y_num, q=min(5, max(2, int(np.sqrt(len(y_num))))), labels=False, duplicates="drop")
    except Exception: return None
    counts = pd.Series(bins).value_counts(dropna=True)
    return pd.Series(bins, index=y.index) if not counts.empty and counts.min() >= 2 else None

def model_catalog(task_type: str, n_classes: int = 2) -> dict[str, Any]:
    if normalize_task(task_type) == "regression":
        return {"Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.001, max_iter=10000), "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000), "Random Forest": RandomForestRegressor(n_estimators=350, random_state=42, n_jobs=-1), "Extra Trees": ExtraTreesRegressor(n_estimators=450, random_state=42, n_jobs=-1), "Gradient Boosting": GradientBoostingRegressor(random_state=42), "SVR (RBF)": SVR(C=10.0, gamma="scale"), "kNN": KNeighborsRegressor(n_neighbors=5)}
    return {"Logistic Regression": LogisticRegression(max_iter=4000), "Random Forest": RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1), "Extra Trees": ExtraTreesClassifier(n_estimators=450, random_state=42, n_jobs=-1), "Gradient Boosting": GradientBoostingClassifier(random_state=42), "SVM (RBF)": SVC(C=5.0, gamma="scale", probability=True, random_state=42), "kNN": KNeighborsClassifier(n_neighbors=5), "Naive Bayes": GaussianNB()}

def make_pipeline(model: Any, task_type: str, n_features: int, n_samples: int) -> Pipeline:
    percentile = 100
    if n_features > max(40, n_samples * 2): percentile = max(10, min(60, int(100 * max(10, n_samples * 2) / n_features)))
    score_func = f_regression if normalize_task(task_type) == "regression" else f_classif
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold(0.0)), ("selector", SelectPercentile(score_func=score_func, percentile=percentile)), ("scaler", StandardScaler()), ("model", model)])

def train_models(desc_df: pd.DataFrame, target: str, task_type: str, selected_models: list[str], options: DescriptorOptions, test_size: float, cv_folds: int) -> dict[str, Any]:
    features = [c for c in descriptor_columns(desc_df) if c != target]
    if not features: raise ValueError("No numeric descriptors are available for modeling.")
    task = normalize_task(task_type); X = desc_df[features].replace([np.inf, -np.inf], np.nan)
    if task == "regression":
        y = pd.to_numeric(desc_df[target], errors="coerce"); mask = y.notna(); X, y = X.loc[mask], y.loc[mask]; encoder = None
    else:
        labels = desc_df[target].astype(str).str.strip(); mask = labels.notna() & (labels != ""); X, labels = X.loc[mask], labels.loc[mask]; encoder = LabelEncoder(); y = pd.Series(encoder.fit_transform(labels), index=labels.index, name=target)
    if len(X) < 8: raise ValueError("At least 8 valid aptamers are recommended for model training.")
    if task == "classification" and y.nunique() < 2: raise ValueError("Classification requires at least two classes.")
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else regression_bins(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    catalog = model_catalog(task, n_classes=int(y.nunique())); rows = []; bundles = {}
    for name in selected_models:
        if name not in catalog: continue
        pipe = make_pipeline(clone(catalog[name]), task, len(features), len(X_train)); pipe.fit(X_train, y_train); pred = pipe.predict(X_test); cv_summary = {}
        if task == "regression":
            metrics = {"R2_raw": float(r2_score(y_test, pred)), "RMSE": float(math.sqrt(mean_squared_error(y_test, pred))), "MAE": float(mean_absolute_error(y_test, pred))}
            cv = RepeatedKFold(n_splits=min(max(2, cv_folds), max(2, len(X_train)//2)), n_repeats=5, random_state=42)
            scores = cross_validate(pipe, X, y, cv=cv, scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"], n_jobs=-1, error_score=np.nan)
            cv_summary = {"CV_R2_raw_mean": float(np.nanmean(scores["test_r2"])), "CV_R2_raw_std": float(np.nanstd(scores["test_r2"])), "CV_RMSE": float(np.nanmean(-scores["test_neg_root_mean_squared_error"])), "CV_MAE": float(np.nanmean(-scores["test_neg_mean_absolute_error"]))}
            primary = cv_summary["CV_R2_raw_mean"] if np.isfinite(cv_summary["CV_R2_raw_mean"]) else metrics["R2_raw"]
            row = {"Model": name, **metrics, **cv_summary, "ModelQuality_0_100": max(0.0, min(100.0, primary*100)), "HoldoutQuality_0_100": max(0.0, min(100.0, metrics["R2_raw"]*100)), "CVQuality_0_100": max(0.0, min(100.0, cv_summary["CV_R2_raw_mean"]*100))}
        else:
            decoded_true = encoder.inverse_transform(np.asarray(y_test, dtype=int)) if encoder is not None else y_test; decoded_pred = encoder.inverse_transform(np.asarray(pred, dtype=int)) if encoder is not None else pred; average = "binary" if len(np.unique(y)) == 2 else "weighted"
            metrics = {"Accuracy": float(accuracy_score(decoded_true, decoded_pred)), "Precision": float(precision_score(decoded_true, decoded_pred, average=average, zero_division=0)), "Recall": float(recall_score(decoded_true, decoded_pred, average=average, zero_division=0)), "F1": float(f1_score(decoded_true, decoded_pred, average=average, zero_division=0))}
            if hasattr(pipe, "predict_proba") and len(np.unique(y)) == 2: metrics["ROC_AUC"] = float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]))
            min_class = int(y.value_counts().min())
            if min_class >= 2:
                cv = StratifiedKFold(n_splits=min(cv_folds, min_class), shuffle=True, random_state=42); scores = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_weighted"], n_jobs=-1, error_score=np.nan); cv_summary = {"CV_F1": float(np.nanmean(scores["test_f1_weighted"])), "CV_Accuracy": float(np.nanmean(scores["test_accuracy"]))}
            primary = cv_summary.get("CV_F1", metrics["F1"]); row = {"Model": name, **metrics, **cv_summary, "ModelQuality_0_100": max(0.0, min(100.0, primary*100))}
        rows.append(row); final_pipe = make_pipeline(clone(catalog[name]), task, len(features), len(X)); final_pipe.fit(X, y); bundles[name] = ModelBundle(name, task, target, features, options, final_pipe, encoder, metrics, cv_summary, datetime.now().isoformat(timespec="seconds"))
    if not rows: raise ValueError("No selected model could be trained.")
    leaderboard = pd.DataFrame(rows).sort_values("ModelQuality_0_100", ascending=False).reset_index(drop=True); best_name = str(leaderboard.iloc[0]["Model"])
    return {"leaderboard": leaderboard, "bundles": bundles, "best_name": best_name, "best": bundles[best_name], "features": features}

def predict_with_bundle(bundle: ModelBundle, desc_df: pd.DataFrame) -> pd.DataFrame:
    X = desc_df.reindex(columns=bundle.features).replace([np.inf, -np.inf], np.nan); pred = bundle.pipeline.predict(X); out = desc_df[["Name", "Sequence", "MoleculeType"]].copy()
    if bundle.task_type == "regression":
        raw = pd.Series(pd.to_numeric(pred, errors="coerce"), index=out.index, dtype=float)
        if infer_direction(bundle.target) == "minimize": out["Prediction"] = positive_lower_score(raw); out["PredictionMeaning"] = "Positive optimized score; higher is better."; out["RankingScore"] = out["Prediction"]; out["RawModelPrediction"] = raw
        else: out["Prediction"] = raw; out["PredictionMeaning"] = "Predicted target value; higher is better."; out["RankingScore"] = normalized_higher_score(raw)
        out["NormalizedScore_0_100"] = normalized_higher_score(out["RankingScore"])
    else:
        labels = bundle.label_encoder.inverse_transform(pred.astype(int)) if bundle.label_encoder is not None else pred; out["PredictedClass"] = labels
        if hasattr(bundle.pipeline, "predict_proba"): out["PredictionProbability"] = np.max(bundle.pipeline.predict_proba(X), axis=1); out["RankingScore"] = out["PredictionProbability"] * 100
        else: out["RankingScore"] = 0.0
    out = out.sort_values("RankingScore", ascending=False).reset_index(drop=True); out.insert(0, "Rank", np.arange(1, len(out)+1)); return out

def prediction_display(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c != "RawModelPrediction"]].copy()

def dataframe_csv(df: pd.DataFrame) -> bytes: return df.to_csv(index=False).encode("utf-8")

def dataframe_excel(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sheets.items(): (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()

def plot_model_comparison(df: pd.DataFrame) -> go.Figure: return px.bar(df, x="Model", y="ModelQuality_0_100", template=PLOT_TEMPLATE, title="Model Comparison")
def plot_prediction_ranking(df: pd.DataFrame) -> go.Figure: return px.bar(df.head(30), x="Name", y="RankingScore", color="MoleculeType" if "MoleculeType" in df.columns else None, template=PLOT_TEMPLATE, title="Aptamer Ranking")

def plot_pca(desc_df: pd.DataFrame, color: str | None = None) -> go.Figure:
    feats = descriptor_columns(desc_df)
    if len(feats) < 2 or len(desc_df) < 2: return px.scatter(title="PCA needs at least two samples and two descriptors", template=PLOT_TEMPLATE)
    X = SimpleImputer(strategy="median").fit_transform(desc_df[feats]); X = StandardScaler().fit_transform(X); pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    plot_df = desc_df[["Name", "MoleculeType"]].copy() if "MoleculeType" in desc_df.columns else desc_df[["Name"]].copy(); plot_df["PC1"] = pcs[:,0]; plot_df["PC2"] = pcs[:,1]
    return px.scatter(plot_df, x="PC1", y="PC2", hover_name="Name", color=color if color in plot_df.columns else None, template=PLOT_TEMPLATE, title="PCA Aptamer Descriptor Space")

def render_home() -> None:
    st.markdown("""<div class="apt-hero"><h1>Aptamer QSAR Tool</h1><p>RNA/DNA aptamer descriptor engineering, interaction-aware QSAR modeling, prediction, visualization, and report export.</p><p><b>Developed by Ahmed G. Soliman.</b></p></div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3); c1.metric("Sequence descriptors", "DNA/RNA"); c2.metric("Interaction descriptors", len(INTERACTION_COLUMNS)); c3.metric("ML models", "Regression/Class")
    st.info("Workflow: upload aptamers, add interaction counts if available, calculate descriptors, train a model, then rank new aptamers.")

def render_input() -> None:
    st.subheader("1) Input RNA/DNA Aptamers"); molecule_type = st.radio("Molecule type", ["Auto", "DNA", "RNA"], horizontal=True)
    if st.button("Load Example Aptamer Dataset", use_container_width=True): st.session_state.raw_df = pd.read_csv(DATA_DIR / "example_aptamers.csv"); st.success("Example dataset loaded.")
    left,right = st.columns(2)
    with left: text = st.text_area("Paste sequences or FASTA", height=180, placeholder=">apt1\nGGGTTAGGGTTAGGG\napt2, ACGTACGT")
    with right: uploaded = st.file_uploader("Upload CSV, Excel, TXT, FASTA", type=["csv", "xlsx", "xls", "txt", "fasta", "fa", "fna"])
    manual_df = parse_manual_sequences(text); uploaded_df = read_uploaded_file(uploaded) if uploaded is not None else pd.DataFrame()
    if uploaded_df is not None and not uploaded_df.empty: st.session_state.raw_df = uploaded_df
    elif not manual_df.empty: st.session_state.raw_df = manual_df
    raw_df = st.session_state.raw_df
    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        seq_col_guess = guess_sequence_column(raw_df); seq_col = st.selectbox("Sequence column", raw_df.columns, index=list(raw_df.columns).index(seq_col_guess) if seq_col_guess in raw_df.columns else 0)
        validated = validate_dataframe(raw_df, seq_col, molecule_type); st.session_state.validated_df = validated
        good = validated[validated["Valid"] == True]; bad = validated[validated["Valid"] == False]
        c1,c2,c3 = st.columns(3); c1.metric("Valid aptamers", len(good)); c2.metric("Invalid rows", len(bad)); c3.metric("Duplicates", int(good.duplicated("Sequence").sum()))
        st.dataframe(validated, use_container_width=True, height=320)

def render_interactions() -> None:
    st.subheader("2) Aptamer-Target Interaction Features"); validated = st.session_state.validated_df
    if not isinstance(validated, pd.DataFrame) or validated.empty: st.info("Load and validate aptamers first."); return
    st.markdown("Enter experimentally measured, docking-derived, MD-derived, or manually curated interaction counts per aptamer. Leave unknown values as 0.")
    uploaded = st.file_uploader("Optional: upload interaction table", type=["csv", "xlsx", "xls"], key="interaction_upload")
    if uploaded is not None: st.session_state.interaction_df = read_uploaded_file(uploaded)
    if st.session_state.interaction_df is None or not isinstance(st.session_state.interaction_df, pd.DataFrame): st.session_state.interaction_df = empty_interaction_df(validated[validated["Valid"] == True])
    editable = st.data_editor(st.session_state.interaction_df, use_container_width=True, height=360, num_rows="dynamic"); st.session_state.interaction_df = editable
    available = [c for c in INTERACTION_COLUMNS if c in editable.columns]
    if available:
        totals = editable[available].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sort_values(ascending=False)
        st.plotly_chart(px.bar(totals.reset_index(), x="index", y=0, labels={"index":"Interaction type",0:"Total count"}, template=PLOT_TEMPLATE, title="Interaction Type Totals"), use_container_width=True)

def render_descriptors() -> None:
    st.subheader("3) Calculate Aptamer Descriptors"); validated = st.session_state.validated_df
    if not isinstance(validated, pd.DataFrame) or validated.empty: st.info("Load aptamers first."); return
    c1,c2,c3 = st.columns(3); include_dimer = c1.checkbox("Include dinucleotide composition", value=True); include_trimer = c2.checkbox("Include trinucleotide composition", value=False); include_int = c3.checkbox("Include interaction descriptors", value=True)
    options = DescriptorOptions(include_dinucleotide=include_dimer, include_trinucleotide=include_trimer, include_interactions=include_int); st.session_state.descriptor_options = options
    if st.button("Calculate Descriptors", type="primary"):
        desc = calculate_descriptors(validated, st.session_state.interaction_df, options); st.session_state.descriptor_df = desc; st.success(f"Calculated {len(descriptor_columns(desc))} numeric descriptors for {len(desc)} aptamers.")
    desc = st.session_state.descriptor_df
    if isinstance(desc, pd.DataFrame) and not desc.empty: st.dataframe(desc, use_container_width=True, height=360); st.download_button("Download Descriptor Table", data=dataframe_csv(desc), file_name="aptamer_descriptors.csv", mime="text/csv")

def render_train() -> None:
    st.subheader("4) Train QSAR Model"); desc = st.session_state.descriptor_df
    if not isinstance(desc, pd.DataFrame) or desc.empty: st.info("Calculate descriptors first."); return
    targets = target_candidates(desc)
    if not targets: st.warning("No target/activity column was found. Add Kd_nM, ActivityScore, IC50, BindingScore, etc."); return
    target = st.selectbox("Target column", targets, index=0); task_type = st.radio("Task", ["Regression", "Classification"], horizontal=True)
    catalog = model_catalog(task_type, n_classes=2); default = [m for m in ["Extra Trees", "Random Forest", "Gradient Boosting", "SVR (RBF)", "Ridge"] if m in catalog]
    selected = st.multiselect("Models", list(catalog.keys()), default=default or list(catalog.keys())[:4])
    c1,c2 = st.columns(2); test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05); cv_folds = c2.slider("CV folds", 2, 10, 5, 1)
    if st.button("Train & Compare", type="primary"):
        try:
            with st.spinner("Training aptamer QSAR models..."): result = train_models(desc, target, task_type, selected, st.session_state.descriptor_options, float(test_size), int(cv_folds))
            st.session_state.training_result = result; st.session_state.active_bundle = result["best"]; st.success(f"Training completed. Best model: {result['best_name']}")
        except Exception as exc: st.error(f"Training failed: {exc}")
    result = st.session_state.training_result
    if isinstance(result, dict):
        st.dataframe(result["leaderboard"], use_container_width=True); st.plotly_chart(plot_model_comparison(result["leaderboard"]), use_container_width=True)
        with st.expander("Raw scientific R2 diagnostics"):
            raw_cols = [c for c in ["Model", "R2_raw", "CV_R2_raw_mean", "CV_R2_raw_std", "RMSE", "MAE"] if c in result["leaderboard"].columns]; st.dataframe(result["leaderboard"][raw_cols], use_container_width=True)
        bundle = result["best"]; st.write("Selected model summary"); st.dataframe(pd.DataFrame({"Item":["Model","Task","Target","Descriptors","Created"],"Value":[bundle.model_name,bundle.task_type,bundle.target,len(bundle.features),bundle.created_at]}), hide_index=True, use_container_width=True)
        if st.button("Save Best Model Locally"):
            path = MODEL_DIR / f"aptamer_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"; joblib.dump(bundle, path); st.success(f"Saved model: {path.name}")

def render_evaluate() -> None:
    st.subheader("5) Evaluate"); result = st.session_state.training_result
    if not isinstance(result, dict): st.info("Train a model first."); return
    leaderboard = result["leaderboard"]; best = result["best"]; st.metric("Best model", best.model_name); row = leaderboard[leaderboard["Model"] == best.model_name].iloc[0]
    if best.task_type == "regression":
        c1,c2,c3,c4 = st.columns(4); c1.metric("Model Quality", f"{row.get('ModelQuality_0_100',0):.2f}/100"); c2.metric("CV Quality", f"{row.get('CVQuality_0_100',0):.2f}/100"); c3.metric("RMSE", f"{row.get('RMSE',np.nan):.4f}"); c4.metric("MAE", f"{row.get('MAE',np.nan):.4f}")
    else:
        c1,c2,c3,c4 = st.columns(4); c1.metric("Accuracy", f"{row.get('Accuracy',np.nan):.3f}"); c2.metric("Precision", f"{row.get('Precision',np.nan):.3f}"); c3.metric("Recall", f"{row.get('Recall',np.nan):.3f}"); c4.metric("F1", f"{row.get('F1',np.nan):.3f}")

def render_predict() -> None:
    st.subheader("6) Predict New Aptamers"); bundle = st.session_state.active_bundle
    if not isinstance(bundle, ModelBundle): st.info("Train a model first."); return
    text = st.text_area("Paste new aptamer sequences", height=150, placeholder=">candidate1\nGGGTTAGGGTTAGGG")
    uploaded = st.file_uploader("Or upload prediction file", type=["csv","xlsx","xls","txt","fasta","fa"], key="pred_upload")
    raw = read_uploaded_file(uploaded) if uploaded is not None else parse_manual_sequences(text)
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        seq_col = guess_sequence_column(raw) or raw.columns[0]; validated = validate_dataframe(raw, seq_col, bundle.descriptor_options.molecule_type); st.dataframe(validated, use_container_width=True, height=220)
        st.markdown("Optional interaction descriptors for prediction candidates")
        pred_interactions = st.data_editor(empty_interaction_df(validated[validated["Valid"] == True]), use_container_width=True, height=260, num_rows="dynamic", key="pred_int_editor")
        if st.button("Run Aptamer Prediction", type="primary"):
            desc = calculate_descriptors(validated, pred_interactions, bundle.descriptor_options); pred = predict_with_bundle(bundle, desc); st.session_state.prediction_df = pred; st.session_state.prediction_descriptor_df = desc
    pred = st.session_state.prediction_df
    if isinstance(pred, pd.DataFrame) and not pred.empty:
        shown = prediction_display(pred); st.dataframe(shown, use_container_width=True, height=300); st.plotly_chart(plot_prediction_ranking(pred), use_container_width=True)
        if "RawModelPrediction" in pred.columns:
            with st.expander("Raw model diagnostics"):
                st.dataframe(pred[[c for c in ["Rank","Name","Sequence","RawModelPrediction","Prediction","RankingScore"] if c in pred.columns]], use_container_width=True)
        st.download_button("Download Predictions CSV", data=dataframe_csv(shown), file_name="aptamer_predictions.csv", mime="text/csv")

def render_visuals() -> None:
    st.subheader("7) Visualizations"); desc = st.session_state.descriptor_df
    if not isinstance(desc, pd.DataFrame) or desc.empty: st.info("Calculate descriptors first."); return
    st.plotly_chart(plot_pca(desc, color="MoleculeType"), use_container_width=True)
    feats = descriptor_columns(desc); selected = st.selectbox("Descriptor distribution", feats, index=0)
    st.plotly_chart(px.histogram(desc, x=selected, color="MoleculeType" if "MoleculeType" in desc.columns else None, marginal="box", template=PLOT_TEMPLATE), use_container_width=True)
    int_cols = [c for c in desc.columns if c.startswith("INT_") and c != "INT_TotalInteractions"]
    if int_cols:
        totals = desc[int_cols].sum().sort_values(ascending=False).head(20)
        st.plotly_chart(px.bar(totals.reset_index(), x="index", y=0, labels={"index":"Interaction descriptor",0:"Total"}, template=PLOT_TEMPLATE, title="Interaction Descriptor Summary"), use_container_width=True)

def render_export() -> None:
    st.subheader("8) Export"); tables = {}
    for name,key in [("Validated","validated_df"),("Interactions","interaction_df"),("Descriptors","descriptor_df"),("Predictions","prediction_df")]:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty: tables[name] = prediction_display(df) if name == "Predictions" else df
    result = st.session_state.training_result
    if isinstance(result, dict): tables["ModelLeaderboard"] = result["leaderboard"]
    if not tables: st.info("Nothing to export yet."); return
    st.download_button("Download Full Workbook", data=dataframe_excel(tables), file_name="aptamer_qsar_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    html = f"<html><body><h1>Aptamer QSAR Report</h1><p>Generated: {datetime.now().isoformat(timespec='seconds')}</p><p>Developed by {DEVELOPER_NAME}. This report summarizes RNA/DNA descriptors, user-entered interaction features, model comparison, and predictions.</p></body></html>"
    st.download_button("Download HTML Summary", data=html.encode("utf-8"), file_name="aptamer_qsar_summary.html", mime="text/html")

def render_about() -> None:
    st.subheader("About")
    st.markdown(f"""
### Developed by {DEVELOPER_NAME}
Portfolio: {DEVELOPER_PORTFOLIO}

### Scientific basis
This tool is designed for RNA and DNA aptamers. It calculates sequence-derived nucleic-acid descriptors including length, GC content, base composition, dinucleotide/trinucleotide composition, molecular-weight approximation, extinction proxy, melting-temperature approximations, entropy, homopolymer runs, G-rich motifs, and self-complementarity proxy.

The tool also supports user-entered aptamer-target interaction descriptors. These can come from experiments, docking, molecular dynamics, structural annotation, or manual curation. Interaction features include hydrogen bonds, hydrophobic contacts, pi-stacking, electrostatic contacts, salt bridges, metal coordination, van der Waals contacts, water bridges, base-pairing contacts, base-stacking contacts, and total target-contact counts.

Machine-learning models use scikit-learn pipelines with imputation, variance filtering, feature selection, scaling, repeated cross-validation, and regression/classification evaluation.
""")

def main() -> None:
    init_state(); st.sidebar.title(APP_NAME); st.sidebar.caption(f"Developed by {DEVELOPER_NAME}")
    tabs = st.tabs(["Home", "Input", "Interactions", "Descriptors", "Train", "Evaluate", "Predict", "Visualizations", "Export", "About"])
    with tabs[0]: render_home()
    with tabs[1]: render_input()
    with tabs[2]: render_interactions()
    with tabs[3]: render_descriptors()
    with tabs[4]: render_train()
    with tabs[5]: render_evaluate()
    with tabs[6]: render_predict()
    with tabs[7]: render_visuals()
    with tabs[8]: render_export()
    with tabs[9]: render_about()

if __name__ == "__main__": main()

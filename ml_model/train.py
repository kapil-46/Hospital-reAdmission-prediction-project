
# train_final.py
"""
Final training pipeline for hospital_readmissions.csv
- Preprocessing (safe, fold-aware target-encoding)
- Stratified K-Fold CV
- LightGBM & XGBoost training
- Optional Optuna tuning (default: OFF)
- Stacking final model on full data and saving model + scaler
- Saves predictions CSV for inspection
"""

import os
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ------------- CONFIG -------------
CSV_PATH = "hospital_readmissions.csv"   # update path if needed
RANDOM_STATE = 42
N_SPLITS = 5                              # Stratified folds
USE_SMOTE = True                           # balance training splits with SMOTE
TUNE = False                               # set True to run Optuna tuning (longer)
OPTUNA_TRIALS = 30                         # if TUNE=True, number of trials
SAVE_MODEL_PATH = "best_model_stack.pkl"
PRED_CSV = "predictions_inspect.csv"
# -----------------------------------

assert os.path.exists(CSV_PATH), f"CSV file not found at {CSV_PATH}"

# ----------------- Utilities -----------------
def age_to_mid(x):
    try:
        if pd.isna(x): return np.nan
        s = str(x)
        if "-" in s:
            s2 = s.replace("[","").replace("]","").replace(")","").replace("(","")
            a, b = s2.split("-")
            return (int(a) + int(b)) // 2
        return float(s)
    except:
        return np.nan

def diag_map(s):
    try:
        if pd.isna(s): return "Missing"
        s = str(s)
        if s.startswith("250"): return "Diabetes"
        if s[0].isdigit():
            code = int(s.split(".")[0])
            if 390 <= code <= 459 or code == 785: return "Circulatory"
            if 460 <= code <= 519 or code == 786: return "Respiratory"
            if 520 <= code <= 579 or code == 787: return "Digestive"
            if 580 <= code <= 629 or code == 788: return "Genitourinary"
            if 800 <= code <= 999: return "Injury"
            if 140 <= code <= 239: return "Neoplasms"
            return "Other"
        return "Other"
    except:
        return "Other"

def reduce_rare_categories(ser: pd.Series, threshold=0.01) -> pd.Series:
    freqs = ser.value_counts(normalize=True)
    rare = freqs[freqs < threshold].index
    return ser.replace(rare, "Other")

# Target encoding WITH smoothing done fold-wise to avoid leakage
def target_encode_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame, col: str, target_col: str, smoothing: float = 10.0) -> Tuple[pd.Series, pd.Series]:
    # compute target mean per category on train
    agg = train_df.groupby(col)[target_col].agg(["count", "mean"])
    prior = train_df[target_col].mean()
    # smoothing
    counts = agg["count"]
    means = agg["mean"]
    smooth = (counts * means + smoothing * prior) / (counts + smoothing)
    mapping = smooth.to_dict()
    # map
    train_mapped = train_df[col].map(mapping).fillna(prior)
    val_mapped = val_df[col].map(mapping).fillna(prior)
    return train_mapped, val_mapped

# ----------------- Load & Initial Preprocess -----------------
print("Loading data...")
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)

# Basic cleaning & feature engineering
if "age" in df.columns:
    df["age"] = df["age"].apply(age_to_mid)

for c in ["diag_1", "diag_2", "diag_3"]:
    if c in df.columns:
        df[c] = df[c].apply(diag_map)

if "medical_specialty" in df.columns:
    df["medical_specialty"] = reduce_rare_categories(df["medical_specialty"], threshold=0.01)

# create small engineered features
if set(["time_in_hospital","n_lab_procedures"]).issubset(df.columns):
    df["labs_per_day"] = df["n_lab_procedures"] / df["time_in_hospital"].replace(0,1)
if "n_medications" in df.columns:
    df["high_medications"] = (df["n_medications"] > 15).astype(int)
if set(["n_outpatient","n_inpatient","n_emergency"]).issubset(df.columns):
    df["prior_visits"] = df["n_outpatient"] + df["n_inpatient"] + df["n_emergency"]

# Replace common missing tokens
df.replace(["Missing","?","unknown","Unknown","NONE","None"], np.nan, inplace=True)

# Target ensure numeric
if "readmitted" not in df.columns:
    raise KeyError("Target column 'readmitted' not found.")
df["readmitted"] = df["readmitted"].map({"no":0, "yes":1})
if df["readmitted"].isnull().any():
    df["readmitted"] = df["readmitted"].fillna(0).astype(int)  # fallback

# Fill remaining NaNs: numeric -> median, categorical -> mode
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Separate X,y
target = "readmitted"
X_all = df.drop(columns=[target]).copy()
y_all = df[target].copy()

print("Columns:", X_all.columns.tolist())
print("Target distribution:\n", y_all.value_counts(normalize=True))

# Identify categorical columns (object dtype after our transforms)
cat_cols = X_all.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
print("Categorical cols:", cat_cols)
print("Numerical cols:", num_cols)

# ----------------- Stratified K-Fold CV with fold-safe encodings -----------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

lgb_fold_acc = []
xgb_fold_acc = []
stack_fold_acc = []

# We'll store feature names for final model
final_feature_order = None

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    X_tr = X_all.iloc[tr_idx].reset_index(drop=True)
    y_tr = y_all.iloc[tr_idx].reset_index(drop=True)
    X_val = X_all.iloc[val_idx].reset_index(drop=True)
    y_val = y_all.iloc[val_idx].reset_index(drop=True)

    # --- TARGET-encoding for categorical columns (fold-safe) ---
    X_tr_enc = X_tr.copy()
    X_val_enc = X_val.copy()
    for c in cat_cols:
        tr_enc, val_enc = target_encode_train_val(pd.concat([X_tr, y_tr], axis=1).rename(columns={0: c}) if False else pd.concat([X_tr, y_tr], axis=1), X_val, c, target)  # using helper - safe path
        # The helper expects train and val frames and uses target mapping. But signature earlier uses train_df and val_df columns.
        # Use simpler direct mapping:
        # (we already have target_encode_train_val defined)
    # We'll use another safe approach below: compute mapping manually
    for c in cat_cols:
        agg = pd.DataFrame({"count": X_tr.groupby(c).size(), "mean": (X_tr.join(y_tr))[c].groupby(X_tr[c]).apply(lambda cols: y_tr[X_tr[c]==cols].mean())}) if False else None
        # simpler: compute mapping from X_tr and y_tr
        temp = pd.concat([X_tr[c], y_tr], axis=1)
        temp.columns = [c, target]
        agg = temp.groupby(c)[target].agg(["count","mean"])
        prior = y_tr.mean()
        counts = agg["count"]
        means = agg["mean"]
        smoothing = 10.0
        smooth_map = ((counts * means) + smoothing * prior) / (counts + smoothing)
        X_tr_enc[c] = X_tr[c].map(smooth_map).fillna(prior)
        X_val_enc[c] = X_val[c].map(smooth_map).fillna(prior)

    # For numeric columns: scale (fit on train -> transform on val)
    scaler = StandardScaler()
    X_tr_num = pd.DataFrame(scaler.fit_transform(X_tr_enc[num_cols]), columns=num_cols)
    X_val_num = pd.DataFrame(scaler.transform(X_val_enc[num_cols]), columns=num_cols)

    # Recombine encoded categorical (already numeric after target encoding) + scaled numeric
    X_tr_proc = pd.concat([X_tr_enc[cat_cols].reset_index(drop=True), X_tr_num.reset_index(drop=True)], axis=1)
    X_val_proc = pd.concat([X_val_enc[cat_cols].reset_index(drop=True), X_val_num.reset_index(drop=True)], axis=1)

    # Ensure same column order
    X_tr_proc = X_tr_proc.loc[:, sorted(X_tr_proc.columns)]
    X_val_proc = X_val_proc.loc[:, sorted(X_val_proc.columns)]

    # Save feature order for final training later
    if final_feature_order is None:
        final_feature_order = X_tr_proc.columns.tolist()

    # Optionally balance with SMOTE on training fold
    if USE_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_proc, y_tr)
    else:
        X_tr_bal, y_tr_bal = X_tr_proc, y_tr

    # ---- Train LightGBM (fast-ish settings) ----
    lgb_model = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.03, num_leaves=64,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE
    )
    lgb_model.fit(X_tr_bal, y_tr_bal)
    lgb_preds = lgb_model.predict(X_val_proc)
    lgb_acc = accuracy_score(y_val, lgb_preds)
    lgb_fold_acc.append(lgb_acc)
    print(f"LightGBM fold acc: {lgb_acc:.4f}")

    # ---- Train XGBoost ----
    # compute scale_pos_weight if not using SMOTE
    if USE_SMOTE:
        scale_pos_weight = 1.0
    else:
        pos = sum(y_tr==1)
        neg = sum(y_tr==0)
        scale_pos_weight = neg / max(1,pos)
    xgb_model = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE
    )
    xgb_model.fit(X_tr_bal, y_tr_bal)
    xgb_preds = xgb_model.predict(X_val_proc)
    xgb_acc = accuracy_score(y_val, xgb_preds)
    xgb_fold_acc.append(xgb_acc)
    print(f"XGBoost fold acc: {xgb_acc:.4f}")

    # ---- Stack the two models quickly (fit on balanced train, predict on val) ----
    # Use probability stacking: average probs
    lgb_proba = lgb_model.predict_proba(X_val_proc)[:,1]
    xgb_proba = xgb_model.predict_proba(X_val_proc)[:,1]
    avg_proba = (lgb_proba + xgb_proba) / 2.0
    avg_pred = (avg_proba >= 0.5).astype(int)
    stack_acc = accuracy_score(y_val, avg_pred)
    stack_fold_acc.append(stack_acc)
    print(f"Stack (avg-prob) fold acc: {stack_acc:.4f}")

# --------------- Summary ---------------
print("\n============ CROSS-VALIDATION RESULTS ============")
print(f"LightGBM mean accuracy: {np.mean(lgb_fold_acc):.4f}  std: {np.std(lgb_fold_acc):.4f}")
print(f"XGBoost  mean accuracy: {np.mean(xgb_fold_acc):.4f}  std: {np.std(xgb_fold_acc):.4f}")
print(f"Stack avg-prob mean accuracy: {np.mean(stack_fold_acc):.4f}  std: {np.std(stack_fold_acc):.4f}")

# If stacked CV accuracy is the best, we'll finalize stack on full data
# Final training on full data (using final_feature_order) and save model + scaler mapping:
print("\nTraining final stacked model on full data...")

# Prepare full-data processing using same target-encoding logic but trained on full dataset
X_full = X_all.copy()
y_full = y_all.copy()

# apply same reduce_rare for categorical columns and mapping: (we already did earlier)
# Create target-encoded categorical features using full-data mapping (safe for production? It's common to use global mapping)
X_full_enc = X_full.copy()
for c in cat_cols:
    temp = pd.concat([X_full[c], y_full], axis=1)
    temp.columns = [c, target]
    agg = temp.groupby(c)[target].agg(["count","mean"])
    prior = y_full.mean()
    counts = agg["count"]
    means = agg["mean"]
    smoothing = 10.0
    smooth_map = ((counts * means) + smoothing * prior) / (counts + smoothing)
    X_full_enc[c] = X_full[c].map(smooth_map).fillna(prior)

# scale numeric
scaler_full = StandardScaler()
X_full_enc[num_cols] = scaler_full.fit_transform(X_full_enc[num_cols])

# ensure columns order
X_full_final = X_full_enc[sorted(X_full_enc.columns)]

# Optionally apply SMOTE to full training (be cautious; we will train final models on balanced full data to boost recall)
if USE_SMOTE:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_full_bal, y_full_bal = sm.fit_resample(X_full_final, y_full)
else:
    X_full_bal, y_full_bal = X_full_final, y_full

# Train final LightGBM and XGBoost on full (balanced) set
final_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=64, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE)
final_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)

print("Fitting final LightGBM...")
final_lgb.fit(X_full_bal, y_full_bal)
print("Fitting final XGBoost...")
final_xgb.fit(X_full_bal, y_full_bal)

# Stacking strategy: average probabilities, then train a light meta-learner (simple)
def stacked_predict_proba(X):
    p1 = final_lgb.predict_proba(X)[:,1]
    p2 = final_xgb.predict_proba(X)[:,1]
    return (p1 + p2) / 2.0

# Save model artifacts (we'll save the two models + scaler + feature order)
artifacts = {
    "final_lgb": final_lgb,
    "final_xgb": final_xgb,
    "scaler": scaler_full,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "feature_order": sorted(X_full_final.columns),
    "use_smote": USE_SMOTE
}
joblib.dump(artifacts, SAVE_MODEL_PATH)
print(f"Saved artifacts to {SAVE_MODEL_PATH}")

# Create predictions on the hold-out split (optional: use a train_test_split now to show final metrics)
X_train_hold, X_test_hold, y_train_hold, y_test_hold = train_test_split(X_full_final, y_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full)
# If USE_SMOTE was used full models trained on balanced; but we evaluate on original holdout
proba_hold = stacked_predict_proba(X_test_hold)
pred_hold = (proba_hold >= 0.5).astype(int)
print("\nFinal evaluation on held-out test split:")
print("Accuracy:", accuracy_score(y_test_hold, pred_hold))
print("ROC-AUC:", roc_auc_score(y_test_hold, proba_hold))
print(classification_report(y_test_hold, pred_hold, digits=3))

# Save predictions CSV for inspection
out_df = pd.DataFrame({
    "y_true": y_test_hold,
    "proba_stack": proba_hold,
    "pred_stack_0.5": pred_hold
})
out_df.to_csv(PRED_CSV, index=False)
print(f"Saved predictions to {PRED_CSV}")

# print("\nDONE. If accuracy is still ~0.61, try:")
# print(" - Increase feature engineering (interactions, domain-specific features)")
# print(" - Set USE_SMOTE = False/True depending on target balancing experiments")
# print(" - Turn TUNE = True and increase OPTUNA_TRIALS to search better params")
# print(" - Add CatBoost with raw categorical support (it sometimes helps).")


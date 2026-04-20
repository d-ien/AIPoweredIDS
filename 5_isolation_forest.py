import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import kagglehub
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import( classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score)

#style
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_PALETTE = {0: "#4878CF", 1: "#D65F5F"}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def save_fig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" saved -> figures/{name}")

#load data
print("Loading dataset...")
path = kagglehub.dataset_download("arunavakrchakraborty/financial-payment-services-fraud-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f" Loaded: {df.shape[0]:,} rows")

"""PREPROCESSING"""
df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

df_filtered["orig_balance_delta"] = (df_filtered["newbalanceOrig"] - df_filtered["oldbalanceOrg"])
df_filtered["dest_balance_delta"] = (df_filtered["newbalanceDest"] - df_filtered["oldbalanceDest"])
df_filtered["orig_drained"] = (df_filtered["newbalanceOrig"] == 0).astype(int)
df_filtered["type_encoded"] = (df_filtered["type"] == "TRANSFER").astype(int)

FEATURES = [
    "step", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_delta", "dest_balance_delta",
    "orig_drained", "type_encoded"
]
TARGET = "isFraud"

X = df_filtered[FEATURES]
y = df_filtered[TARGET]

#standard stratified split with the same params from phase 2 and 3 to make sure the test set is identical
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f" Fraud in test test: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

""" ISOLATION FOREST - TRAINING """
# take only legit transactions from training set
X_train_legit = X_train_scaled[y_train == 0]
print(f"\n Training Isolation Forest on {X_train_legit.shape[0]:,}"
      f"only legitimate transactions")
print(f" Fraud cases not included in training: {(y_train == 1).sum():.}")

fraud_rate = y.mean()
print(f" contamination set to know fraud rate: {fraud_rate:.6f}")

iso = IsolationForest(
    n_estimators=200,
    contamination=fraud_rate,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

iso.fit(X_train_legit)
print("Isolation Forest trained on legitimate transactions only")

"""PREDICTION AND SCORING

-ve scores = more anolalous and +ve scores = more normal but will convert -1 to 1 to fraud and 1 to 0 for legit
to compare against the isFaud labels"""

#negate the anomoly scores so higher = more suspicicious
anomaly_scores = -iso.decision_function(X_test_scaled)

# -1 anomoly -> 1 fraud and 1 normal -> 0 legit
raw_predictions = iso.predict(X_test_scaled)
y_pred = np.where(raw_predictions == -1, 1, 0)

print(f"\n Transactions flagged as anomalous: {y_pred.sum():,}"
      f"({y_pred.mean()*100:.2f}% of test set)")

"""EVALUATION"""

roc_auc = roc_auc_score(y_test, anomaly_scores)
avg_prc = average_precision_score(y_test, anomaly_scores)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 55)
print(" ISOLATION FOREST RESULTS")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0))
print(f" ROC-AUC : {roc_auc:.4f}")
print(f" PR-AUC : {avg_prc:.4f}")
print("=" * 55)

""" FIG 24 - CONFUSION MATRIX """

###############################################################################
### Multidimensional Scaling in Gini Pseudo Metric Spaces on UCI datasets #####
###############################################################################

import pandas as pd
import gc
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
import math
from collections import defaultdict
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer, fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, pairwise_distances
import scipy.stats as ss
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from typing import Literal, Dict, Tuple, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import time
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning  
import warnings
from tqdm import TqdmWarning
from Gini_MDS import GiniMDS
import torch
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=TqdmWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# Datasets
# Wholesale
def load_wholesale():
    df = pd.read_csv('datasets/Wholesale_customers_data.csv')
    X = df.drop('Channel', axis=1).values
    y = df['Channel']
    return X, y.ravel()

# Iris
def loadiris():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y.ravel()
    
# Wine
def loadwine():
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y.ravel()
    
# Breast
def loadcancer():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    return X, y.ravel()
    
# Sonar
def load_sonar():
    sonar = fetch_openml(name='sonar', version=1, as_frame=False)
    X = sonar.data
    y = sonar.target
    return X, y.ravel()

# Iniosphere
def load_iono():
    df = pd.read_csv('datasets/ionosphere_data.csv')
    df['column_a'] = df.column_a.astype('float64')
    X = df.values[:, :-1]
    y = df.values[:, -1]
    return X, y.ravel()

#Banknote
def load_bank():
    data = pd.read_csv('datasets/BankNote_Authentication.csv')
    X = data.drop('class', axis=1).values
    y = data['class'].values
    return X, y.ravel()

#Liver
def load_liver():
    data = pd.read_csv('datasets/indian_liver_patient.csv')
    dataset = data.dropna()
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    # Encode categorical variables
    encode_X = LabelEncoder()
    X[:,1] = encode_X.fit_transform(X[:,1])
    return X, y.ravel()
    
# German
def load_german():
    df = pd.read_csv('datasets/german_credit_data.csv')
    df = df.copy() 
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
    # Encode categorical variables
    label_encoders = {}
    for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    df = df.replace([np.inf, -np.inf], np.nan)
    X = df.drop(columns=['Credit amount']).values
    y = df['Credit amount'] > df['Credit amount'].median()
    return X, y.ravel()

# Australian
def load_australian():
    df=pd.read_table('datasets/australian.csv',sep=',')
    df.columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','Y']
    X=df.drop('Y',axis=1).values
    y=df['Y'].values
    return X, y.ravel()

# Vehicle
def load_vehicle():
    df = pd.read_csv('datasets/vehicle.csv')
    df = df.replace({'car': 0, 'bus': 1, 'van':2})
    df.apply(lambda x: sum(x.isnull()))
    vehicle_df = df.copy()
    vehicle_df.fillna(vehicle_df.median(), axis=0, inplace=True)
    vehicle_df.apply(lambda x: sum(x.isnull()))
    vehicle_df_new = vehicle_df.copy()
    X = vehicle_df_new.drop('class', axis =1).values
    y = vehicle_df_new['class']
    return X, y.ravel()

# Heart
def load_heart():
    data = pd.read_csv('datasets/heart.csv')
    X = data.drop(columns=['thal']).values
    y = data.thal
    return X, y.ravel()
    
# Glass
def load_glass():
    df = pd.read_csv('datasets/glass.csv')
    X = df.drop(columns='Type').values
    y = df['Type']
    return X, y.ravel()

# Balance
def load_balance():
    df = pd.read_csv('datasets/balance-scale.csv')
    df['Class'] = LabelEncoder().fit_transform(df['Class'].tolist())
    y = df[['Class']].values
    X = df.drop(['Class'], axis = 1).values
    return X, y.ravel()

# Haberman
def load_haberman():
    df = pd.read_csv("datasets/haberman.csv")
    col_names=['age','year','node','status']
    df.columns=col_names
    X=df.drop('status',axis=1).values
    y=df['status']
    return X, y.ravel()

# QSAR
def load_qsar():
    df = pd.read_csv("datasets/qsar.csv")
    X = df.iloc[:, 3:12].values
    y = df['Class'].values
    y = LabelEncoder().fit_transform(y)
    return X, y.ravel()

# Functions to train

DATASETS = {
    "wholesale": load_wholesale,
    "iris": loadiris,
    "wine": loadwine,
    "breast_cancer": loadcancer,
    "sonar": load_sonar,
    "ionosphere": load_iono,
    "banknote": load_bank,
    "indian_liver": load_liver,
    "german_credit": load_german,
    "australian": load_australian,
    "vehicle": load_vehicle,
    "heart": load_heart,
    "glass": load_glass,
    "balance_scale": load_balance,
    "haberman": load_haberman,
    "qsar": load_qsar,
}

DEVICE = "cuda:0"

def euclidean_distance_matrix(X):
    with torch.no_grad():
        T = torch.as_tensor(X, dtype=torch.float64)
        diff = T[:, None, :] - T[None, :, :]
        D = torch.sqrt(torch.clamp((diff ** 2).sum(dim=-1), min=0.0))
        return D.cpu().numpy().astype(np.float64)

def safe_trustworthiness(X, Z, n_neighbors=5):
    try:
        return float(trustworthiness(X, Z, n_neighbors=n_neighbors, metric="euclidean"))
    except Exception:
        return np.nan

def safe_silhouette(Z, y):
    try:
        y = np.asarray(y)
        if len(np.unique(y)) < 2 or len(y) <= len(np.unique(y)):
            return np.nan
        return float(silhouette_score(Z, y, metric="euclidean"))
    except Exception:
        return np.nan

def run_gini_mds(X, method, nu, n_components=2, device=DEVICE, dtype=torch.float64,
                 gini_mode="rank", max_iter=800, tol=1e-7, row_center=True):
    model = GiniMDS(n_components=int(n_components), nu=float(nu),
                    gini_mode=gini_mode, mds_method=method,
                    dtype=dtype, device=device, max_iter=max_iter, tol=tol, random_state=0)
    Z = model.fit_transform(X=X)
    D_fit = model.train_D_
    s1 = model.stress(D_fit, Z, kind="stress1")
    #sS = model.stress(D_fit, Z, kind="sammon")
    tw = safe_trustworthiness(X, Z)
    return model, Z, s1, tw

def run_euclid_mds(X, method, n_components=2, device=DEVICE, dtype=torch.float64,
                   iterative_on_cpu=True, max_iter=800, tol=1e-7):
    D = euclidean_distance_matrix(X)
    dev = device if method == "cmds" or not iterative_on_cpu else "cpu"
    dt  = dtype if method == "cmds" else torch.float32
    model = GiniMDS(n_components=int(n_components), mds_method=method,
                    dtype=dt, device=dev, max_iter=max_iter, tol=tol, random_state=0)
    Z = model.fit_transform(D=D)
    if method == "sammon":
        s1 = model.stress(D, Z, kind="sammon")
    elif method == "huber":
        s1 = model.stress(D, Z, kind="huber")
    else:
        s1 = model.stress(D, Z, kind="stress1")
    tw = safe_trustworthiness(X, Z)
    return model, Z, s1, tw

def neighborhood_hit(Z, y, n_neighbors=10):
    Z = np.asarray(Z); y = np.ravel(y)
    nn = NearestNeighbors(n_neighbors=n_neighbors+1).fit(Z)
    ind = nn.kneighbors(return_distance=False)[:, 1:]  
    return float((y[ind] == y[:, None]).mean())


# Experience stress with noise

# =========================
# Parameters
# =========================
nu_grid = np.arange(1.1, 5.0 + 1e-9, 0.1)
methods = ["cmds", "smacof", "sammon", "huber"]
gini_method = "cmds"
sigma = 10          # standard deviation for noise
num_noise = 0.1     # fraction of noisy points
n_runs = 100     # number of runs

# =========================
# Store results
# =========================
all_results = defaultdict(list)

t0 = time.time()
for run_idx in range(n_runs):
    print(f"Run {run_idx+1}/{n_runs}")
    
    for ds_name, loader in DATASETS.items():
        X, y = loader()
        X = np.asarray(X, dtype=np.float64)
        
        # Add noise
        num_samples = X.shape[0]
        num_noisy = int(num_noise * num_samples)
        noisy_indices = np.random.choice(num_samples, size=num_noisy, replace=False)
        noise_std = sigma * np.std(X)
        X_noisy = X.copy()
        X_noisy[noisy_indices] += np.random.normal(0, noise_std, size=(num_noisy, X.shape[1]))
        
        # Reference distances
        D_ref = pairwise_distances(X_noisy, metric='euclidean')
        
        for m in (1, 2, 3):
            # --- Euclidean methods ---
            for method in methods:
                _, Z_e, s1_e, tw_e = run_euclid_mds(X_noisy, method, n_components=m)
                sil_e = safe_silhouette(Z_e, y)
                nei_e = neighborhood_hit(Z_e, y, n_neighbors=10)

                all_results[(ds_name, m, method)].append({
                    "stress1": s1_e,
                    "trustworthiness": tw_e,
                    "silhouette": sil_e,
                    "Neighbors": nei_e,
                    "best_nu": None
                })
            
            # --- Gini grid search ---
            best = (None, np.inf, None, None, None, None)
            for nu in nu_grid:
                _, Z_g, s1_g, tw_g = run_gini_mds(X_noisy, gini_method, nu, n_components=m, row_center=False)
                sil_g = safe_silhouette(Z_g, y)
                nei_g = neighborhood_hit(Z_g, y, n_neighbors=10)
                
                if s1_g < best[1]:
                    best = (float(nu), s1_g, tw_g, sil_g, nei_g)
            
            all_results[(ds_name, m, f"gini/{gini_method}")].append({
                "stress1": best[1],
                "trustworthiness": best[2],
                "silhouette": best[3],
                "Neighbors": best[4],
                "best_nu": best[0]
            })

elapsed_total = time.time() - t0
print(f"\n=== Total experiment time: {elapsed_total:.2f} seconds ===")

# =========================
# Average and std over runs
# =========================
rows = []
for (ds_name, m, technique), metrics_list in all_results.items():
    avg_metrics = {k: np.mean([d[k] for d in metrics_list if d[k] is not None]) for k in metrics_list[0]}
    std_metrics = {f"{k}_std": np.std([d[k] for d in metrics_list if d[k] is not None]) for k in metrics_list[0]}
    row = {"dataset": ds_name, "components": m, "technique": technique, **avg_metrics, **std_metrics}
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("results_summary_sigma_10_noise_01.csv", index=False)
# =========================
# Optional: display stress with best nu
# =========================
def fmt_stress(stress, nu):
    if nu is None or (isinstance(nu, float) and np.isnan(nu)):
        return f"{stress:.6f}"
    return f"{stress:.6f} (ν={nu:.2f})"

df["disp_stress1"] = [fmt_stress(s, nu) for s, nu in zip(df["stress1"], df["best_nu"])]

# =========================
# Pivot tables for reporting
# =========================
table_stress1 = df.pivot_table(index=["dataset", "components"], columns="technique",
                               values="disp_stress1", aggfunc="first")
table_trust = df.pivot_table(index=["dataset", "components"], columns="technique",
                             values="trustworthiness", aggfunc="first").round(6)
table_sil = df.pivot_table(index=["dataset", "components"], columns="technique",
                           values="silhouette", aggfunc="first").round(6)
table_neihb = df.pivot_table(index=["dataset", "components"], columns="technique",
                             values="Neighbors", aggfunc="first").round(6)
table_bestnu = df.pivot_table(index=["dataset", "components"], columns="technique",
                              values="best_nu", aggfunc="first").round(2)

# =========================
# Compute ranks
# =========================
df["stress_rank"] = df.groupby(["dataset", "components"])["stress1"].rank(method="min", ascending=True)
df["silhouette_rank"] = df.groupby(["dataset", "components"])["silhouette"].rank(method="min", ascending=False)
df["trust_rank"] = df.groupby(["dataset", "components"])["trustworthiness"].rank(method="min", ascending=False)
df["neighbor_rank"] = df.groupby(["dataset", "components"])["Neighbors"].rank(method="min", ascending=False)

table_rank_stress = df.pivot_table(index=["dataset", "components"], columns="technique",
                                   values="stress_rank", aggfunc="first").astype("Int64")
table_rank_silhouette = df.pivot_table(index=["dataset", "components"], columns="technique",
                                       values="silhouette_rank", aggfunc="first").astype("Int64")
table_rank_trust = df.pivot_table(index=["dataset", "components"], columns="technique",
                                  values="trust_rank", aggfunc="first").astype("Int64")
table_rank_neigh = df.pivot_table(index=["dataset", "components"], columns="technique",
                                  values="neighbor_rank", aggfunc="first").astype("Int64")

# =========================
# Print results
# =========================
print("\n=== Rank (1 = best) — Stress1 ===")
print(table_rank_stress.to_string())

print("\n=== Rank (1 = best) — Silhouette ===")
print(table_rank_silhouette.to_string())

print("\n=== Rank (1 = best) — Trust ===")
print(table_rank_trust.to_string())

print("\n=== Rank (1 = best) — Neighbors ===")
print(table_rank_neigh.to_string())

print("\n=== Stress1 (best ν shown for Gini) ===")
print(table_stress1.to_string())

print("\n=== Trustworthiness ===")
print(table_trust.to_string())

print("\n=== Silhouette ===")
print(table_sil.to_string())

print("\n=== Neighborhood Hit ===")
print(table_neihb.to_string())


metrics = ["stress1", "trustworthiness", "silhouette", "Neighbors"]

rows_long = []

for idx, row in df.iterrows():
    ds = row["dataset"]
    comp = row["components"]
    tech = row["technique"]
    
    for metric in metrics:
        value = row[metric]
        std = row.get(f"{metric}_std", np.nan)
        # Ajouter le rank si disponible
        rank_col = f"{metric}_rank"
        rank = row.get(rank_col, np.nan)
        
        rows_long.append({
            "dataset": ds,
            "components": comp,
            "technique": tech,
            "metric": metric,
            "mean": value,
            "std": std,
            "rank": rank
        })

df_long = pd.DataFrame(rows_long)

# Sauvegarder tout dans un seul CSV
df_long.to_csv("noise_sigma_10_noise_01.csv", index=False)


# Graph 
"""
# Datasets (abscissa)
datasets = [
    "australian", "balance_scale", "banknote", "breast_cancer",
    "german_credit", "glass", "haberman", "heart",
    "indian_liver", "ionosphere", "iris", "qsar",
    "sonar", "vehicle", "wholesale", "wine"
]

# Neighborhood Hit – component 1
cmds = [
    0.566087,  # australian
    0.547520,  # balance_scale
    0.646210,  # banknote
    0.831810,  # breast_cancer
    0.493200,  # german_credit
    0.306542,  # glass
    0.621242,  # haberman
    0.437954,  # heart
    0.625907,  # indian_liver
    0.581766,  # ionosphere
    0.796000,  # iris
    0.498203,  # qsar
    0.508173,  # sonar
    0.553546,  # vehicle
    0.573864,  # wholesale
    0.610112   # wine
]

gini = [
    0.570290,
    0.566400,
    0.638411,
    0.855360,
    0.495600,
    0.376168,
    0.626144,
    0.442574,
    0.629188,
    0.615385,
    0.598667,
    0.496791,
    0.500962,
    0.551773,
    0.580682,
    0.598315
]

huber = [
    0.566232,
    0.547680,
    0.646210,
    0.831810,
    0.493200,
    0.306542,
    0.620915,
    0.437954,
    0.625907,
    0.581766,
    0.796000,
    0.498845,
    0.508173,
    0.553428,
    0.573864,
    0.610674
]

# Plot
plt.figure(figsize=(11, 5))

plt.plot(datasets, cmds, marker='o', label="CMDS")
plt.plot(datasets, gini, marker='s', label="Gini/CMDS")
plt.plot(datasets, huber, marker='^', label="Huber")


plt.xticks(rotation=45, ha="right")
plt.xlabel("Dataset")
plt.ylabel("Neighborhood Hit")
plt.title("Neighborhood Hit – Component 1")
plt.legend()

ax = plt.gca()
ax.spines["bottom"].set_visible(False)

plt.tight_layout()
plt.show()

datasets = [
    "australian", "balance_scale", "banknote", "breast_cancer",
    "german_credit", "glass", "haberman", "heart",
    "indian_liver", "ionosphere", "iris", "qsar",
    "sonar", "vehicle", "wholesale", "wine"
]

cmds = [
     0.121486, -0.065298,  0.113821,  0.262744,
     0.000351, -0.736109,  0.194934, -0.743384,
    -0.098391, -0.048659,  0.263378, -0.245277,
     0.027469,  0.005982, -0.012657,  0.165436
]

plt.plot(datasets, cmds, marker='o')
plt.axhline(0)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Silhouette score")
plt.title("CMDS – Component 1")
plt.tight_layout()
plt.show()"""
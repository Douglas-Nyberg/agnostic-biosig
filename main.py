"""Agnostic biosignature classification. Cleaves et al. (2023) PNAS."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr, skew, kurtosis
from scipy.signal import correlate
import matplotlib.pyplot as plt

from src.data_loader import load_data, get_species, get_age
from src.preprocessing import full_preprocessing_pipeline

def parse_feature(f):
    if ';' in str(f): parts = f.split(';'); return int(parts[0]), int(parts[1])
    return None, None

def peak_count(x, thresh=0.01): return np.sum(x > thresh)

def shannon_entropy(x, thresh=0.01):
    active = x[x > thresh]
    if len(active) < 2: return 0.0
    p = active / active.sum()
    return -(p * np.log(p + 1e-10)).sum()

def gini(x, thresh=0.01):
    active = np.sort(x[x > thresh])
    n = len(active)
    if n < 2: return 0.0
    idx = np.arange(1, n + 1)
    return ((2 * idx - n - 1) * active).sum() / (n * active.sum())

def mz_spread(x, features, thresh=0.01):
    mz = np.array([parse_feature(f)[1] for f in features])
    active = mz[x > thresh]
    return len(set(active[active != None])) if len(active) > 0 else 0

def rt_spread(x, features, thresh=0.01):
    scans = np.array([parse_feature(f)[0] for f in features])
    active = scans[x > thresh]
    active = active[active != None].astype(float)
    return np.std(active) if len(active) >= 2 else 0.0

def calc_kurtosis(x, thresh=0.01):
    active = x[x > thresh]
    return kurtosis(active, fisher=True) if len(active) >= 4 else 0.0

def calc_skewness(x, thresh=0.01):
    active = x[x > thresh]
    return skew(active) if len(active) >= 3 else 0.0

def rt_mz_corr(x, features, thresh=0.01):
    scans, mzs = [], []
    for j, f in enumerate(features):
        if x[j] > thresh and ';' in str(f):
            s, m = f.split(';')
            scans.append(float(s)); mzs.append(float(m))
    if len(scans) < 5: return 0.0
    r, _ = pearsonr(scans, mzs)
    return 0.0 if np.isnan(r) else r

def rt_bimodality(x, features, thresh=0.01):
    scans = [float(f.split(';')[0]) for j, f in enumerate(features) 
             if x[j] > thresh and ';' in str(f)]
    if len(scans) < 10: return 0.0
    scans = np.array(scans)
    s, k = skew(scans), kurtosis(scans, fisher=False)
    return (s**2 + 1) / k if k != 0 else 0.0

def mass_periodicity(x, features, thresh=0.01):
    spectrum = np.zeros(201)
    for j, f in enumerate(features):
        if x[j] > thresh and ';' in str(f):
            mz = int(float(f.split(';')[1]))
            if 0 <= mz <= 200: spectrum[mz] += x[j]
    if spectrum.sum() == 0: return 0.0
    ac = correlate(spectrum, spectrum, mode='full')[len(spectrum)-1:]
    ac = ac / ac[0] if ac[0] > 0 else ac
    return np.max(ac[2:50]) if len(ac) > 50 else 0.0

def cooccurrence_score(X, thresh=0.7):
    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    adj = (np.abs(corr) > thresh).astype(int)
    np.fill_diagonal(adj, 0)
    neighbors = adj.sum(axis=1)
    
    scores = []
    for i in range(len(X)):
        active = np.where(X[i] > 0.01)[0]
        if len(active) < 2: scores.append(0.0); continue
        friends_present = adj[active][:, active].sum(axis=1)
        total = neighbors[active]
        has_friends = total > 0
        if has_friends.sum() == 0: scores.append(0.0); continue
        scores.append(np.mean(friends_present[has_friends] / total[has_friends]))
    return np.array(scores)

def load_and_prepare():
    dfs, files = load_data()
    feature_df = full_preprocessing_pipeline(dfs, files, mz_range=(50, 200))
    
    species_dict, age_dict = get_species(), get_age()
    names = feature_df['sample_name'].tolist()
    
    valid, labels = [], []
    for n in names:
        n_str = Path(n).name if hasattr(n, 'name') or '/' in str(n) else str(n)
        species = species_dict.get(n_str)
        valid.append(species is not None)
        if species: labels.append(species)
    
    df_valid = feature_df[valid].reset_index(drop=True)
    clean_names = [Path(n).name if '/' in str(n) else str(n) for n, v in zip(names, valid) if v]
    
    exclude = ['sample_name', 'species', 'age_ma', 'Perc_Nnonzero']
    feat_cols = [c for c in df_valid.columns if c not in exclude]
    
    X, y = df_valid[feat_cols].values, np.array(labels)
    print(f"Loaded {len(y)} samples ({sum(y=='A')} abiotic, {sum(y=='B')} biotic), {X.shape[1]} features")
    return X, y, feat_cols, clean_names, age_dict

def calculate_metrics(X, y, features):
    cooc = cooccurrence_score(X)
    
    rows = []
    for i, x in enumerate(X):
        pc = peak_count(x)
        rows.append({
            'peak_count': pc,
            'entropy': shannon_entropy(x),
            'mz_coverage': mz_spread(x, features),
            'rt_spread': rt_spread(x, features),
            'cooccurrence_score': cooc[i],
            'rt_mz_corr': rt_mz_corr(x, features),
            'kurtosis': calc_kurtosis(x),
            'skewness': calc_skewness(x),
            'gini': gini(x),
            'rt_bimodality': rt_bimodality(x, features),
            'mass_periodicity': mass_periodicity(x, features),
            'functional_complexity': pc * cooc[i],
            'label': y[i]
        })
    
    df = pd.DataFrame(rows)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def cv_acc(X, y, cv):
    X_scaled = StandardScaler().fit_transform(X)
    preds = cross_val_predict(LogisticRegression(max_iter=1000), X_scaled, y, cv=cv)
    return accuracy_score(y, preds)

def train_models(metrics_df):
    y = (metrics_df['label'] == 'B').astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    
    models = {
        'baseline': ['peak_count', 'entropy', 'mz_coverage', 'cooccurrence_score', 'gini'],
        'with_shape': ['peak_count', 'entropy', 'mz_coverage', 'cooccurrence_score', 'gini', 'kurtosis', 'skewness'],
        'with_structure': ['peak_count', 'entropy', 'mz_coverage', 'cooccurrence_score', 'gini', 'kurtosis', 'skewness', 'rt_mz_corr'],
    }
    
    print("\n=== Model Progression ===")
    results = {}
    for name, cols in models.items():
        acc = cv_acc(metrics_df[cols].values, y, cv)
        results[name] = acc
        print(f"  {name}: {acc:.1%}")
    
    return results

def anomaly_detection(metrics_df):
    cols = ['functional_complexity', 'gini', 'rt_mz_corr', 'rt_bimodality', 'mass_periodicity', 'kurtosis']
    X = metrics_df[cols].values
    y_true = (metrics_df['label'] == 'B').astype(int).values
    
    X_abiotic = X[metrics_df['label'] == 'A']
    scaler = StandardScaler().fit(X_abiotic)
    X_ab_scaled, X_scaled = scaler.transform(X_abiotic), scaler.transform(X)
    
    methods = {
        'elliptic': EllipticEnvelope(contamination=0.05, random_state=42),
        'iforest': IsolationForest(contamination=0.4, random_state=42),
        'lof': LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.4),
    }
    
    print("\n=== Anomaly Detection ===")
    results = {}
    for name, model in methods.items():
        try:
            model.fit(X_ab_scaled)
            preds = (model.predict(X_scaled) == -1).astype(int)
            acc = accuracy_score(y_true, preds)
            results[name] = acc
            print(f"  {name}: {acc:.1%}")
        except Exception as e:
            print(f"  {name}: failed ({e})")
    return results

def rf_baseline(X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    preds = cross_val_predict(rf, X, y, cv=cv)
    return accuracy_score(y, preds)

def scatter(df, x, y, path):
    plt.figure(figsize=(8, 6))
    for label, color in [('A', '#E74C3C'), ('B', '#3498DB')]:
        sub = df[df['label'] == label]
        plt.scatter(sub[x], sub[y], c=color, alpha=0.7, s=60, label=label)
    plt.xlabel(x), plt.ylabel(y), plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def create_plots(df, output_dir):
    Path(output_dir).mkdir(exist_ok=True)
    plots = [
        ('functional_complexity', 'kurtosis'),
        ('peak_count', 'rt_mz_corr'),
        ('peak_count', 'cooccurrence_score'),
        ('rt_bimodality', 'mass_periodicity'),
        ('entropy', 'kurtosis'),
    ]
    for x, y in plots:
        scatter(df, x, y, f"{output_dir}/{x}_vs_{y}.png")
    print(f"Saved {len(plots)} plots to {output_dir}/")

def main():
    output_dir = 'results'
    Path(output_dir).mkdir(exist_ok=True)
    
    X, y, features, names, age_dict = load_and_prepare()
    metrics_df = calculate_metrics(X, y, features)
    metrics_df['sample_name'] = names
    
    model_results = train_models(metrics_df)
    anomaly_results = anomaly_detection(metrics_df)
    rf_acc = rf_baseline(X, y)
    
    create_plots(metrics_df, output_dir)
    metrics_df.to_csv(f'{output_dir}/sample_metrics.csv', index=False)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Random Forest:       {rf_acc:.1%}")
    print(f"  Best Logistic Reg:   {max(model_results.values()):.1%}")
    if anomaly_results:
        print(f"  Best Anomaly:        {max(anomaly_results.values()):.1%}")
    
    return metrics_df, model_results, rf_acc

if __name__ == "__main__":
    main()
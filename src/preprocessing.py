"""Mass spectrometry preprocessing: baseline removal, peak detection, feature extraction."""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from joblib import Parallel, delayed

def sqrt_transform(x): return np.sqrt(np.maximum(x, 0))

def smooth(signal, half_win=5):
    return uniform_filter1d(signal.astype(float), size=2*half_win+1, mode='nearest')

def snip_baseline(signal, iters=40):
    n = len(signal)
    w = np.log(np.log(np.sqrt(signal + 1) + 1) + 1)
    
    for p in range(1, iters + 1):
        avg = (np.roll(w, p) + np.roll(w, -p)) / 2
        mask = np.zeros(n, dtype=bool)
        mask[p:n-p] = True
        w = np.where(mask & (avg < w), avg, w)
    
    baseline = (np.exp(np.exp(w) - 1) - 1) ** 2 - 1
    return np.maximum(signal - np.maximum(baseline, 0), 0)

def estimate_noise(signal):
    mad = np.median(np.abs(signal - np.median(signal)))
    return mad * 1.4826

def detect_peaks(signal, half_win=20, snr=4):
    noise = estimate_noise(signal)
    if noise == 0: noise = 1e-10
    peaks, _ = find_peaks(signal, height=snr * noise, distance=half_win)
    return (peaks, signal[peaks]) if len(peaks) > 0 else (np.array([]), np.array([]))

def preprocess_sample(df, mz_range=(50, 200), num_scans=3240):
    mz_start, mz_end = mz_range
    num_mz = mz_end - mz_start + 1
    
    df = df.iloc[:num_scans].copy()
    mz_cols = [str(mz) for mz in range(mz_start, mz_end + 1)]
    matrix = sqrt_transform(df[mz_cols].values.astype(float))
    
    for j in range(num_mz):
        matrix[:, j] = smooth(matrix[:, j])
        matrix[:, j] = snip_baseline(matrix[:, j])
    
    mi, ma = matrix.min(), matrix.max()
    matrix = (matrix - mi) / (ma - mi) if ma - mi > 0 else np.zeros_like(matrix)
    
    peaks = np.zeros_like(matrix)
    for j in range(num_mz):
        idx, vals = detect_peaks(matrix[:, j])
        if len(idx) > 0: peaks[idx, j] = vals
    
    return peaks

def process_sample_wrapper(args):
    i, df, mz_range, num_scans = args
    try:
        return i, preprocess_sample(df, mz_range, num_scans), True
    except Exception as e:
        print(f"Warning: sample {i} failed: {e}")
        return i, None, False

def cluster_scans(peak_matrices, mz_range=(50, 200), cluster_dist=20):
    mz_start, mz_end = mz_range
    cluster_info = {}
    
    for mz_idx in range(mz_end - mz_start + 1):
        mz = mz_start + mz_idx
        scans = np.concatenate([np.where(pm[:, mz_idx] > 0)[0] for pm in peak_matrices])
        
        if len(scans) == 0:
            cluster_info[mz] = []
            continue
        
        if len(np.unique(scans)) > 1:
            clusters = fcluster(linkage(scans.reshape(-1, 1), method='complete'), 
                               t=cluster_dist, criterion='distance')
            ranges = []
            for cid in np.unique(clusters):
                cs = scans[clusters == cid]
                ranges.append({'min': cs.min(), 'max': cs.max(), 'mean': int(round(cs.mean()))})
            cluster_info[mz] = ranges
        else:
            cluster_info[mz] = [{'min': scans[0], 'max': scans[0], 'mean': scans[0]}]
    
    return cluster_info

def extract_features(peak_matrices, cluster_info, mz_range=(50, 200)):
    mz_start, mz_end = mz_range
    
    feature_names = []
    for mz in range(mz_start, mz_end + 1):
        for sr in cluster_info.get(mz, []):
            feature_names.append(f"{sr['mean']};{mz}")
    
    features = []
    for pm in peak_matrices:
        row = []
        for mz in range(mz_start, mz_end + 1):
            mz_idx = mz - mz_start
            for sr in cluster_info.get(mz, []):
                vals = pm[sr['min']:sr['max']+1, mz_idx]
                row.append(vals.max() if len(vals) > 0 else 0)
        features.append(row)
    
    df = pd.DataFrame(features, columns=feature_names)
    df['Perc_Nnonzero'] = (df > 0).sum(axis=1) / len(feature_names) if feature_names else 0
    return df

def remove_low_variance(df, freq_cut=19, unique_cut=10):
    drop = []
    for col in df.columns:
        vals = df[col].values
        unique_pct = len(np.unique(vals)) / len(vals) * 100
        if unique_pct < unique_cut:
            vc = pd.Series(vals).value_counts()
            if len(vc) >= 2 and vc.iloc[0] / vc.iloc[1] > freq_cut:
                drop.append(col)
    return df.drop(columns=drop)

def remove_correlated(df, thresh=0.85):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > thresh)]
    return df.drop(columns=drop)

def full_preprocessing_pipeline(dfs_list, file_names, mz_range=(50, 200), num_scans=3240, n_jobs=-1):
    args = [(i, dfs_list[i], mz_range, num_scans) for i in range(len(dfs_list))]
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(process_sample_wrapper)(a) for a in args)
    
    peak_matrices, valid_idx = [], []
    for i, pm, ok in sorted(results):
        if ok:
            peak_matrices.append(pm)
            valid_idx.append(i)
    
    print(f"Processed {len(peak_matrices)} samples")
    
    cluster_info = cluster_scans(peak_matrices, mz_range)
    df = extract_features(peak_matrices, cluster_info, mz_range)
    df = remove_low_variance(df)
    df = remove_correlated(df)
    df['sample_name'] = [file_names[i].name if hasattr(file_names[i], 'name') else str(file_names[i]) 
                         for i in valid_idx]
    
    print(f"Final features: {df.shape[1] - 2}")
    return df
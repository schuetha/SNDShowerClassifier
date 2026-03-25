import numpy as np
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
from collections import defaultdict
from math import floor
import os

def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

# ---------- feature builder ----------
def build_x(event_hits, det_types_known=None, ori_types_known=None, stats=None):
    """
    Build node features:
      [ xy, z, log1p(E), one-hot(orientation), one-hot(det_type) ]
    """
    ev = to_numpy(event_hits)

    xy  = ev[:, 0].astype(np.float32)
    z   = ev[:, 1].astype(np.float32)

    e_raw = ev[:, 2].astype(np.float32)
    # Clip negatives / NaNs before log1p
    e_raw = np.nan_to_num(e_raw, nan=0.0, posinf=None, neginf=0.0)
    e_raw = np.clip(e_raw, 0.0, None)
    e_log = np.log1p(e_raw)

    det_raw = ev[:, 3]  # detector type (categorical)
    ori_raw = ev[:, 4]  # orientation (categorical)

    # ---- normalization for continuous features (train-only stats) ----
    if stats is not None:
        xy    = (xy    - stats['XY_mean'])    / (stats['XY_std']    + 1e-8)
        z     = (z     - stats['Z_mean'])     / (stats['Z_std']     + 1e-8)
        e_log = (e_log - stats['E_log_mean']) / (stats['E_log_std'] + 1e-8)

    feats = [xy[:, None], z[:, None], e_log[:, None]]

    # ---- one-hot: ORIENTATION ----
    # If your orientation values are floats but represent categories, make them integers.
    # If they are truly continuous, you must first bin or map them.
    if ori_types_known is None:
        # infer from this event only (not recommended for training!)
        ori_types_known = np.unique(ori_raw.astype(np.int64))
    ori_to_idx = {o: i for i, o in enumerate(ori_types_known)}
    ori_idx = np.array([ori_to_idx.get(int(o), len(ori_types_known)-1) for o in ori_raw], dtype=np.int64)
    ori_oh = np.eye(len(ori_types_known), dtype=np.float32)[ori_idx]
    feats.append(ori_oh)

    # ---- one-hot: DETECTOR TYPE ----
    if det_types_known is None:
        det_types_known = np.unique(det_raw.astype(np.int64))
    det_to_idx = {d: i for i, d in enumerate(det_types_known)}
    det_idx = np.array([det_to_idx.get(int(d), len(det_types_known)-1) for d in det_raw], dtype=np.int64)
    det_oh = np.eye(len(det_types_known), dtype=np.float32)[det_idx]
    feats.append(det_oh)

    x = np.concatenate(feats, axis=1).astype(np.float32)
    return torch.from_numpy(x)

def make_event(event_hits, label, det_types_known=None, ori_types_known=None, stats=None):
    ev = to_numpy(event_hits)

    zq = np.rint(ev[:, 1] / 10.0) * 10.0
    z_index = torch.tensor(zq, dtype=torch.float32)          # <-- copy (NOT from_numpy)
    times = torch.tensor(ev[:, 5], dtype=torch.float32)      # <-- copy (NOT from_numpy)
    z_t = torch.stack([z_index, times], dim=1)               # (N,2) tensor for monotonic KNN

    x  = build_x(ev, det_types_known=det_types_known, ori_types_known=ori_types_known, stats=stats)
    y  = torch.tensor(abs(int(label)), dtype=torch.long)

    pos  = torch.tensor(ev[:, :3], dtype=torch.float32)      # already a copy
    flag = torch.tensor(ev[:, 4], dtype=torch.long)          # <-- copy (NOT from_numpy)

    return Data(x=x, y=y, pos=pos, flag=flag, z_time=z_t)

# ---------- utilities ----------
def compute_global_det_types(all_events):
    vals = []
    for ev in tqdm(all_events, desc="Scanning det types for vocabulary"):
        ev_np = to_numpy(ev)
        vals.append(np.unique(ev_np[:, 3].astype(np.int64)))
    det_types = np.unique(np.concatenate(vals)) if vals else np.array([], dtype=np.int64)
    return det_types.tolist()

def compute_global_ori_values(all_events):
    """
    Build a stable orientation vocabulary across the whole dataset.
    If your orientations are floats but categorical, we cast to int.
    If truly continuous, define a binning strategy before one-hotting.
    """
    vals = []
    for ev in tqdm(all_events, desc="Scanning orientation values for vocabulary"):
        ev_np = to_numpy(ev)
        # If orientation are floats that map to categories but not integers,
        # consider rounding: e.g., np.rint(ev_np[:,4]).astype(np.int64)
        vals.append(np.unique(ev_np[:, 4].astype(np.int64)))
    ori_types = np.unique(np.concatenate(vals)) if vals else np.array([], dtype=np.int64)
    return ori_types.tolist()

def fit_stats_on_train(all_events, train_idx):
    """
    Compute mean/std for XY, Z, log1p(E) using Welford's algorithm
    over only the TRAIN split. Robust to variable-length events and NaNs.
    """
    def welford_update(state, x):
        count, mean, M2 = state
        for xi in x:
            count += 1
            delta = xi - mean
            mean += delta / count
            M2 += delta * (xi - mean)
        return count, mean, M2

    def finalize(state):
        count, mean, M2 = state
        if count < 2:
            return float(mean), 1.0
        var = M2 / (count - 1)
        return float(mean), float(np.sqrt(max(var, 1e-12)))

    XY_state   = (0, 0.0, 0.0)
    Z_state    = (0, 0.0, 0.0)
    ELOG_state = (0, 0.0, 0.0)

    for i in tqdm(train_idx, desc="Fitting stats on train (streaming)"):
        ev = to_numpy(all_events[i])

        xy = ev[:, 0].astype(np.float32)
        z  = ev[:, 1].astype(np.float32)

        e_raw = ev[:, 2].astype(np.float32)
        e_raw = np.nan_to_num(e_raw, nan=0.0, posinf=None, neginf=0.0)
        e_raw = np.clip(e_raw, 0.0, None)
        e_log = np.log1p(e_raw)

        xy   = xy[np.isfinite(xy)]
        z    = z[np.isfinite(z)]
        e_log = e_log[np.isfinite(e_log)]

        XY_state   = welford_update(XY_state,   xy)
        Z_state    = welford_update(Z_state,    z)
        ELOG_state = welford_update(ELOG_state, e_log)

    XY_mean, XY_std       = finalize(XY_state)
    Z_mean, Z_std         = finalize(Z_state)
    E_log_mean, E_log_std = finalize(ELOG_state)

    stats = {
        'XY_mean': XY_mean,   'XY_std': XY_std,
        'Z_mean':  Z_mean,    'Z_std':  Z_std,
        'E_log_mean': E_log_mean, 'E_log_std': E_log_std,
    }
    return stats

def build_split_graphs(indices, all_events, labels, det_types_known, ori_types_known, stats):
    gs = []
    for i in tqdm(indices, desc="Building graphs"):
        gs.append(
            make_event(
                all_events[i], int(labels[i]),
                det_types_known=det_types_known,
                ori_types_known=ori_types_known,
                stats=stats
            )
        )
    return gs

def stream_save(indices, all_events, labels, det_types_known, ori_types_known, stats, out_path, chunk_size=50000):
    print(len(all_events)/chunk_size, "Number of all files")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start, part = 0, 0
    while start < len(indices):
        part += 1
        chunk_idx = indices[start:start+chunk_size]
        start += chunk_size
        graphs = build_split_graphs(chunk_idx, all_events, labels, det_types_known, ori_types_known, stats)
        torch.save(graphs, f"{out_path}.part{part:03d}.pt")

# python3 FCGNN/GNN_flavour_oh_ori.py

# ---------- main ----------
if __name__ == "__main__":
    # data = torch.load("/eos/user/s/schuetha/GNN_data.pt", map_location="cpu")
    data = torch.load("/eos/user/s/schuetha/signal_flavour/GNN_data_train_small_correct_all_flavour.pt", map_location="cpu")

    # Expecting:
    #   data["features"] : list/array of (n_hits_i, 5)
    #   data["flavours"] : 1D labels
    if "features" in data:
        all_events = data["features"]
    else:
        raise KeyError("Could not find 'events' in GNN_data.pt. Please store per-event arrays under data['features'].")

    print("=== Dataset info ===")
    print("path:", "/eos/user/s/schuetha/signal_flavour/GNN_data_train_small_correct_all_flavour.pt")
    print("type(all_events):", type(all_events))
    print("len(all_events):", len(all_events))
    print("type(first event):", type(all_events[0]))
    print("shape/len(first event):", getattr(all_events[0], "shape", None), len(all_events[0]))

    total_hits = sum(len(ev) for ev in all_events)
    print("Total events:", len(all_events))
    print("Total hits:", total_hits)
    print("Avg hits/event:", total_hits / len(all_events))

    labels = np.asarray(data["flavours"], dtype=np.int64)
    num_events = len(labels)
    assert num_events == len(all_events), "events and labels length mismatch"

    # # //////////////////////////////////////////////////////////////////////////
    # drop = {16, -16}
    # keep_mask = ~np.isin(labels, list(drop))

    # all_events = [all_events[i] for i in np.nonzero(keep_mask)[0]]
    # labels = labels[keep_mask]

    # print(f"Kept {len(labels)} / {num_events} events (dropped {(~keep_mask).sum()})")
    # # //////////////////////////////////////////////////////////////////////////
    
    # stratified split ratios
    train_ratio, val_ratio, test_ratio = 0.50, 0.20, 0.30
    rng = np.random.default_rng(42)

    # gather indices by class
    cls_to_idx = defaultdict(list)
    for i, y in enumerate(labels):
        cls_to_idx[int(y)].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for cls, idxs in tqdm(cls_to_idx.items(), desc="Stratified split per class"):
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = floor(train_ratio * n)
        n_val   = floor(val_ratio   * n)

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train+n_val])
        test_idx.extend(idxs[n_train+n_val:])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    # global vocabularies (stable one-hot across all splits)
    det_types_known = compute_global_det_types(all_events)
    if len(det_types_known) == 0:
        det_types_known = [0]  # fallback

    ori_types_known = compute_global_ori_values(all_events)
    if len(ori_types_known) == 0:
        ori_types_known = [0]  # fallback

    # fit normalization on TRAIN only, then apply to all
    stats = fit_stats_on_train(all_events, train_idx)

    # ---- streaming save (memory-friendly) ----
    base_out = "/eos/user/s/schuetha/signal_background_data_loader_all_flavour_oh_ori_flavour"
    stream_save(train_idx, all_events, labels, det_types_known, ori_types_known, stats,
                out_path=f"{base_out}/GNN_dataset_s_b_train", chunk_size=5000)
    stream_save(val_idx,   all_events, labels, det_types_known, ori_types_known, stats,
                out_path=f"{base_out}/GNN_dataset_s_b_val",   chunk_size=5000)
    stream_save(test_idx,  all_events, labels, det_types_known, ori_types_known, stats,
                out_path=f"{base_out}/GNN_dataset_s_b_test",  chunk_size=5000)

    # Save ancillary info for consistent reload
    torch.save({
        "det_types_known": det_types_known,
        "ori_types_known": ori_types_known,
        "stats": stats,
        "split_counts": {
            "train": len(train_idx),
            "val":   len(val_idx),
            "test":  len(test_idx),
        }
    }, f"{base_out}/GNN_dataset_meta.pt")

    print("Done.")
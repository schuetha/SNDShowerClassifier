#!/usr/bin/env python3
print("Starting training script...")
import glob
import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import yaml
from FCGNN.my_model import FCGNN
# GraphGym / YACS cfg (used by ExpanderEdgeFixer via torch_geometric.graphgym.config.cfg)
from torch_geometric.graphgym.config import cfg, set_cfg
from yacs.config import CfgNode as CN
print("Finished imports.")

# -------------------- DDP helpers --------------------
def is_distributed_run():
    return int(os.getenv("WORLD_SIZE", "1")) > 1


def setup_ddp_if_needed():
    if is_distributed_run():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)


def get_rank():
    return dist.get_rank() if is_distributed_run() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed_run() else 1


def cleanup_ddp_if_needed():
    if is_distributed_run():
        dist.barrier()
        dist.destroy_process_group()


# -------------------- IO helpers --------------------
def load_split(pattern, name=""):
    """
    Load a list of PyG Data objects saved with torch.save(list_of_graphs).
    PyTorch 2.6+ defaults to weights_only=True which breaks non-weight pickles.
    For trusted dataset files, set weights_only=False.
    """
    graphs = []
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    for p in tqdm(paths, desc=f"Loading dataset: {name}", leave=False):
        graphs_part = torch.load(p, map_location="cpu", weights_only=False)
        # graphs_part is expected to be a list[Data]
        graphs.extend(graphs_part)
    return graphs


def drop_empty(graphs, name=""):
    kept = []
    for g in tqdm(graphs, desc=f"Dropping empty graphs: {name}", leave=False):
        # robust num_nodes check
        if hasattr(g, "num_nodes") and g.num_nodes is not None:
            nnodes = int(g.num_nodes)
        else:
            x = getattr(g, "x", None)
            nnodes = int(x.size(0)) if x is not None else 0

        if nnodes > 0:
            kept.append(g)
    return kept


def sanity_check_graphs(graphs, name=""):
    """
    Optional but recommended: your preprocess uses batch.pos and expects edge_attr for edge_fixer.
    If missing, you'll crash later with a less-informative error.
    """
    if len(graphs) == 0:
        raise RuntimeError(f"{name}: no graphs after loading.")

    g0 = graphs[0]
    missing = []
    if not hasattr(g0, "x"):
        missing.append("x")
    if not hasattr(g0, "pos"):
        missing.append("pos (needed because preprocess_batch uses batch.pos)")
    if not hasattr(g0, "edge_attr"):
        missing.append("edge_attr (needed because ExpanderEdgeFixer(add_edge_index=True) concatenates it)")
    if not hasattr(g0, "edge_index"):
        missing.append("edge_index")

    if missing:
        raise RuntimeError(
            f"{name}: first graph is missing fields: {missing}\n"
            f"Either:\n"
            f"  - add these when building/saving the dataset, or\n"
            f"  - adjust preprocess_batch / edge_fixer usage accordingly."
        )


# -------------------- train/eval --------------------
def run_epoch(model, loader, device, crit, opt=None):
    train = opt is not None
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, leave=False, desc="Training" if train else "Evaluating"):
        batch = batch.to(device, non_blocking=True)

        logits = model(batch)
        loss = crit(logits, batch.y)

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        total_loss += loss.item() * batch.y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# -------------------- GraphGym cfg setup --------------------
def init_graphgym_cfg_from_yaml_dict(config_dict):
    """
    ExpanderEdgeFixer imports `cfg` from torch_geometric.graphgym.config at import time and reads:
      - cfg.gt.dim_hidden / cfg.gt.dim_edge
      - cfg.prep.use_exp_edges / cfg.prep.exp
    Your YAML is NOT a GraphGym YAML, so we map only the needed keys.

    This must run BEFORE you instantiate ExpanderEdgeFixer (i.e., before FCGNN() constructor).
    """
    # Reset cfg to GraphGym defaults
    set_cfg(cfg)
    cfg.set_new_allowed(True)

    # Ensure subnodes exist
    if not hasattr(cfg, "gt") or cfg.gt is None:
        cfg.gt = CN()
    if not hasattr(cfg, "prep") or cfg.prep is None:
        cfg.prep = CN()

    # Provide safe defaults (avoid AttributeError)
    # gt
    if not hasattr(cfg.gt, "dim_hidden"):
        cfg.gt.dim_hidden = 64
    if not hasattr(cfg.gt, "dim_edge"):
        cfg.gt.dim_edge = None  # ExpanderEdgeFixer will set dim_edge=dim_hidden if None

    # prep
    if not hasattr(cfg.prep, "exp"):
        cfg.prep.exp = False
    if not hasattr(cfg.prep, "use_exp_edges"):
        # your exp_edge_fixer.py uses cfg.prep.use_exp_edges and cfg.prep.exp
        # If you want to use expander edges, set use_exp_edges=True and also store expander_edges in Data.
        cfg.prep.use_exp_edges = False

    # Now overwrite from your YAML dict if provided
    if isinstance(config_dict, dict):
        if "gt" in config_dict and isinstance(config_dict["gt"], dict):
            for k, v in config_dict["gt"].items():
                setattr(cfg.gt, k, v)
        if "prep" in config_dict and isinstance(config_dict["prep"], dict):
            for k, v in config_dict["prep"].items():
                setattr(cfg.prep, k, v)

    cfg.set_new_allowed(False)
    cfg.freeze()


# -------------------- main --------------------
if __name__ == "__main__":
    print("First check points:")
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument(
                        "-m",
                        "--models",
                        type=str,
                        default="/afs/cern.ch/user/s/schuetha/work/public/GNN/FCGNN/Full_model_GraphGPS.yaml",
                        help="Model path (YAML)",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-o", "--output", type=str, default="models_GNN_Graph_GPS", help="Output directory")
    args = parser.parse_args()
    print("Second check points:")
    setup_ddp_if_needed()
    rank, world_size = get_rank(), get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")
        print("Epochs:", args.epoch)

    # -------- Load YAML --------
    with open(args.models) as f:
        config = yaml.safe_load(f)

    # -------- Initialize GraphGym cfg (needed by ExpanderEdgeFixer) --------
    init_graphgym_cfg_from_yaml_dict(config)

    # -------- Build model --------
    model = FCGNN(
                    config,
                    graph_level=True,
                ).to(device)

    if rank == 0:
        print(model)
        print("Successfully loading the model.")

    # -------- Load data --------
    # data_path = "/eos/user/s/schuetha/signal_background_data_loader_all_flavour_oh_ori_flag"

    # data_path = "/eos/user/s/schuetha/signal_background_data_loader_all_flavour_oh_ori_flag_with_time_newest"

    data_path = "/eos/user/s/schuetha/signal_background_data_loader_all_flavour_oh_ori_300_cut"

    if rank == 0:
        print("Start loading the data: Whole dataset")

    train_graphs = drop_empty(load_split(f"{data_path}/GNN_dataset_s_b_train.part*.pt", name="Train"), name="Train")
    val_graphs = drop_empty(load_split(f"{data_path}/GNN_dataset_s_b_val.part*.pt", name="Val"), name="Val")

    # Optional sanity checks (highly recommended given your preprocess/edge_fixer expectations)
    sanity_check_graphs(train_graphs, name="Train")
    sanity_check_graphs(val_graphs, name="Val")

    if rank == 0:
        print("Finished loading the training/validation data")
        print(f"Train graphs: {len(train_graphs)} | Val graphs: {len(val_graphs)}")

    # -------- Label remap + class weights --------
    def get_scalar_y(g):
        if torch.is_tensor(g.y):
            return int(g.y.item())
        return int(g.y)

    all_for_vocab = train_graphs + val_graphs
    classes = sorted({get_scalar_y(g) for g in all_for_vocab})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes_data = len(classes)
    print(f"Original classes: {classes}")

    def remap_inplace(graphs):
        for g in graphs:
            y_old = get_scalar_y(g)
            y_new = class_to_idx.get(y_old, num_classes_data - 1)
            g.y = torch.tensor(y_new, dtype=torch.long)

    remap_inplace(train_graphs)
    remap_inplace(val_graphs)

    ys_remapped = np.array([get_scalar_y(g) for g in train_graphs], dtype=np.int64)
    if ys_remapped.size == 0:
        raise RuntimeError("Train set is empty after loading/remap. Check your dataset glob paths.")

    counts = np.bincount(ys_remapped, minlength=num_classes_data)
    freq = counts / counts.sum()

    # Try to infer output classes from last linear if present; else use data classes
    # num_classes_model = None
    # for name, mod in model.named_modules():
    #     if isinstance(mod, torch.nn.Linear):
    #         num_classes_model = mod.out_features
    # if num_classes_model is None:
    #     num_classes_model = num_classes_data
    # Replace the module scanning with:
    num_classes_model = config["model"]["layers"][-1]["params"].get("num_classes", num_classes_data)
    print(f"Model output classes: {num_classes_model} | Data classes: {num_classes_data}")
    obs_weights = 1.0 / np.maximum(freq, 1e-12)
    weights = np.ones(num_classes_model, dtype=np.float32)
    weights[:num_classes_data] = obs_weights
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    if rank == 0:
        print(f"Classes (original): {classes}")
        print(f"Remapped to       : 0..{num_classes_data-1}")
        print("Class counts      :", counts.tolist())
        print("Class weights     :", weights.tolist())

    # -------- Samplers & Loaders --------
    if is_distributed_run():
        from torch.utils.data import DistributedSampler

        train_sampler = DistributedSampler(
            train_graphs, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_graphs, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
    else:
        train_sampler, val_sampler = None, None
    
    def check_feature_dims(graphs, name, max_print=10):
        keys = ["x", "edge_attr", "pos", "z_index"]
        dims = {k: {} for k in keys}

        for i, g in enumerate(graphs[:5000]):  # scan some
            for k in keys:
                if hasattr(g, k):
                    t = getattr(g, k)
                    if torch.is_tensor(t) and t.dim() >= 2:
                        dims[k].setdefault(int(t.size(1)), 0)
                        dims[k][int(t.size(1))] += 1

        print(f"\n[{name}] feature-dim histogram:")
        for k in keys:
            if dims[k]:
                print(f"  {k}: {sorted(dims[k].items())}")

    check_feature_dims(train_graphs, "train")
    check_feature_dims(val_graphs, "val")
    
    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
    )

    if is_distributed_run():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss(weight=class_weight)

    # -------- Train loop --------
    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = [], [], [], []

    if rank == 0:
        for split_name, graphs in [("train", train_graphs), ("val", val_graphs)]:
            zeros = sum(1 for g in graphs if int(getattr(g, "num_nodes", g.x.size(0))) == 0)
            print(f"{split_name}: {zeros} empty graphs")

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in (tqdm(range(args.epoch), desc="Epochs") if rank == 0 else range(args.epoch)):
        if is_distributed_run():
            train_loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        tr_loss, tr_acc = run_epoch(model, train_loader, device, crit, opt=opt)
        va_loss, va_acc = run_epoch(model, val_loader, device, crit, opt=None)

        if rank == 0:
            print(
                f"Epoch {epoch+1:02d} | train: Loss {tr_loss:.4f} | Acc {tr_acc:.3f} "
                f"|| val: Loss {va_loss:.4f} | Acc {va_acc:.3f}"
            )
            train_loss_hist.append(tr_loss)
            train_acc_hist.append(tr_acc)
            val_loss_hist.append(va_loss)
            val_acc_hist.append(va_acc)

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch

                outdir = f"{args.output}/{args.epoch}_epoch"
                os.makedirs(outdir, exist_ok=True)
                to_save = model.module if hasattr(model, "module") else model
                torch.save(to_save.state_dict(), f"{outdir}/model_{args.epoch}_epoch_best.pth")

    # -------- Save latest & plots (rank 0 only) --------
    if rank == 0:
        outdir = f"{args.output}/{args.epoch}_epoch"
        os.makedirs(outdir, exist_ok=True)

        to_save = model.module if hasattr(model, "module") else model
        torch.save(to_save.state_dict(), f"{outdir}/model_{args.epoch}_epoch_latest.pth")
        print(f"Saved latest model to: {outdir}/model_{args.epoch}_epoch_latest.pth")
        fig, ax = plt.subplots()
        ax.plot(train_loss_hist, label="train")
        ax.plot(val_loss_hist, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.savefig(f"{outdir}/loss_{args.epoch}_epoch.pdf")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(train_acc_hist, label="train")
        ax.plot(val_acc_hist, label="val")
        if best_epoch >= 0:
            ax.scatter(best_epoch, best_val_acc, s=25, zorder=5, label=f"Best (epoch {best_epoch+1}, {best_val_acc:.3f})")
            ax.text(best_epoch, best_val_acc + 0.01, f"{best_val_acc:.3f}", ha="center", va="bottom")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        plt.savefig(f"{outdir}/Accuracy_{args.epoch}_epoch.pdf")
        plt.close(fig)

    cleanup_ddp_if_needed()

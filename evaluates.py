import glob, os, argparse
import numpy as np
import torch, torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from FCGNN.my_model import FCGNN
from torch_geometric.graphgym.config import cfg, set_cfg
from yacs.config import CfgNode as CN
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from eval_stat_block import evaluation_multiclass
import yaml
import argparse 

# -------------------- DDP helpers --------------------
def is_distributed_run():
    return int(os.getenv("WORLD_SIZE", "1")) > 1

def setup_ddp_if_needed():
    if is_distributed_run():
        # torchrun sets MASTER_ADDR/PORT, RANK, WORLD_SIZE, LOCAL_RANK
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
def load_split(pattern, name=''):
    graphs = []
    for p in tqdm(sorted(glob.glob(pattern)), desc="Loading the dataset:" + f" {name}", leave=False):
        graphs.extend(torch.load(p, map_location="cpu", weights_only=False))
    return graphs

def drop_empty(graphs, name=''):
    kept = [g for g in tqdm(graphs, desc="Dropping the empty graph:" + f" {name}", leave=False) if getattr(g, 'num_nodes', getattr(g, 'x', None).size(0))]
    return kept

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

# python3 evaluates.py
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument(
                        "-m",
                        "--models",
                        type=str,
                        default="/afs/cern.ch/user/s/schuetha/work/public/GNN/FCGNN/Full_model_GraphGPS.yaml",
                        help="Model path (YAML)",
    )
    parser.add_argument("-o", "--output", type=str, default="models_GNN_Graph_GPS", help="Output directory")
    args = parser.parse_args()

    setup_ddp_if_needed()
    rank, world_size = get_rank(), get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")

    # Load YAML
    with open(args.models) as f:
        config = yaml.safe_load(f)

    # -------- Initialize GraphGym cfg (needed by ExpanderEdgeFixer) --------
    init_graphgym_cfg_from_yaml_dict(config)

    # -------- Build model --------
    model = FCGNN(
                    config,
                    graph_level=True,
                ).to(device)
    print(model)
    print("Successfully loading the model.")

    data_path = "/eos/user/s/schuetha/signal_background_data_loader_all_flavour_oh_ori_300_cut"
    # -------- load data -------- 
    train_graphs = drop_empty(load_split(f"{data_path}/GNN_dataset_s_b_train.part*.pt", name="Train"), name="Train")
    print("Finish loaded the train data")
    predic_graphs = drop_empty(load_split(f"{data_path}/GNN_dataset_s_b_test.part*.pt", name="Test"), name="Test")
    print("Finish loaded the test data")
    
    # --- build contiguous labels from TRAIN only ---
    train_ys = np.array([int(g.y) for g in train_graphs])
    predic_ys = np.array([int(g.y) for g in predic_graphs])
    classes = sorted(np.unique(predic_ys).tolist())
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def remap_inplace(graphs):
        for g in graphs:
            g.y = torch.tensor(class_to_idx[int(g.y)], dtype=torch.long)

    remap_inplace(predic_graphs)
    remap_inplace(train_graphs)

    ys_remapped = np.array([int(g.y) for g in predic_graphs])  # 0..C-1
    ts_remapped = np.array([int(g.y) for g in train_graphs])

    counts = np.bincount(ts_remapped, minlength=num_classes)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-12)
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    if rank == 0:
        print(f"Classes (original): {classes}")
        print(f"Num classes       : {num_classes}")
        print(f"Remapped to       : 0..{num_classes-1}")
        print("Class counts       :", counts.tolist())
        print("Class weights      :", weights.tolist())
        print(f"Train graphs      : {len(train_graphs)}")

    # -------- samplers & loaders --------
    # Plain Python lists implement __len__/__getitem__, so PyTorch Sampler works fine.
    if is_distributed_run():
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_graphs, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        predic_sampler = DistributedSampler(predic_graphs, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    else:
        train_sampler, predic_sampler = None, None

    train_loader = DataLoader(
        train_graphs, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=2, pin_memory=True, persistent_workers=False
    )

    predic_loader = DataLoader(
        predic_graphs, batch_size=args.batch_size,
        sampler=predic_sampler, shuffle=(predic_sampler is None),
        num_workers=2, pin_memory=True, persistent_workers=False
    )

    # -------- model / loss / opt --------
    # in_dim = train_graphs[0].x.size(1)
    # print(f"Input feature dimension: {in_dim}")
    # model = Graphomer(in_dim=in_dim, out_dim=num_classes, hidden=64, dropout=0.3, k=16, graph_level=True).to(device)
    if is_distributed_run():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    model.load_state_dict(torch.load(f"{args.output}/{args.epoch}_epoch/model_{args.epoch}_epoch_best.pth", map_location=device))
    outdirs = f"{args.output}/plots_eval_{args.epoch}_epochs_signal_background"
    os.makedirs(outdirs, exist_ok=True)
    metrics = evaluation_multiclass(
                                    model,
                                    train_loader,
                                    predic_loader,             # or test_loader
                                    num_classes=num_classes,
                                    device=device,
                                    class_names=classes,  # or ["0","12","14"] if you inverted the mapping
                                    bins=50,
                                    outdir=outdirs
                                   )
    
    print(metrics)

    # quick empty-graph count (per-rank, but we print only rank 0)
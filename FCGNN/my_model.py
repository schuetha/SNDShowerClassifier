import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, GravNetConv
# from FCGNN.DirGravNet import DirGravNetConv
# from FCGNN.Graph_GPS import GraphGPS
# from graphgps.encoder.exp_edge_fixer import ExpanderEdgeFixer
# from graphgps.transform.expander_edges import generate_random_expander
from torch_geometric.data import Batch
# from FCGNN.Dir_local_GravNet import DirGravNetConv_local
# from FCGNN.GraphGPS_local import GraphGPS_local
from FCGNN.SNDShowerClassifier import SNDShowerClassifier
from FCGNN.Edge_conv import Edge_conv

CTOR_MAP = {
    # "DirGravNetConv": DirGravNetConv,
    "SNDShowerClassifier": SNDShowerClassifier,
    "SNDShowerClassifier_V2": SNDShowerClassifier_V2,
    "GravNetConv": GravNetConv,
    # "GraphGPS": GraphGPS,
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "Dropout": nn.Dropout,
    "BatchNorm1d": nn.BatchNorm1d,
    "ELU": nn.ELU,
    "DynamicEdgeConv": Edge_conv,
    # "DirGravNetConv_local": DirGravNetConv_local,
    # "GraphGPS_local": GraphGPS_local,
}

# GRAPH_LAYERS = (DirGravNetConv, GraphGPS, DirGravNetConv_local, GraphGPS_local)
GRAPH_LAYERS = (SNDShowerClassifier, GravNetConv, SNDShowerClassifier_V2)

# def add_expander_edges_to_batch_ptr(batch, degree, algorithm="Hamiltonian", rng=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     device = batch.x.device

#     # CPU copy ONLY for Python/NumPy logic
#     ptr_cpu = batch.ptr.cpu()

#     exp_edges_all = []

#     for g in range(ptr_cpu.numel() - 1):
#         start = int(ptr_cpu[g].item())
#         end   = int(ptr_cpu[g + 1].item())
#         n = end - start
#         if n <= 1:
#             continue

#         d = min(degree, n - 1)

#         if algorithm == "Random-d":
#             senders = list(range(n)) * d
#             receivers = []
#             for _ in range(d):
#                 receivers.extend(rng.permutation(list(range(n))).tolist())
#             senders, receivers = senders + receivers, receivers + senders

#         elif algorithm == "Random-d-2":
#             senders = list(range(n)) * d
#             receivers = rng.permutation(senders).tolist()
#             senders, receivers = senders + receivers, receivers + senders

#         elif algorithm == "Hamiltonian":
#             senders, receivers = [], []
#             for _ in range(d):
#                 perm = rng.permutation(list(range(n))).tolist()
#                 for idx, v in enumerate(perm):
#                     u = perm[idx - 1]
#                     senders.extend([v, u])
#                     receivers.extend([u, v])
#         else:
#             raise ValueError("algorithm must be Random-d, Random-d-2, or Hamiltonian")

#         senders = np.asarray(senders, dtype=np.int64)
#         receivers = np.asarray(receivers, dtype=np.int64)

#         mask = senders != receivers
#         senders = senders[mask] + start
#         receivers = receivers[mask] + start

#         edges = np.stack([senders, receivers], axis=1)  # [E,2]
#         exp_edges_all.append(torch.from_numpy(edges))    # still CPU

#     if len(exp_edges_all) == 0:
#         batch.expander_edges = torch.empty((0, 2), dtype=torch.long, device=device)
#     else:
#         batch.expander_edges = torch.cat(exp_edges_all, dim=0).to(device=device)

#     return batch

class FCGNN(nn.Module):
    def __init__(
        self,
        config,
        graph_level: bool = True,
    ):
        super().__init__()
        self.graph_level = graph_level
        self.plan = []

        for layer in config["model"]["layers"]:
            t = layer["type"]
            if t == "global_mean_pool":
                self.plan.append(("pool", None))
                continue
            params = layer.get("params", {})
            mod = CTOR_MAP[t](**params)
            name = layer.get("name") or f"auto_{len(self.plan)}"
            setattr(self, name, mod)
            self.plan.append(("call", name))

    def forward(self, batch):
        # ----- RUN PLAN -----
        for op, name in self.plan:
            if op == "call":
                mod = getattr(self, name)

                if isinstance(mod, SNDShowerClassifier):
                    return mod(batch)
                
                elif isinstance(mod, SNDShowerClassifier_V2):
                    return mod(batch)
                
                elif isinstance(mod, GravNetConv):
                    batch.x = mod(batch.x, batch)
                
                elif isinstance(mod, Edge_conv):
                    batch = mod(batch)

                # if isinstance(mod, DirGravNetConv):
                #     batch = mod(batch)
                
                # elif isinstance(mod, DirGravNetConv_local):
                #     batch = mod(batch)

                # elif isinstance(mod, GraphGPS):
                #     batch = mod(batch)

                # elif isinstance(mod, GraphGPS_local):
                #     batch = mod(batch)
                
                else:
                    batch.x = mod(batch.x)

            elif op == "pool":
                if self.graph_level:
                    batch.x = global_mean_pool(batch.x, batch.batch)

        return batch.x
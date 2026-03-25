"""
SNDShowerClassifier: GravNet + DGN Z-directional aggregation + Forward Centrality + GMT

Architecture:
    Block 0: Station centroid spread (NEW — per-hit topology features)
    Block 1: GravNet + DGN aggregation × N layers
    Block 2: Forward centrality from Z-filtered DAG (optional)
    Block 3: GMT pooling
    Block 4: MLP classifier × M layers

Ablation flags:
    use_dgn_aggregation: True  → [mean, max, smooth_Z, deriv_Z] (4 aggregations)
                         False → [mean, max] only (standard GravNet)
    use_centrality:      True  → append fc_local before GMT
                         False → skip centrality computation
    use_reachability:    True  → append fc_reach before GMT
                         False → skip reachability
    use_station_spread:  True  → prepend station centroid features (Block 0)
                         False → skip, original in_ch unchanged
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import knn_graph
from torch_geometric.nn.aggr import GraphMultisetTransformer
from torch_scatter import scatter_mean, scatter_max, scatter_sum

# ─── Lazy-import centrality to avoid CUDA compilation when unused ───
_centrality_fns = {}


def _get_centrality_fns():
    if not _centrality_fns:
        if torch.cuda.is_available():
            from Forward_Centrality_Transformer.forward_centrality import (
                forward_centrality as reach_ext,
            )
            from Forward_Centrality_Transformer.local_forward_degree import (
                local_centrality,
            )
        else:
            from Forward_Centrality_Transformer.forward_reachability import (
                forward_reachability as reach_ext,
            )
            from Forward_Centrality_Transformer.forward_local_centrality import (
                forward_local_centrality as local_centrality,
            )
        _centrality_fns["reach"] = reach_ext
        _centrality_fns["local"] = local_centrality
    return _centrality_fns["reach"], _centrality_fns["local"]


# ═══════════════════════════════════════════════════════════════════
# Block 0: Station Centroid Spread
# ═══════════════════════════════════════════════════════════════════

class StationCentroidBlock(nn.Module):
    """
    Compute per-hit shower topology features based on station centroids.

    For each hit, computes its distance from the centroid of all hits
    in the same (event, Z-station, orientation) group.

    Physics motivation:
        ν_e CC  → single tight EM core → hits cluster near centroid → SMALL distance
        NC π⁰  → two displaced γ cores → centroid sits between them → MEDIUM distance
        NC had → scattered fragments    → hits spread widely        → LARGE distance

    Since SciFi fibers alternate vertical (X) and horizontal (Y),
    each hit only measures one coordinate. We group by orientation
    so that X-measuring and Y-measuring hits get separate centroids.

    Output features per hit (concatenated to existing node features):
        dist_from_centroid:  |XY_hit - centroid|           (shower compactness)
        station_std:         std(XY) in same station       (shower width)
        station_nhits_frac:  n_hits_station / n_hits_event (energy concentration)

    Requires batch to have:
        batch.pos[:, 0]    = XY coordinate (X or Y depending on fiber orientation)
        batch.z_time[:, 0] = quantized Z (station identifier)
        batch.flag         = orientation (0=horizontal/Y, 1=vertical/X)
        batch.batch        = event index
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _make_group_id(batch_idx, z_q, ori):
        """
        Create unique contiguous group IDs for (event, station, orientation).
        Uses torch.unique on the composite key.
        """
        key = torch.stack([
            batch_idx.long(),
            z_q.long(),
            ori.long(),
        ], dim=1)  # (N, 3)
        _, group_id = torch.unique(key, dim=0, return_inverse=True)
        return group_id

    def forward(self, xy, z_q, ori, batch_idx):
        """
        Args:
            xy:        (N,) XY coordinate of each hit
            z_q:       (N,) quantized Z (station index)
            ori:       (N,) orientation flag (0 or 1)
            batch_idx: (N,) event index from batch.batch

        Returns:
            features:  (N, 3) tensor of [dist_from_centroid, station_std, station_nhits_frac]
        """
        N = xy.size(0)
        device = xy.device
        dtype = xy.dtype

        # ─── Unique group per (event, station, orientation) ───
        group_id = self._make_group_id(batch_idx, z_q, ori)
        n_groups = group_id.max().item() + 1

        # ─── 1. Centroid per group ───
        centroid = scatter_mean(xy, group_id, dim=0, dim_size=n_groups)

        # ─── 2. Distance from centroid ───
        dist = (xy - centroid[group_id]).abs()

        # ─── 3. Station std (shower width) ───
        # Var = E[X²] - E[X]²
        xy_sq_mean = scatter_mean(xy * xy, group_id, dim=0, dim_size=n_groups)
        variance = (xy_sq_mean - centroid * centroid).clamp(min=0)
        std_per_group = variance.sqrt()
        station_std = std_per_group[group_id]

        # ─── 4. Station hit fraction (what fraction of event hits are in this station) ───
        ones = torch.ones(N, device=device, dtype=dtype)
        # Hits per group
        nhits_group = scatter_sum(ones, group_id, dim=0, dim_size=n_groups)
        # Total hits per event
        nhits_event = scatter_sum(ones, batch_idx, dim=0)
        # Fraction
        frac = nhits_group[group_id] / nhits_event[batch_idx].clamp(min=1)

        # ─── Stack features ───
        features = torch.stack([dist, station_std, frac], dim=-1)  # (N, 3)

        return features


# ═══════════════════════════════════════════════════════════════════
# Block 1: GravNet + DGN Layer
# ═══════════════════════════════════════════════════════════════════

class GravNetDGNLayer(nn.Module):
    """
    Single GravNet layer with optional DGN-style Z-directional aggregation.

    Aggregations:
        Always:   mean, max           (standard GravNet)
        Optional: smooth_Z, deriv_Z   (DGN-inspired, if use_dgn=True)
    """

    def __init__(self, in_ch, out_ch, space_dim, prop_dim, k, use_dgn=True):
        super().__init__()
        self.k = k
        self.use_dgn = use_dgn

        self.lin_s = Linear(in_ch, space_dim)
        self.lin_h = Linear(in_ch, prop_dim)
        self.lin_skip = Linear(in_ch, out_ch, bias=False)

        n_agg = 4 if use_dgn else 2  # [mean, max] or [mean, max, smooth, deriv]
        self.lin_agg = Linear(n_agg * prop_dim, out_ch)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

        # Store edge_index for centrality (Block 2)
        self._edge_index = None

    def forward(self, x, z, batch):
        N = x.size(0)

        # Build undirected KNN in learned latent space
        s = self.lin_s(x)
        h = self.lin_h(x)
        edge_index = knn_graph(s, k=self.k, batch=batch)
        self._edge_index = edge_index

        # GravNet edge weights
        ew = (s[edge_index[0]] - s[edge_index[1]]).pow(2).sum(-1)
        ew = torch.exp(-10.0 * ew)

        src, dst = edge_index
        m = h[src] * ew.unsqueeze(-1)

        # ─── Standard aggregations (always) ───
        agg_mean = scatter_mean(m, dst, dim=0, dim_size=N)
        agg_max = scatter_max(m, dst, dim=0, dim_size=N)[0]

        if self.use_dgn:
            # ─── Z-directional field (DGN-inspired) ───
            dz = z[src] - z[dst]
            dz_abs_sum = scatter_sum(dz.abs(), dst, dim=0, dim_size=N)
            F = dz / (dz_abs_sum[dst] + 1e-8)

            agg_smooth = scatter_mean(
                m * F.abs().unsqueeze(-1), dst, dim=0, dim_size=N
            )
            agg_deriv = scatter_sum(
                m * F.unsqueeze(-1), dst, dim=0, dim_size=N
            )

            agg_all = torch.cat(
                [agg_mean, agg_max, agg_smooth, agg_deriv], dim=-1
            )
        else:
            agg_all = torch.cat([agg_mean, agg_max], dim=-1)

        out = self.lin_skip(x) + self.lin_agg(agg_all)
        out = self.act(self.norm(out))

        return out


# ═══════════════════════════════════════════════════════════════════
# Full Model
# ═══════════════════════════════════════════════════════════════════

class SNDShowerClassifier(nn.Module):
    """
    Full model for SND@LHC shower classification.

    Block 0: Station centroid spread features     (optional, NEW)
    Block 1: GravNet + DGN aggregation × N layers
    Block 2: Forward centrality from Z-filtered DAG
    Block 3: GMT pooling  (PyG nn.aggr API)
    Block 4: MLP classifier × M layers
    """

    STATION_SPREAD_DIM = 3  # [dist, std, frac]

    def __init__(
        self,
        in_ch,
        hidden_ch=64,
        space_dim=4,
        prop_dim=32,
        k=16,
        n_gravnet_layers=2,
        n_mlp_layers=2,
        gmt_heads=4,
        gmt_k=5,
        dropout=0.3,
        num_classes=4,
        use_dgn_aggregation=True,
        use_centrality=True,
        use_reachability=True,
        use_station_spread=True,
    ):
        super().__init__()
        self.use_centrality = use_centrality
        self.use_reachability = use_reachability
        self.use_station_spread = use_station_spread

        # ═══ Block 0: Station centroid spread ═══
        if use_station_spread:
            self.station_spread = StationCentroidBlock()
            gravnet_in_ch = in_ch + self.STATION_SPREAD_DIM
        else:
            self.station_spread = None
            gravnet_in_ch = in_ch

        # ═══ Block 1: GravNet + DGN × N ═══
        self.gravnet_layers = nn.ModuleList()
        for i in range(n_gravnet_layers):
            self.gravnet_layers.append(
                GravNetDGNLayer(
                    in_ch=gravnet_in_ch if i == 0 else hidden_ch,
                    out_ch=hidden_ch,
                    space_dim=space_dim,
                    prop_dim=prop_dim,
                    k=k,
                    use_dgn=use_dgn_aggregation,
                )
            )

        # ═══ Block 3: GMT pooling (PyG nn.aggr API) ═══
        gmt_in_ch = hidden_ch
        if use_centrality:
            gmt_in_ch += 1
        if use_reachability:
            gmt_in_ch += 1

        if gmt_in_ch != hidden_ch:
            self.pre_gmt_proj = Linear(gmt_in_ch, hidden_ch)
        else:
            self.pre_gmt_proj = None

        self.gmt = GraphMultisetTransformer(
            channels=hidden_ch,
            k=gmt_k,
            heads=gmt_heads,
            layer_norm=False,
            dropout=0.0,
        )

        # ═══ Block 4: MLP × M ═══
        mlp_layers = []
        for i in range(n_mlp_layers):
            mlp_layers.extend([
                Linear(hidden_ch, hidden_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        mlp_layers.append(Linear(hidden_ch, num_classes))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(self, batch):
        x = batch.x
        z = batch.z_time[:, 0]
        ptr = batch.ptr
        device = x.device

        # ═══ Block 0: Station centroid spread ═══
        if self.use_station_spread:
            xy = batch.pos[:, 0]              # XY coordinate
            z_q = batch.z_time[:, 0]          # quantized Z (station ID)
            ori = batch.flag.float()          # orientation (0 or 1)

            spread_feats = self.station_spread(
                xy, z_q, ori, batch.batch
            )  # (N, 3): [dist_from_centroid, station_std, nhits_frac]

            x = torch.cat([x, spread_feats], dim=-1)

        # ═══ Block 1: GravNet + DGN × N ═══
        h = x
        for layer in self.gravnet_layers:
            h = layer(h, z, batch.batch)

        # ═══ Block 2: Forward centrality (optional) ═══
        edge_index = self.gravnet_layers[-1]._edge_index

        if self.use_centrality or self.use_reachability:
            reach_ext, local_centrality = _get_centrality_fns()

            # Filter to forward edges → DAG
            src, dst = edge_index
            forward_mask = z[src] < z[dst]
            edge_index_dag = edge_index[:, forward_mask]

            extras = []

            if self.use_centrality:
                fc_local = local_centrality(
                    edge_index_dag,
                    ptr,
                    normalize=True,
                    k_norm=self.gravnet_layers[-1].k,
                ).view(-1, 1).to(device=device, dtype=h.dtype)
                fc_local = torch.log1p(fc_local.clamp(min=0))
                extras.append(fc_local)

            if self.use_reachability:
                fc_reach = reach_ext(
                    edge_index_dag,
                    ptr,
                    include_self=False,
                ).view(-1, 1).to(device=device, dtype=h.dtype)
                fc_reach = torch.log1p(fc_reach.clamp(min=0))
                extras.append(fc_reach)

            h_enriched = torch.cat([h] + extras, dim=-1)
        else:
            h_enriched = h

        # ═══ Project to GMT channels if needed ═══
        if self.pre_gmt_proj is not None:
            h_enriched = self.pre_gmt_proj(h_enriched)

        # ═══ Block 3: GMT pooling ═══
        event_feat = self.gmt(h_enriched, index=batch.batch)

        # ═══ Block 4: Classify ═══
        return self.classifier(event_feat)
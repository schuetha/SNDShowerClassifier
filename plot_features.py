#!/usr/bin/env python3
"""
Plot feature distributions (X, Y, Z, Energy) per class
from the GNN_data_*.pt file produced by GNN_s_b.py

Usage:
    python3 plot_features.py -i /path/to/GNN_data_train_small_correct_all_flavour.pt -o feature_distributions.pdf
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(path):
    """Load the .pt file and return (features_list, labels)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    features = data["features"]   # list of (N_hits, 6) tensors
    labels = np.asarray(data["flavours"], dtype=np.int64)
    return features, labels


def separate_xy_by_orientation(features, labels):
    """
    Split XY column into X (vertical fibers, ori=1) and Y (horizontal fibers, ori=0).
    
    Columns in each event tensor:
        0: XY  (X if vertical, Y if horizontal)
        1: Z
        2: Energy
        3. QDC
        4: dettype
        5: orientation (0=horizontal→Y, 1=vertical→X)
        6: time
    """
    class_data = {}  # {class_label: {'X': [], 'Y': [], 'Z': [], 'Energy': []}}

    classes = sorted(set(labels.tolist()))
    for c in classes:
        class_data[c] = {'X': [], 'Y': [], 'Z': [], 'Energy': [], 'QDC': []}

    for i, (ev, label) in enumerate(zip(features, labels)):
        ev_np = ev.numpy().astype(np.float32)
        if ev_np.shape[0] == 0:
            continue

        label = int(label)
        xy  = ev_np[:, 0]
        z   = ev_np[:, 1]
        e   = ev_np[:, 2]
        QDC = ev_np[:, 3]
        ori = ev_np[:, 4].astype(np.int64)
        
        # Vertical fibers measure X
        mask_x = ori == 1
        # Horizontal fibers measure Y
        mask_y = ori == 0

        if mask_x.sum() > 0:
            class_data[label]['X'].append(xy[mask_x])
        if mask_y.sum() > 0:
            class_data[label]['Y'].append(xy[mask_y])

        class_data[label]['Z'].append(z)
        class_data[label]['Energy'].append(e)
        class_data[label]['QDC'].append(QDC)

    # Concatenate all events per class
    for c in classes:
        for key in ['X', 'Y', 'Z', 'Energy', 'QDC']:
            if class_data[c][key]:
                class_data[c][key] = np.concatenate(class_data[c][key])
            else:
                class_data[c][key] = np.array([])

    return class_data, classes


def plot_distributions(class_data, classes, output_path, max_events=None):
    """Create the 2x2 panel plot of X, Y, Z, Energy distributions."""

    # Class labels and colors matching the screenshot style
    colors = {
        0:  '#1f77b4',   # blue  - Background
        12: '#ff7f0e',   # orange - CC nue
        14: '#2ca02c',   # green  - CC numu
        23: '#d62728',   # red    - NC
    }
    class_names = {0: '0', 12: '12', 14: '14', 23: '23'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ─── (a) X Distribution ───
    ax = axes[0, 0]
    for c in classes:
        x_data = class_data[c]['X']
        if len(x_data) == 0:
            continue
        # Filter out zeros (fake coordinates)
        x_data = x_data[x_data != 0]
        if len(x_data) == 0:
            continue
        ax.hist(x_data, bins=100, density=True, histtype='step',
                linewidth=1.2, color=colors.get(c, 'gray'),
                label=class_names.get(c, str(c)))
    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Probability')
    ax.set_title('X Distribution')
    ax.legend()

    # ─── (b) Y Distribution ───
    ax = axes[0, 1]
    for c in classes:
        y_data = class_data[c]['Y']
        if len(y_data) == 0:
            continue
        y_data = y_data[y_data != 0]
        if len(y_data) == 0:
            continue
        ax.hist(y_data, bins=100, density=True, histtype='step',
                linewidth=1.2, color=colors.get(c, 'gray'),
                label=class_names.get(c, str(c)))
    ax.set_xlabel('Y [cm]')
    ax.set_ylabel('Probability')
    ax.set_title('Y Distribution')
    ax.legend()

    # ─── (c) Z Distribution ───
    ax = axes[1, 0]
    for c in classes:
        z_data = class_data[c]['Z']
        if len(z_data) == 0:
            continue
        ax.hist(z_data, bins=100, density=True, histtype='step',
                linewidth=1.2, color=colors.get(c, 'gray'),
                label=class_names.get(c, str(c)))
    ax.set_xlabel('Z [cm]')
    ax.set_ylabel('Probability')
    ax.set_title('Z Distribution')
    ax.legend()

    # # ─── (d) Energy Distribution (log-scale) ───
    # ax = axes[1, 1]
    # for c in classes:
    #     e_data = class_data[c]['Energy']
    #     if len(e_data) == 0:
    #         continue
    #     # Filter valid energies and take log
    #     e_data = e_data[np.isfinite(e_data) & (e_data > 0)]
    #     if len(e_data) == 0:
    #         continue
    #     log_e = np.log(e_data)
    #     ax.hist(log_e, bins=100, density=True, histtype='step',
    #             linewidth=1.2, color=colors.get(c, 'gray'),
    #             label=class_names.get(c, str(c)))
    # ax.set_xlabel('log(Energy)')
    # ax.set_ylabel('Probability')
    # ax.set_title('Energy Distribution (log-scale)')
    # ax.legend()

    # ─── (e) QDC Distribution (log-scale) ───
    ax = axes[1, 1]
    for c in classes:
        QDC = class_data[c]['QDC']
        if len(QDC) == 0:
            continue
        # Filter valid QDC and take log
        QDC = QDC[np.isfinite(QDC) & (QDC > 0)]
        if len(QDC) == 0:
            continue
        log_QDC = np.log(QDC)
        ax.hist(log_QDC, bins=100, density=True, histtype='step',
                linewidth=1.2, color=colors.get(c, 'gray'),
                label=class_names.get(c, str(c)))
    ax.set_xlabel('log(QDC)')
    ax.set_ylabel('Probability')
    ax.set_title('QDC Distribution (log-scale)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

# python3 plot_features.py -i /eos/user/s/schuetha/signal_background_data_new_dataset_with_time/GNN_data_train_small_correct_all_flavour.pt -o feature_distributions_new_process_one.pdf

# python3 plot_features.py -i /eos/user/s/schuetha/signal_background_data_new_dataset/GNN_data_train_small_correct_all_flavour.pt -o feature_distributions_new_process_one_signal_background.pdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot feature distributions per class')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to GNN_data_*.pt file')
    parser.add_argument('-o', '--output', type=str, default='feature_distributions.pdf',
                        help='Output plot file (pdf/png)')
    parser.add_argument('-n', '--max_events', type=int, default=None,
                        help='Max events to process (for speed)')
    args = parser.parse_args()

    print("Loading data...")
    features, labels = load_data(args.input)

    if args.max_events is not None:
        features = features[:args.max_events]
        labels = labels[:args.max_events]

    print(f"Loaded {len(features)} events")
    print(f"Classes: {sorted(set(labels.tolist()))}")
    print(f"Counts: { {int(c): int((labels==c).sum()) for c in sorted(set(labels.tolist()))} }")

    print("Processing features...")
    class_data, classes = separate_xy_by_orientation(features, labels)

    print("Plotting...")
    plot_distributions(class_data, classes, args.output)
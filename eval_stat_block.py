import os, numpy as np, torch
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
    MulticlassF1Score, MulticlassAUROC, MulticlassConfusionMatrix
)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

@torch.no_grad()
def evaluation_multiclass(
    model,
    train_loader,
    test_loader,
    num_classes: int,
    device="cuda:0",
    class_names=None,                 # e.g., {"0": Neutral hadron, "12": Electron neutrino, "14": Muon neutrino, "23": Neutral current}
    bins: int = 50,
    outdir: str = "models_GNN/plots"
):
    """
    Evaluate a multiclass classifier on train & test.
    - Computes macro metrics (acc/prec/recall/F1/AUC) and confusion matrix
    - Plots per-class ROC curves (one-vs-rest) and confusion matrix heatmap
    - Plots per-class score histograms (train vs test)
    Returns a dict of key metrics.
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    model.to(device)

    # ---------- 1) Gather logits and labels ----------
    def collect(loader):
        all_logits, all_targets = [], []
        for batch in tqdm(loader, desc="Collecting outputs", leave=False):
            batch = batch.to(device, non_blocking=True)
            logits = model(batch)  # [B, C]
            all_logits.append(logits.detach().cpu())
            all_targets.append(batch.y.detach().cpu())
        return torch.cat(all_logits), torch.cat(all_targets)  # [N,C], [N]

    logits_tr, y_tr = collect(train_loader)
    logits_te, y_te = collect(test_loader)

    # Convert to probabilities
    # probs_tr = to_probs(logits_tr)
    # probs_te = to_probs(logits_te)
    # Always reset number of classes based on the model's actual output
    # num_classes = probs_te.shape[1]
    # print(f"[eval] Detected {num_classes} output classes from model")
    probs_tr = torch.softmax(logits_tr, dim=1)  # [N,C]
    probs_te = torch.softmax(logits_te, dim=1)  # [N,C]

    # ---------- 2) Torchmetrics (macro) ----------
    device_t = torch.device(device)
    probs_te_d = probs_te.to(device_t)
    y_te_d     = y_te.to(device_t)

    acc   = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device_t)(probs_te_d, y_te_d).item()
    prec  = MulticlassPrecision(num_classes=num_classes, average="macro").to(device_t)(probs_te_d, y_te_d).item()
    rec   = MulticlassRecall(num_classes=num_classes, average="macro").to(device_t)(probs_te_d, y_te_d).item()
    f1    = MulticlassF1Score(num_classes=num_classes, average="macro").to(device_t)(probs_te_d, y_te_d).item()
    auc_m = MulticlassAUROC(num_classes=num_classes, average="macro").to(device_t)(probs_te_d, y_te_d).item()

    cm    = MulticlassConfusionMatrix(num_classes=num_classes, normalize='true').to(device_t)(probs_te_d, y_te_d)
    cm    = cm.cpu().numpy()  # [C,C]

    # ---------- 3) ROC curves (OVR) with sklearn ----------
    y_true_np = y_te.numpy()
    P = probs_te.numpy()
    classes = np.arange(num_classes)
    if class_names is None:
        class_names = [str(c) for c in classes]

    aucs = {}
    plt.figure()

    if num_classes == 2:
        # Binary case: use the positive class (column 1)
        p_pos = P[:, 1] if P.shape[1] == 2 else P.squeeze()
        fpr, tpr, _ = roc_curve(y_true_np, p_pos)
        auc_pos = roc_auc_score(y_true_np, p_pos)
        aucs[class_names[1]] = float(auc_pos)
        # (Optional) you can also plot class 0 by symmetry,
        # but it's redundant because AUC0 == 1 - AUC1 if thresholds are symmetric.
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[1]} (AUC={auc_pos:.3f})")
        macro_auc = float(auc_pos)  # macro==binary AUC here
    else:
        # Multiclass OVR
        Y_bin = label_binarize(y_true_np, classes=classes)  # shape [N,C]
        for c in tqdm(classes, desc="ROC curves", leave=False):
            print(f"[eval] Computing ROC for class {c} ({class_names[c]})")
            fpr, tpr, _ = roc_curve(Y_bin[:, c], P[:, c])
            auc_c = roc_auc_score(Y_bin[:, c], P[:, c])
            aucs[class_names[c]] = float(auc_c)
            plt.plot(fpr, tpr, lw=2, label=f"{class_names[c]} (AUC={auc_c:.3f})")
        macro_auc = roc_auc_score(y_true_np, P, multi_class="ovr", average="macro")

    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (OVR), macro AUC={macro_auc:.3f}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_multiclass.pdf"))
    plt.close()

    # ---------- 4) Confusion matrix heatmap (matplotlib) ----------
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(classes); ax.set_xticklabels(class_names)
    ax.set_yticks(classes); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in tqdm(range(num_classes), desc="Confusion matrix", leave=False):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]:.6f}", ha="center", va="center",
                    color=("white" if cm[i,j] > cm.max()*0.5 else "black"), fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.pdf"))
    plt.close()

    # ---------- 5) Per-class score histograms (train vs test) ----------
    # For each class c, plot distribution of the model's score for that class on samples of class c
    for c in tqdm(classes, desc="Score histograms", leave=False):
        mask_tr = (y_tr.numpy() == c)
        mask_te = (y_te.numpy() == c)
        scores_tr = P_tr = probs_tr.numpy()[mask_tr, c]
        scores_te = P_te = probs_te.numpy()[mask_te, c]

        plt.figure(figsize=(6,4))
        plt.hist(scores_te, bins=bins, range=(0,1), density=True, alpha=0.35, label=f"{class_names[c]} (test)")
        plt.hist(scores_tr, bins=bins, range=(0,1), density=True, histtype="step", linewidth=1.5, label=f"{class_names[c]} (train)")
        plt.xlabel(f"P(class={class_names[c]})"); plt.ylabel("Density")
        plt.title(f"Score distrib: class {class_names[c]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"scores_class_{class_names[c]}.pdf"))
        plt.close()

    # ---------- 6) Return summary ----------
    return {
        "accuracy_micro": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "auc_macro_ovr": auc_m,          # torchmetrics macro AUC
        "auc_macro_sklearn_ovr": float(macro_auc),
        "auc_per_class": aucs,
        "confusion_matrix": cm,
    }
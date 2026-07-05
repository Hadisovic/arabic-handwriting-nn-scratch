import os
import numpy as np
import matplotlib.pyplot as plt
from train import diagnose

CLASS_NAMES = [
    'Alaf (a)', 'Ba (b)', 'Taa (t)', 'Tha (th)', 'Jeem (j)',
    'Haa (h)', 'Kha (kh)', 'Dal (d)', 'Thal (dh)', 'Raa (r)'
]
CLASS_NAMES_ASCII = [
    'Alaf  (0)', 'Ba    (1)', 'Taa   (2)', 'Tha   (3)', 'Jeem  (4)',
    'Haa   (5)', 'Kha   (6)', 'Dal   (7)', 'Thal  (8)', 'Raa   (9)'
]

COLORS = ['royalblue', 'tomato', 'forestgreen', 'darkorchid', 'darkorange', 'teal']

def _flag_box(ax, train_acc, test_acc, loc='upper right'):
    """Adds a colored diagnosis text box to a matplotlib axes."""
    label, color = diagnose(train_acc, test_acc)
    ax.text(0.98, 0.02, label,
            transform=ax.transAxes,
            fontsize=8, fontweight='bold', color='white',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.85))

def ensure_plots_dir(path='plots'):
    os.makedirs(path, exist_ok=True)
    return path

# ── Single model history ──────────────────────────────────────────────────────
def plot_single_history(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ep = range(1, len(history['train_loss']) + 1)

    ax1.plot(ep, history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    ax1.plot(ep, history['test_loss'],  label='Test Loss',  color='tomato',    lw=2, ls='--')
    ax1.set_title('Loss vs Epochs', fontsize=13)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, [a*100 for a in history['train_acc']], label='Train Acc', color='royalblue', lw=2)
    ax2.plot(ep, [a*100 for a in history['test_acc']],  label='Test Acc',  color='tomato',    lw=2, ls='--')
    ax2.set_title('Accuracy vs Epochs', fontsize=13)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    # Diagnosis flag on the accuracy plot
    _flag_box(ax2, history['train_acc'][-1], history['test_acc'][-1])

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ── Multi-config loss comparison ──────────────────────────────────────────────
def plot_loss_comparison(results_dict, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, (lbl, hist) in enumerate(results_dict.items()):
        c  = COLORS[i % len(COLORS)]
        ep = range(1, len(hist['train_loss']) + 1)
        ax1.plot(ep, hist['train_loss'],              label=lbl, color=c, lw=2)
        ax2.plot(ep, [a*100 for a in hist['test_acc']], label=lbl, color=c, lw=2)

    ax1.set_title('Training Loss Comparison'); ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss');      ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title('Test Accuracy Comparison'); ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)');            ax2.legend(); ax2.grid(True, alpha=0.3)

    # Add a diagnosis summary table as a text box on the loss plot
    lines = []
    for lbl, hist in results_dict.items():
        flag, _ = diagnose(hist['train_acc'][-1], hist['test_acc'][-1])
        lines.append(f"{lbl}: {flag}")
    summary = "\n".join(lines)
    ax1.text(0.02, 0.98, summary,
             transform=ax1.transAxes,
             fontsize=7, va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ── Per-config train-vs-test gap subplots ─────────────────────────────────────
def plot_train_test_gap(results_dict, title, save_path):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    if n == 1: axes = [axes]

    for ax, (lbl, hist) in zip(axes, results_dict.items()):
        ep = range(1, len(hist['train_acc']) + 1)
        ax.plot(ep, [a*100 for a in hist['train_acc']], label='Train', color='royalblue', lw=2)
        ax.plot(ep, [a*100 for a in hist['test_acc']],  label='Test',  color='tomato',    lw=2, ls='--')
        ft = hist['train_acc'][-1]*100
        fe = hist['test_acc'][-1]*100
        ax.set_title(f'{lbl}\nTrain={ft:.1f}%  Test={fe:.1f}%', fontsize=10)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

        # Diagnosis flag in the bottom-right of every subplot
        _flag_box(ax, hist['train_acc'][-1], hist['test_acc'][-1])

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, save_path, train_acc=None, test_acc=None):
    n  = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=11, fontweight='bold')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel('Predicted Class', fontsize=12, labelpad=10)
    ax.set_ylabel('Actual Class',    fontsize=12, labelpad=10)

    # Add diagnosis to the confusion matrix title
    if train_acc is not None and test_acc is not None:
        flag, color = diagnose(train_acc, test_acc)
        full_title = f"{title}\n[{flag}]"
        ax.set_title(full_title, fontsize=12, fontweight='bold', pad=15, color=color)
    else:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    per_class = cm.diagonal() / cm.sum(axis=1)
    overall   = cm.diagonal().sum() / cm.sum()
    print(f"  Overall Test Accuracy: {overall*100:.1f}%")
    for i, name in enumerate(CLASS_NAMES_ASCII):
        print(f"    {name:<14}: {per_class[i]*100:.1f}%")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    return cm

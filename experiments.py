import os
import numpy as np
from data_loader import load_dataset
from train import run_training, run_training_reg, diagnose_print
from visualize import ensure_plots_dir, plot_loss_comparison, plot_train_test_gap

def run_all_experiments(X_train, Y_train, X_test, Y_test, PLOTS="plots", EPOCHS=200):
    """Runs all 5 required experiments from the assignment and saves comparison plots."""
    all_runs = []   # (label, hyperparams_dict, history)

    # ── Experiment 1: Learning Rate Sensitivity ────────────────────────────────
    print("=" * 60)
    print("Experiment 1: Learning Rate Sensitivity (LR = 0.1, 0.01, 0.001)")
    print("=" * 60)
    exp1 = {}
    for lr in [0.1, 0.01, 0.001]:
        print(f"  Training LR={lr} ...")
        _, hist = run_training(X_train, Y_train, X_test, Y_test,
                               hidden_size=64, lr=lr, batch_size=16,
                               num_epochs=EPOCHS, seed=42)
        lbl = f'LR={lr}'
        exp1[lbl] = hist
        diagnose_print(lbl, hist['train_acc'][-1], hist['test_acc'][-1])
        all_runs.append((lbl, {'lr': lr, 'hidden': 64, 'batch': 16,
                                'dropout': 0.0, 'l2': 0.0}, hist))

    plot_loss_comparison(exp1, 'Experiment 1: Learning Rate Sensitivity',
                         f'{PLOTS}/exp1_learning_rate.png')
    plot_train_test_gap(exp1,  'Experiment 1: Train vs Test Accuracy per LR',
                        f'{PLOTS}/exp1_lr_gap.png')

    # ── Experiment 2: Batch Size Effect ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 2: Batch Size Effect (batch = 8, 16, 32)")
    print("=" * 60)
    exp2 = {}
    for bs in [8, 16, 32]:
        print(f"  Training batch={bs} ...")
        _, hist = run_training(X_train, Y_train, X_test, Y_test,
                               hidden_size=64, lr=0.01, batch_size=bs,
                               num_epochs=EPOCHS, seed=42)
        lbl = f'Batch={bs}'
        exp2[lbl] = hist
        diagnose_print(lbl, hist['train_acc'][-1], hist['test_acc'][-1])
        all_runs.append((lbl, {'lr': 0.01, 'hidden': 64, 'batch': bs,
                                'dropout': 0.0, 'l2': 0.0}, hist))

    plot_loss_comparison(exp2, 'Experiment 2: Batch Size Effect',
                         f'{PLOTS}/exp2_batch_size.png')
    plot_train_test_gap(exp2,  'Experiment 2: Train vs Test Accuracy per Batch Size',
                        f'{PLOTS}/exp2_batch_gap.png')

    # ── Experiment 3: Model Capacity ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Model Capacity (H = 16, 32, 64, 128)")
    print("=" * 60)
    exp3 = {}
    for h in [16, 32, 64, 128]:
        print(f"  Training hidden={h} ...")
        _, hist = run_training(X_train, Y_train, X_test, Y_test,
                               hidden_size=h, lr=0.01, batch_size=16,
                               num_epochs=EPOCHS, seed=42)
        lbl = f'H={h}'
        exp3[lbl] = hist
        diagnose_print(lbl, hist['train_acc'][-1], hist['test_acc'][-1])
        all_runs.append((lbl, {'lr': 0.01, 'hidden': h, 'batch': 16,
                                'dropout': 0.0, 'l2': 0.0}, hist))

    plot_loss_comparison(exp3, 'Experiment 3: Model Capacity (Hidden Neurons)',
                         f'{PLOTS}/exp3_capacity.png')
    plot_train_test_gap(exp3,  'Experiment 3: Underfitting vs Overfitting per Hidden Size',
                        f'{PLOTS}/exp3_capacity_gap.png')

    # ── Experiment 4: Regularization ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 4: Regularization (Dropout & L2 Weight Decay)")
    print("=" * 60)
    reg_configs = [
        ('No Regularization', 0.0,  0.0),
        ('Dropout=0.2',       0.2,  0.0),
        ('L2 lambda=0.01',    0.0,  0.01),
        ('Dropout+L2',        0.2,  0.01),
    ]
    exp4 = {}
    for lbl, drop, lam in reg_configs:
        print(f"  Training {lbl} ...")
        _, hist = run_training_reg(X_train, Y_train, X_test, Y_test,
                                   hidden_size=64, lr=0.01, batch_size=16,
                                   num_epochs=EPOCHS, seed=42,
                                   dropout_rate=drop, lambda_l2=lam)
        exp4[lbl] = hist
        diagnose_print(lbl, hist['train_acc'][-1], hist['test_acc'][-1])
        all_runs.append((lbl, {'lr': 0.01, 'hidden': 64, 'batch': 16,
                                'dropout': drop, 'l2': lam}, hist))

    plot_loss_comparison(exp4, 'Experiment 4: Regularization Comparison',
                         f'{PLOTS}/exp4_regularization.png')
    plot_train_test_gap(exp4,  'Experiment 4: Train vs Test Gap (Regularization)',
                        f'{PLOTS}/exp4_regularization_gap.png')

    # ── Experiment 5: Initialization Sensitivity ──────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 5: Initialization Sensitivity (seeds 42, 123, 7)")
    print("=" * 60)
    exp5 = {}
    for seed in [42, 123, 7]:
        print(f"  Training seed={seed} ...")
        _, hist = run_training(X_train, Y_train, X_test, Y_test,
                               hidden_size=64, lr=0.01, batch_size=16,
                               num_epochs=EPOCHS, seed=seed)
        lbl = f'Seed={seed}'
        exp5[lbl] = hist
        diagnose_print(lbl, hist['train_acc'][-1], hist['test_acc'][-1])
        all_runs.append((lbl, {'lr': 0.01, 'hidden': 64, 'batch': 16,
                                'dropout': 0.0, 'l2': 0.0}, hist))

    plot_loss_comparison(exp5, 'Experiment 5: Initialization Sensitivity (Different Seeds)',
                         f'{PLOTS}/exp5_init_sensitivity.png')
    
    return all_runs

if __name__ == "__main__":
    PLOTS = ensure_plots_dir('plots')
    EPOCHS = 200
    X_train, Y_train, X_test, Y_test = load_dataset()
    run_all_experiments(X_train, Y_train, X_test, Y_test, PLOTS, EPOCHS)

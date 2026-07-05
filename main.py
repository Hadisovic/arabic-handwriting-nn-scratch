import os
import numpy as np
from data_loader import load_dataset
from train import run_training_reg, diagnose, diagnose_print, forward
from visualize import ensure_plots_dir, plot_confusion_matrix, plot_single_history
from experiments import run_all_experiments

def main():
    PLOTS  = ensure_plots_dir('plots')
    EPOCHS = 200

    # 1. Load Data
    X_train, Y_train, X_test, Y_test = load_dataset()

    # 2. Run all 5 Required Experiments
    all_runs = run_all_experiments(X_train, Y_train, X_test, Y_test, PLOTS, EPOCHS)

    # 3. Regularization Grid Search to find Best Fit Model
    print("\n" + "=" * 60)
    print("Searching for GOOD FIT model (target: gap <= 4%)")
    print("=" * 60)

    reg_search = [
        (0.0,  0.0),
        (0.2,  0.0),
        (0.3,  0.0),
        (0.4,  0.0),
        (0.5,  0.0),
        (0.3,  0.001),
        (0.4,  0.001),
        (0.4,  0.005),
        (0.5,  0.001),
    ]

    gf_params  = None
    gf_hist    = None
    gf_label   = "dropout=0.0, L2=0.0"
    gf_dropout = 0.0
    gf_l2      = 0.0
    found_good  = False

    for drop, lam in reg_search:
        cfg = f"dropout={drop}, L2={lam}"
        print(f"  Trying {cfg} ...")
        p, h = run_training_reg(X_train, Y_train, X_test, Y_test,
                                hidden_size=64, lr=0.01, batch_size=16,
                                num_epochs=300, seed=42,
                                dropout_rate=drop, lambda_l2=lam)
        tr  = h['train_acc'][-1]
        te  = h['test_acc'][-1]
        flag, _ = diagnose(tr, te)
        print(f"    Train: {tr*100:.1f}%  |  Test: {te*100:.1f}%  |  [{flag}]")

        # Keep track of the first best model or smallest gap
        if gf_params is None:
            gf_params = p; gf_hist = h; gf_label = cfg
            gf_dropout = drop; gf_l2 = lam

        if "GOOD FIT" in flag:
            gf_params = p; gf_hist = h; gf_label = cfg
            gf_dropout = drop; gf_l2 = lam
            print(f"    *** GOOD FIT achieved — stopping search ***")
            found_good = True
            break
        elif (tr - te) * 100 < (gf_hist['train_acc'][-1] - gf_hist['test_acc'][-1]) * 100:
            gf_params = p; gf_hist = h; gf_label = cfg
            gf_dropout = drop; gf_l2 = lam

    if not found_good:
        print("\n  Note: GOOD FIT threshold not reached — using smallest-gap config.")

    bm_train = gf_hist['train_acc'][-1]
    bm_test  = gf_hist['test_acc'][-1]
    print(f"\n  Best fit config  : {gf_label}")
    diagnose_print("Final Best Model", bm_train, bm_test)

    # 4. Generate Confusion Matrix and History Plots
    A2_final, _ = forward(X_test,
                          gf_params['W1'], gf_params['b1'],
                          gf_params['W2'], gf_params['b2'])
    y_pred = np.argmax(A2_final, axis=1)
    y_true = np.argmax(Y_test,   axis=1)

    plot_confusion_matrix(y_true, y_pred,
                          title=f'Confusion Matrix — {gf_label}',
                          save_path=f'{PLOTS}/confusion_matrix.png',
                          train_acc=bm_train, test_acc=bm_test)

    plot_single_history(gf_hist,
                        title=f'Best Model History — {gf_label}',
                        save_path=f'{PLOTS}/best_model_history.png')

    # 5. Print Best Accuracy Model across Experiments
    best_lbl, best_hp, best_run = max(all_runs, key=lambda x: x[2]['test_acc'][-1])
    b_tr  = best_run['train_acc'][-1]
    b_te  = best_run['test_acc'][-1]
    b_gap = (b_tr - b_te) * 100
    b_flag, _ = diagnose(b_tr, b_te)

    # 6. Save best model parameters to weights file so classify.py can load them
    weights_file = "best_model_weights.npz"
    np.savez(weights_file, **gf_params)
    print("\n" + "=" * 60)
    print(f"SUCCESS: Best fit parameters saved successfully to '{weights_file}'!")
    print("=" * 60)

    # Final Summary Outputs
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nPlots saved to plots/ folder:")
    for f in sorted(os.listdir(PLOTS)):
        print(f"  plots/{f}")

    print("\n" + "=" * 60)
    print("BEST ACCURACY MODEL  (across 5 experiments)")
    print("=" * 60)
    print(f"  Configuration    : {best_lbl}")
    print(f"  Learning Rate    : {best_hp['lr']}")
    print(f"  Hidden Neurons   : {best_hp['hidden']}")
    print(f"  Batch Size       : {best_hp['batch']}")
    print(f"  Dropout Rate     : {best_hp['dropout']}")
    print(f"  L2 Lambda        : {best_hp['l2']}")
    print(f"  ---")
    print(f"  Train Accuracy   : {b_tr*100:.2f}%")
    print(f"  Test  Accuracy   : {b_te*100:.2f}%")
    print(f"  Gap              : {b_gap:.2f}%")
    print(f"  Diagnosis        : [{b_flag}]")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("BEST FIT MODEL  (minimized generalization gap)")
    print("=" * 60)
    em_tr  = bm_train * 100
    em_te  = bm_test  * 100
    em_gap = em_tr - em_te
    em_flag, _ = diagnose(bm_train, bm_test)
    print(f"  Config           : {gf_label}")
    print(f"  Dropout Rate     : {gf_dropout}")
    print(f"  L2 Lambda        : {gf_l2}")
    print(f"  ---")
    print(f"  Train Accuracy   : {em_tr:.2f}%")
    print(f"  Test  Accuracy   : {em_te:.2f}%")
    print(f"  Gap              : {em_gap:.2f}%")
    print(f"  Diagnosis        : [{em_flag}]")
    print("=" * 60)

    # Delete run_all.py file safely as requested
    run_all_file = "run_all.py"
    if os.path.exists(run_all_file):
        try:
            os.remove(run_all_file)
            print(f"\n[INFO] Cleaned up and deleted temporary file '{run_all_file}' successfully!")
        except Exception as e:
            print(f"\n[Warning] Could not automatically delete '{run_all_file}': {e}")

if __name__ == "__main__":
    main()

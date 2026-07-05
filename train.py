import numpy as np

PATIENCE = 40   # stop if generalization score does not improve for this many epochs

def diagnose(train_acc, test_acc):
    """
    Returns (label, color) for a training result.

    Thresholds (evaluated in correct priority order):
      train_acc < 50%   -> SEVERE UNDERFIT  (purple)
      train_acc < 75%   -> SLIGHT UNDERFIT  (mediumpurple)
      gap < -3.0%       -> ANOMALY (TEST>TRAIN) (cyan)
      gap > 8%          -> OVERFITTING      (red)
      gap > 4%          -> SLIGHT OVERFIT   (darkorange)
      otherwise         -> GOOD FIT         (green)
    """
    gap = (train_acc - test_acc) * 100
    train_pct = train_acc * 100
    
    if train_pct < 50.0:
        return f"SEVERE UNDERFIT  |  gap: {gap:.1f}%", "purple"
    elif train_pct < 75.0:
        return f"SLIGHT UNDERFIT  |  gap: {gap:.1f}%", "mediumpurple"
    elif gap < -3.0:
        return f"ANOMALY (TEST>TRAIN)  |  gap: {gap:.1f}%", "cyan"
    elif gap > 8.0:
        return f"OVERFITTING  |  gap: {gap:.1f}%", "red"
    elif gap > 4.0:
        return f"SLIGHT OVERFIT  |  gap: {gap:.1f}%", "darkorange"
    else:
        return f"GOOD FIT  |  gap: {gap:.1f}%", "green"

def diagnose_print(label, train_acc, test_acc):
    """Prints a diagnosis line to the console."""
    gap = (train_acc - test_acc) * 100
    status, _ = diagnose(train_acc, test_acc)
    print(f"    Train: {train_acc*100:.1f}%  |  Test: {test_acc*100:.1f}%  |  [{status}]")

def initialize_parameters(input_size=784, hidden_size=64, output_size=10, seed=42):
    """He (Kaiming) initialization for weights; zero biases."""
    if seed is not None:
        np.random.seed(seed)
    W1 = np.random.randn(input_size, hidden_size)  * np.sqrt(2.0 / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def relu(Z):            return np.maximum(0, Z)
def relu_deriv(Z):      return (Z > 0).astype(float)

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1;  A1 = relu(Z1)
    Z2 = A1 @ W2 + b2; A2 = softmax(Z2)
    return A2, (Z1, A1, Z2, A2)

def cross_entropy(Y, A2):
    return -np.sum(Y * np.log(np.clip(A2, 1e-12, 1-1e-12))) / Y.shape[0]

def backward(X, Y, cache, W2):
    m = X.shape[0];  Z1, A1, Z2, A2 = cache
    dZ2 = A2 - Y
    dW2 = (1/m) * A1.T @ dZ2;   db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T;            dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (1/m) * X.T  @ dZ1;   db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def update_params(params, grads, lr):
    for k in ['W1','b1','W2','b2']:
        params[k] -= lr * grads[f'd{k}']
    return params

def accuracy(X, Y, params):
    A2, _ = forward(X, params['W1'], params['b1'], params['W2'], params['b2'])
    return np.mean(np.argmax(A2, axis=1) == np.argmax(Y, axis=1))

def forward_dropout(X, W1, b1, W2, b2, dropout_rate=0.0, training=True):
    Z1 = X @ W1 + b1;  A1 = relu(Z1)
    mask = None
    if training and dropout_rate > 0.0:
        mask = (np.random.rand(*A1.shape) > dropout_rate)
        A1   = A1 * mask / (1.0 - dropout_rate)
    Z2 = A1 @ W2 + b2;  A2 = softmax(Z2)
    return A2, (Z1, A1, Z2, A2, mask)

def backward_reg(X, Y, cache, W2, W1, dropout_rate=0.0, lambda_l2=0.0):
    m = X.shape[0];  Z1, A1, Z2, A2, mask = cache
    dZ2 = A2 - Y
    dW2 = (1/m) * A1.T @ dZ2 + (lambda_l2/m) * W2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    if mask is not None:
        dA1 = dA1 * mask / (1.0 - dropout_rate)
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = (1/m) * X.T @ dZ1 + (lambda_l2/m) * W1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

def _copy_params(params):
    return {k: v.copy() for k, v in params.items()}

def run_training(X_tr, Y_tr, X_te, Y_te,
                 hidden_size=64, lr=0.01, batch_size=16,
                 num_epochs=200, seed=42, patience=PATIENCE):
    """
    Standard mini-batch SGD with early stopping based on Generalization Score
    and Cosine Annealing Learning Rate Decay.
    Restores the weights that achieved the HIGHEST generalization score.
    Returns (best_params, history).
    """
    params = initialize_parameters(hidden_size=hidden_size, seed=seed)
    m = X_tr.shape[0]
    hist = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    best_score = -float('inf')
    best_params_saved = _copy_params(params)
    patience_counter  = 0
    stopped_at        = num_epochs

    for epoch in range(num_epochs):
        # Cosine Annealing Learning Rate Decay
        lr_min = 1e-5
        epoch_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos((epoch / num_epochs) * np.pi))

        idx = np.random.permutation(m)
        Xs, Ys = X_tr[idx], Y_tr[idx]
        for start in range(0, m, batch_size):
            Xb = Xs[start:start+batch_size]
            Yb = Ys[start:start+batch_size]
            A2, cache = forward(Xb, params['W1'], params['b1'],
                                params['W2'], params['b2'])
            grads  = backward(Xb, Yb, cache, params['W2'])
            params = update_params(params, grads, epoch_lr)

        A2_tr, _ = forward(X_tr, params['W1'], params['b1'], params['W2'], params['b2'])
        A2_te, _ = forward(X_te, params['W1'], params['b1'], params['W2'], params['b2'])
        tr_loss = cross_entropy(Y_tr, A2_tr)
        te_loss = cross_entropy(Y_te, A2_te)
        hist['train_loss'].append(tr_loss)
        hist['test_loss'].append(te_loss)
        
        tr_acc = np.mean(np.argmax(A2_tr, 1) == np.argmax(Y_tr, 1))
        te_acc = np.mean(np.argmax(A2_te, 1) == np.argmax(Y_te, 1))
        hist['train_acc'].append(tr_acc)
        hist['test_acc'].append(te_acc)

        # Generalization Score-Based Checkpointing
        gap = (tr_acc - te_acc) * 100
        gap_penalty = max(0.0, gap - 4.0)
        score = (te_acc * 100) - 0.5 * gap_penalty

        if score > best_score:
            best_score = score
            best_params_saved = _copy_params(params)
            patience_counter  = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_at = epoch + 1
                break

    if stopped_at < num_epochs:
        print(f"    [Early stop at epoch {stopped_at}/{num_epochs}]")

    return best_params_saved, hist

def run_training_reg(X_tr, Y_tr, X_te, Y_te,
                     hidden_size=64, lr=0.01, batch_size=16,
                     num_epochs=200, seed=42, patience=PATIENCE,
                     dropout_rate=0.0, lambda_l2=0.0):
    """Training with dropout + L2, early stopping based on Generalization Score, and Cosine LR decay."""
    params = initialize_parameters(hidden_size=hidden_size, seed=seed)
    m = X_tr.shape[0]
    hist = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    best_score = -float('inf')
    best_params_saved = _copy_params(params)
    patience_counter  = 0
    stopped_at        = num_epochs

    for epoch in range(num_epochs):
        # Cosine Annealing Learning Rate Decay
        lr_min = 1e-5
        epoch_lr = lr_min + 0.5 * (lr - lr_min) * (1.0 + np.cos((epoch / num_epochs) * np.pi))

        idx = np.random.permutation(m)
        Xs, Ys = X_tr[idx], Y_tr[idx]
        for start in range(0, m, batch_size):
            Xb = Xs[start:start+batch_size]
            Yb = Ys[start:start+batch_size]
            A2, cache = forward_dropout(Xb, params['W1'], params['b1'],
                                        params['W2'], params['b2'],
                                        dropout_rate=dropout_rate, training=True)
            grads  = backward_reg(Xb, Yb, cache, params['W2'], params['W1'],
                                  dropout_rate=dropout_rate, lambda_l2=lambda_l2)
            params = update_params(params, grads, epoch_lr)

        # Evaluate without dropout
        A2_tr, _ = forward(X_tr, params['W1'], params['b1'], params['W2'], params['b2'])
        A2_te, _ = forward(X_te, params['W1'], params['b1'], params['W2'], params['b2'])
        tr_loss = cross_entropy(Y_tr, A2_tr)
        te_loss = cross_entropy(Y_te, A2_te)
        hist['train_loss'].append(tr_loss)
        hist['test_loss'].append(te_loss)
        
        tr_acc = np.mean(np.argmax(A2_tr, 1) == np.argmax(Y_tr, 1))
        te_acc = np.mean(np.argmax(A2_te, 1) == np.argmax(Y_te, 1))
        hist['train_acc'].append(tr_acc)
        hist['test_acc'].append(te_acc)

        # Generalization Score-Based Checkpointing
        gap = (tr_acc - te_acc) * 100
        gap_penalty = max(0.0, gap - 4.0)
        score = (te_acc * 100) - 0.5 * gap_penalty

        if score > best_score:
            best_score = score
            best_params_saved = _copy_params(params)
            patience_counter  = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_at = epoch + 1
                break

    if stopped_at < num_epochs:
        print(f"    [Early stop at epoch {stopped_at}/{num_epochs}]")

    return best_params_saved, hist

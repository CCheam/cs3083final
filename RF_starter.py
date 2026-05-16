# starter code for project 6
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import concurrent.futures

# The following code loads the MNIST training and test datasets
def load_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
'''

'''



# add your code here
#setup vars
BINARY_THRESHOLD=128
#c4.5
TREE_MAX_DEPTH=20
TREE_MIN_SAMP=5
TREE_MIN_GAIN=0.001
#RF
N_TREES=100
BOOTSTRAP_RATIO=1
MIN_SAMPLES=2
MIN_GAIN_RATIO=0.0
MAX_WORK=4
MAX_DEPTH=None
#Adaboost
ADA_LEARNERS=100
ADA_STUMP_DEPTH=3

#prep work
def binaryFlatten(X,limit=BINARY_THRESHOLD):
    flattened=X.reshape(X.shape[0],-1)
    return (flattened >=limit).astype(np.uint8)


def load_extra_dataset(path, label_col=-1, train_ratio=0.8):
    """
    Load a CSV extra dataset from Kaggle/UCI.
    Returns X_train, X_test, y_train, y_test as numpy arrays.
    Adjust skip_header / delimiter for your specific file.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    np.random.shuffle(data)
    split = int(len(data) * train_ratio)
    train, test = data[:split], data[split:]
    if label_col == -1:
        X_tr, y_tr = train[:, :-1], train[:, -1].astype(int)
        X_te, y_te = test[:, :-1],  test[:, -1].astype(int)
    else:
        X_tr = np.delete(train, label_col, axis=1)
        y_tr = train[:, label_col].astype(int)
        X_te = np.delete(test,  label_col, axis=1)
        y_te = test[:,  label_col].astype(int)
    return X_tr, X_te, y_tr, y_te
 #Node class
class Node:
    """
        Leaf node  : attribute=None, children={}, label=predicted class
    Split node : attribute=feature index to split on,
                 children={attribute_value: child_Node, ...},
                 label=majority class (fallback for unseen values)
    """
    __slots__ = ("attribute", "children", "label", "depth")
    def __init__(self):
        self.attribute = None   # int  — feature index to split on
        self.children  = {}     # dict — {value: Node}
        self.label     = None   # int  — majority class (leaf or fallback)
        self.depth     = 0      # int  — depth of this node in the tree
 
#C4.5 math
# 
def entropy(y):
    #H(y) entropy, 0.0 for empty arrays
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    # 0 * log2(0) is defined as 0 — use np.where to avoid nan
    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    return float(-np.sum(probs * log_probs))

def info_gain(X_col,y):
    #gains from spitting y by value in x_Col
    base = entropy(y)
    values, counts = np.unique(X_col, return_counts=True)
    n = len(y)
    weighted = sum(
        (cnt / n) * entropy(y[X_col == val])
        for val, cnt in zip(values, counts)
    )
    return base - weighted

def split_info(X_col):
    """
    SplitInfo — penalises attributes that produce many small branches.
    SplitInfo = 0 when the column is constant (only one unique value).
    """
    _, counts = np.unique(X_col, return_counts=True)
    probs = counts / counts.sum()
    log_probs = np.where(probs > 0, np.log2(probs), 0.0)
    return float(-np.sum(probs * log_probs))

def gain_ratio(X_col, y):
    """
    C4.5 GainRatio = InfoGain / SplitInfo.
    Returns 0.0 when SplitInfo == 0 to avoid division by zero
    """
    si = split_info(X_col)
    if si == 0.0:
        return 0.0
    return info_gain(X_col, y) / si

def majority_label(y):
    #Returns most frequent y label
    values, counts = np.unique(y, return_counts=True)
    return int(values[np.argmax(counts)])

#C4.5 builder
def build_c45_tree(X, y,available_attrs=None,depth=0,
    # ── pruning / complexity controls (single tree) ───────────────────
    max_depth=None,
    min_samples=2,
    min_gain_ratio=0.0,
    # ── randomness controls (used by random forest) ───────────────────────────
    random=False,
    n_features_subset=None,
    # ── sample weights (used by AdaBoost; pure C4.5 ignores this) ─────────────
    sample_weights=None,
):
    """
    Recursively build  C4.5 decision tree.
 
    Parameters
    ----------
    X                : (n_samples, n_features) feature matrix (binary for MNIST)
    y                : (n_samples,) integer label array
    available_attrs  : list of feature indices still eligible for splitting;
                       initialised to all features on the first call
    depth            : current recursion depth (used to enforce max_depth)
    max_depth        : hard depth limit; None = unlimited (used for single tree)
    min_samples      : do not split a node with fewer than this many examples
    min_gain_ratio   : do not split if best gain ratio is below this threshold
    random           : if True, pick a random subset of attributes at each node
                       (enables Random Forest diversity)
    n_features_subset: size of the random attribute subset when random=True
    sample_weights   : per-sample float weights; used by AdaBoost to bias the
                       weighted bootstrap — ignored by pure C4.5 / RF
 
    Returns
    -------
    Node (root of the subtree)
    """
    node = Node()
    node.depth = depth
    node.label = majority_label(y)   
 
    # Base case
    if len(np.unique(y)) == 1:
        return node                  # pure node — no split needed
 
    if available_attrs is None:
        available_attrs = list(range(X.shape[1]))
 
    if len(available_attrs) == 0:
        return node                  # no attributes left
 
    if len(y) < min_samples:
        return node                  # too few samples to split
 
    if max_depth is not None and depth >= max_depth:
        return node                  # depth limit reached
 
    # Select candidate attributes
    if random and n_features_subset is not None:
        k = min(n_features_subset, len(available_attrs))
        candidate_attrs = [int(a) for a in
        np.random.choice(available_attrs, size=k, replace=False)]
    else:
        candidate_attrs = available_attrs  # full C4.5: consider all remaining
 
    #Compute gain ratio for every candidate
    ratios = {attr: gain_ratio(X[:, attr], y) for attr in candidate_attrs}
    best_attr  = max(ratios, key=ratios.get)
    best_ratio = ratios[best_attr]
 
    if best_ratio <= min_gain_ratio:
        return node                  # no attribute improves enough — leaf
 
    # Split on best attribute
    node.attribute = int(best_attr)
    remaining_attrs = [a for a in available_attrs if a != best_attr]
 
    for val in np.unique(X[:, best_attr]):
        mask = X[:, best_attr] == val
        X_sub, y_sub = X[mask], y[mask]
        w_sub = sample_weights[mask] if sample_weights is not None else None
 
        if len(y_sub) == 0:
            # Empty branch: attach a leaf with the parent's majority label
            child = Node()
            child.label = node.label
            child.depth = depth + 1
        else:
            child = build_c45_tree(
                X_sub, y_sub,
                available_attrs=remaining_attrs,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples=min_samples,
                min_gain_ratio=min_gain_ratio,
                random=random,
                n_features_subset=n_features_subset,
                sample_weights=w_sub,
            )
        node.children[int(val)]=child
 
    return node

#prediction
def predict(node,x):
    #Classify x by traversing down, fall back to node label if new attribute
    if node.attribute is None or len(node.children) == 0:
        return node.label                    # leaf node
    val = int(x[node.attribute])
    if val in node.children:
        return predict(node.children[val], x)
    return node.label                        # unseen value — use majority fallback
 
def predict_tree(tree,X):
    """Classify all rows in X using a fitted tree. Returns int label array."""
    return np.array([predict(tree, x) for x in X])
 
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

#Confusian matrix
def confusion_matrix(y_true, y_pred, classes=None):
    """Return (cm, classes) where cm[i][j] = count of true=i predicted as j."""
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    n   = len(classes)
    idx = {int(c): i for i, c in enumerate(classes)}
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[int(t)]][idx[int(p)]] += 1
    return cm, classes
 
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

def resolve_n_features(n_features_total, setting):
    """
    Convert a feature-count setting to an integer.
    setting: "sqrt" | "log2" | int
    """
    if setting == "sqrt":
        return max(1, int(np.sqrt(n_features_total)))
    elif setting == "log2":
        return max(1, int(np.log2(n_features_total)))
    elif isinstance(setting, int):
        return max(1, min(setting, n_features_total))
    raise ValueError(f"Unknown n_features setting: {setting!r}")

#Random Forest
N_FEATURES      = "sqrt" 
def _build_one_rf_tree(args):
    """
    Worker function for one Random Forest tree.
    Must be a module-level function so ProcessPoolExecutor can pickle it.
 
    args = (tree_idx, X, y, bootstrap_size, n_features_subset,
            max_depth, min_samples, min_gain_ratio, seed)
    """
    (tree_idx, X, y, bootstrap_size,
     n_features_subset, max_depth, min_samples, min_gain_ratio, seed) = args
 
    np.random.seed(seed)   # each worker gets its own reproducible seed
 
    # ── Bootstrap sampling ────────────────────────────────────────────────────
    indices = np.random.choice(len(X), size=bootstrap_size, replace=True)
    X_boot, y_boot = X[indices], y[indices]
 
    tree = build_c45_tree(
        X_boot, y_boot,
        max_depth=max_depth,
        min_samples=min_samples,
        min_gain_ratio=min_gain_ratio,
        random=True,
        n_features_subset=n_features_subset,
    )
    print(f">>>tree {tree_idx} is finished")
    return tree
 
def train_random_forest(
    X, y,
    n_trees=N_TREES,
    bootstrap_ratio=BOOTSTRAP_RATIO,
    n_features=N_FEATURES,
    max_depth=MAX_DEPTH,
    min_samples=MIN_SAMPLES,
    min_gain_ratio=MIN_GAIN_RATIO,
    max_workers=MAX_WORK,
):
    """
    Train a Random Forest of C4.5 trees.
    Returns a list of fitted Node (tree root) objects.
    """
    bootstrap_size = int(len(X) * bootstrap_ratio)
    n_features_sub = resolve_n_features(X.shape[1], n_features)
 
    args_list = [
        (i, X, y, bootstrap_size, n_features_sub,
         max_depth, min_samples, min_gain_ratio,
         np.random.randint(0, 2**31))
        for i in range(n_trees)
    ]
 
    start = time.time()
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
            trees = list(exe.map(_build_one_rf_tree, args_list))
    else:
        trees = [_build_one_rf_tree(a) for a in args_list]   # sequential
 
    print(f"Total training time (seconds): {time.time() - start}")
    return trees
 
def predict_random_forest(trees, X):
    """
    Majority across all trees.
    Returns label array of shape (n_samples,).
    """
    # all_preds shape: (n_trees, n_samples)
    all_preds = np.array([predict_tree(tree, X) for tree in trees])
 
    def majority_vote(col):
        values, counts = np.unique(col, return_counts=True)
        return values[np.argmax(counts)]
 
    return np.apply_along_axis(majority_vote, axis=0, arr=all_preds)
 
 #adaboost
def train_adaboost(X, y, n_estimators=ADA_LEARNERS,stump_max_depth=ADA_STUMP_DEPTH,):
    """
    Train an AdaBoost ensemblrusing shallow C4.5 trees as weak learners.
 
    Returns
    -------
    estimators : list of (alpha, tree) tuples
    classes    : array of unique class labels
    """
    n_samples  = len(y)
    classes    = np.unique(y)
    K          = len(classes)
    weights    = np.ones(n_samples) / n_samples
    estimators = []
 
    for t in range(n_estimators):
        # ── Weighted bootstrap: sample proportional to current weights ─────────
        indices = np.random.choice(n_samples, size=n_samples,
                                   replace=True, p=weights)
        X_w, y_w = X[indices], y[indices]
 
        stump = build_c45_tree(
            X_w, y_w,
            max_depth=stump_max_depth,
            min_samples=1,
            min_gain_ratio=0.0,
            random=False,
        )
 
        # ── Weighted error on full training set ───────────────────────────────
        y_pred    = predict_tree(stump,X)
        incorrect = (y_pred != y).astype(float)
        err_t     = float(np.dot(weights, incorrect))
 
        if err_t <= 0:
            err_t = 1e-10           # perfect stump — clip to avoid log(inf)
        if err_t >= 1.0 - (1.0 / K):
            print(f"AdaBoost stopping early at t={t} (err={err_t:.4f})")
            break
 
        # ── SAMME alpha ───────────────────────────────────────────────────────
        alpha_t = np.log((1.0 - err_t) / err_t) + np.log(K - 1)
        estimators.append((alpha_t, stump))
 
        # ── Update and renormalise sample weights ─────────────────────────────
        weights *= np.exp(alpha_t * incorrect)
        weights /= weights.sum()
 
        print(f">>>AdaBoost stump {t} finished  err={err_t:.4f}  alpha={alpha_t:.4f}")
 
    return estimators, classes
 
def predict_adaboost(estimators, classes, X):
    """
    SAMME weighted vote. Returns label array of shape (n_samples,).
    """
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    scores = np.zeros((len(X), len(classes)))
    for alpha, stump in estimators:
        preds = predict_tree(stump,X)
        for i, p in enumerate(preds):
            if int(p) in class_to_idx:
                scores[i, class_to_idx[int(p)]] += alpha
    return classes[np.argmax(scores, axis=1)]


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED ENSEMBLE INTERFACE  ← single function, controlled by METHOD_FLAG
# ─────────────────────────────────────────────────────────────────────────────
METHOD_FLAG='random_forest'
def train_ensemble(X, y, method=METHOD_FLAG, **kwargs):
    """
    Train an ensemble model.
 
    Parameters
    ----------
    X      : (n_samples, n_features) preprocessed feature matrix
    y      : (n_samples,) integer label array
    method : "random_forest"  →  bagging + random feature subsets 
             "adaboost"       →  SAMME boosting with C4.5 stumps 
    kwargs : forwarded to the chosen method's training function
 
    Returns
    -------
    model dict:
        {"method": "random_forest", "payload": [tree, ...]}
        {"method": "adaboost",      "payload": [(alpha, tree), ...],
                                    "classes": array}
    """
    if method == "random_forest":
        trees = train_random_forest(X, y, **kwargs)
        return {"method": "random_forest", "payload": trees}
 
    elif method == "adaboost":
        estimators, classes = train_adaboost(X, y, **kwargs)
        return {"method": "adaboost", "payload": estimators, "classes": classes}
 
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'random_forest' or 'adaboost'."
        )
 
def predict_ensemble(model, X):
    """
    Predict labels using a model dict returned by train_ensemble().
    Dispatches automatically based on model["method"].
    """
    if model["method"] == "random_forest":
        return predict_random_forest(model["payload"], X)
    elif model["method"] == "adaboost":
        return predict_adaboost(model["payload"], model["classes"], X)
    else:
        raise ValueError(f"Unknown method in model: {model['method']!r}")
    
"""
 #Test sets
X_train = load_images("train-images-idx3-ubyte") # training images
y_train = load_labels("train-labels-idx1-ubyte") # training labels

X_test = load_images("t10k-images-idx3-ubyte") # test images
y_test = load_labels("t10k-labels-idx1-ubyte") # test labels

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
plt.imshow(X_train[0], cmap='gray')
plt.title("An image of " + str(y_train[0]))
plt.show()

X_train_bin = binaryFlatten(X_train)
X_test_bin  = binaryFlatten(X_test)
print(f"After binarization: train={X_train_bin.shape} test={X_test_bin.shape}")
#X_train_bin, y_train = X_train_bin[:2000], y_train[:2000]

#C4.5 Test
print("\n" + "="*60)
print("PART 1: Single C4.5 Tree")
print("="*60)
 
t0 = time.time()
single_tree = build_c45_tree(
    X_train_bin, y_train,
    max_depth=TREE_MAX_DEPTH,
    min_samples=TREE_MIN_SAMP,
    min_gain_ratio=TREE_MIN_GAIN,
    random=False,                   # no randomness for single tree
)
print(f"Single tree build time: {time.time() - t0:.2f}s")
 
y_pred_single = predict_tree(single_tree, X_test_bin)
acc_single    = accuracy(y_test, y_pred_single)
print(f"Accuracy of Single Tree: {acc_single:.4f}")
 
cm_single, classes = confusion_matrix(y_test, predict_tree(single_tree, X_test_bin))
print("-------- confusion matrix --------")
print(cm_single)
plot_confusion_matrix(cm_single, classes, title="Single C4.5 Tree — MNIST")

#Adaboost/random forest test
METHOD_FLAG='random_forest'
print("\n" + "="*60)
print(f"PART 2: Ensemble — {METHOD_FLAG}")
print("="*60)
 
t0    = time.time()
model = train_ensemble(X_train_bin, y_train, method=METHOD_FLAG)
print(f"Total training time (seconds): {time.time() - t0:.4f}")
 
y_pred_ens = predict_ensemble(model, X_test_bin)
acc_ens    = accuracy(y_test, y_pred_ens)
print(f"Accuracy of {METHOD_FLAG} is {acc_ens:.4f}")
 
cm_ens, _ = confusion_matrix(y_test, y_pred_ens)
print("-------- confusion matrix --------")
print(cm_ens)
plot_confusion_matrix(
    cm_ens, classes,
    title=f"{METHOD_FLAG.replace('_', ' ').title()} — MNIST"
)
 
"""

#extra data stuff
def load_breast_cancer_dataset(path, train_ratio=0.8):
    """
    Loads and preprocesses the Breast Cancer Wisconsin (Original) dataset.
    Expects the raw 'breast-cancer-wisconsin.data' file.
    """
 #treat numpy as NaN
    data = np.genfromtxt(path, delimiter=",", missing_values="?", filling_values=np.nan)
    
    # 2. Drop NaNs
    data = data[~np.isnan(data).any(axis=1)]
    
    # 3. Drop ID column to prevent overfitting
    data = data[:, 1:]
    
    # Shuffle  cleaned data
    np.random.seed(42) 
    np.random.shuffle(data)
    
    # 4. Split into training and testing sets
    split = int(len(data) * train_ratio)
    train, test = data[:split], data[split:]
    
    # Separate features (X) and target labels (y)
    X_tr, y_tr = train[:, :-1], train[:, -1].astype(int)
    X_te, y_te = test[:, :-1],  test[:, -1].astype(int)
    
    return X_tr, X_te, y_tr, y_te
print("\n" + "="*60)
print("PART 3: Breast Cancer Dataset Evaluation")
print("="*60)

X_train_bc, X_test_bc, y_train_bc, y_test_bc = load_breast_cancer_dataset("breast-cancer-wisconsin.data")
print(f"Loaded Cleaned Data -> Train: {X_train_bc.shape}, Test: {X_test_bc.shape}")

# 2. Test Single C4.5 Tree
print("\n--- Model 1: Single C4.5 Tree ---")
t0 = time.time()
single_tree_bc = build_c45_tree(
    X_train_bc, y_train_bc,
    max_depth=TREE_MAX_DEPTH,
    min_samples=TREE_MIN_SAMP,
    min_gain_ratio=TREE_MIN_GAIN,
    random=False
)
print(f"Build time: {time.time() - t0:.4f}s")
y_pred_single_bc = predict_tree(single_tree_bc, X_test_bc)
acc_single_bc = accuracy(y_test_bc, y_pred_single_bc)
print(f"Accuracy: {acc_single_bc:.4f}")

cm_single_bc, classes_bc = confusion_matrix(y_test_bc, y_pred_single_bc)
print("-------- confusion matrix --------")
print(cm_single_bc)
plot_confusion_matrix(cm_single_bc, classes_bc, title="Single C4.5 Tree — Breast Cancer")

print("\n--- Model 2: Random Forest ---")
t0 = time.time()
rf_model_bc = train_ensemble(
    X_train_bc, y_train_bc,
    method='random_forest',
    n_trees=N_TREES,
    max_depth=None, 
    min_samples=MIN_SAMPLES,
    min_gain_ratio=MIN_GAIN_RATIO
    max_workers=1
)
print(f"Build time: {time.time() - t0:.4f}s")
y_pred_rf_bc = predict_ensemble(rf_model_bc, X_test_bc)
acc_rf_bc = accuracy(y_test_bc, y_pred_rf_bc)
print(f"Accuracy: {acc_rf_bc:.4f}")

cm_rf_bc, _ = confusion_matrix(y_test_bc, y_pred_rf_bc, classes=classes_bc)
print("-------- confusion matrix --------")
print(cm_rf_bc)
plot_confusion_matrix(cm_rf_bc, classes_bc, title="Random Forest — Breast Cancer")


print("\n--- Model 3: AdaBoost ---")
t0 = time.time()
ada_model_bc = train_ensemble(
    X_train_bc, y_train_bc,
    method='adaboost',
    n_estimators=ADA_LEARNERS,
    stump_max_depth=ADA_STUMP_DEPTH
)
print(f"Build time: {time.time() - t0:.4f}s")
y_pred_ada_bc = predict_ensemble(ada_model_bc, X_test_bc)
acc_ada_bc = accuracy(y_test_bc, y_pred_ada_bc)
print(f"Accuracy: {acc_ada_bc:.4f}")

cm_ada_bc, _ = confusion_matrix(y_test_bc, y_pred_ada_bc, classes=classes_bc)
print("-------- confusion matrix --------")
print(cm_ada_bc)
plot_confusion_matrix(cm_ada_bc, classes_bc, title="AdaBoost — Breast Cancer")
import numpy as np
from sklearn.datasets import load_breast_cancer
np.random.seed(42)

def train_test_split(x, y, test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for i in np.unique(y):
        idx = np.where(y == i)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)
    rng.shuffle(train_idx); rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]

data = load_breast_cancer()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(x, y)
for name, labels in [("full", y), ("train", y_train), ("test", y_test)]:
    classes, counts = np.unique(labels, return_counts=True)
    print(name, dict(zip(classes,counts)), "| ratio:", np.round(counts / len(labels), 3))

# mu, sigma = x_train.mean(axis=0), x_train.std(axis=0)
# x_train_s = (x_train - mu) / sigma
# x_test_s  = (x_test  - mu) / sigma
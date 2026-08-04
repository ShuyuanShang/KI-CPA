import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def knn_purity(data, labels: np.ndarray, n_neighbors=30):
    """Computes KNN Purity for ``data`` given the labels.
        Parameters
        ----------
        data:
            Numpy ndarray of data
        labels
            Numpy ndarray of labels
        n_neighbors: int
            Number of nearest neighbors.
        Returns
        -------
        score: float
            KNN purity score. A float between 0 and 1.
    """
    # Handle empty input case
    if len(data) == 0 or len(labels) == 0:
        return 0.0

    labels = LabelEncoder().fit_transform(labels.ravel())

    # Handle case where there's only one sample
    if len(data) == 1:
        return 1.0

    try:
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
        indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]

        # Handle empty indices
        if indices.size == 0:
            return 0.0

        neighbors_labels = np.vectorize(lambda i: labels[i], otypes=[int])(indices)

        # pre cell purity scores
        scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
        res = [
            np.mean(scores[labels == i]) for i in np.unique(labels)
        ]  # per cell-type purity

        return np.mean(res) if res else 0.0
    except Exception:
        # Fallback for any numerical issues
        return 0.0


def entropy_batch_mixing(data, labels,
                         n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    """Computes Entory of Batch mixing metric for ``adata`` given the batch column name.
        Parameters
        ----------
        data
            Numpy ndarray of data
        labels
            Numpy ndarray of labels
        n_neighbors: int
            Number of nearest neighbors.
        n_pools: int
            Number of EBM computation which will be averaged.
        n_samples_per_pool: int
            Number of samples to be used in each pool of execution.
        Returns
        -------
        score: float
            EBM score. A float between zero and one.
    """
    # Handle empty input case
    if len(data) == 0 or len(labels) == 0:
        return 0.0

    def __entropy_from_indices(indices, n_cat):
        return entropy(np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)), base=n_cat)

    n_cat = len(np.unique(labels))
    # print(f'Calculating EBM with n_cat = {n_cat}')

    try:
        neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
        indices = neighbors.kneighbors(data, return_distance=False)[:, 1:]

        # Handle empty indices
        if indices.size == 0:
            return 0.0

        batch_indices = np.vectorize(lambda i: labels[i], otypes=[int])(indices)

        entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat)

        # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
        if n_pools == 1:
            score = np.mean(entropies)
        else:
            score = np.mean([
                np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
                for _ in range(n_pools)
            ])

        return score
    except Exception:
        # Fallback for any numerical issues
        return 0.0

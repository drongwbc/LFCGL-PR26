import scipy.io as scio
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F


def load_data(name, views):
    path = 'data/{}.mat'.format(name)
    data = scio.loadmat(path)
    labels = data['Y']
    labels = np.reshape(labels, (labels.shape[0], ))

    X = []
    for i in range(0, views):
        tmp = data['X' + str(i+1)]
        tmp = tmp.astype(np.float32)
        X.append(torch.from_numpy(tmp).to(dtype=torch.float))

    return X, labels


def graph_normalize(A):
    """
        Normalize the adj matrix
    """
    degree = torch.sum(A, dim=1).clamp_min_(1e-12).rsqrt()
    return (A * degree).t() * degree


def spectral_clustering(A, k, NCut=False, KMrep=1):
    D = np.diag(A.sum(0))
    lap = D - A
    if NCut:
        d_inv = np.linalg.pinv(D)
        sqrt_d_inv = np.sqrt(d_inv)
        lap = np.matmul(np.matmul(sqrt_d_inv, lap), sqrt_d_inv)
    x, V = np.linalg.eigh(lap)
    V = np.real(V)
    x = zip(x, range(len(x)))
    x = sorted(x, key=lambda x: x[0])
    H = np.vstack([V[:, i] for (v, i) in x[:k]]).T
    label = KMeans(n_clusters=k, n_init=KMrep, init='k-means++').fit(H).labels_
    return label


def spectral_clustering_GPU(A, k, NCut=False, KMrep=1):
    degree = torch.sum(A, dim=0).clamp_min_(1e-12)
    lap = -A
    lap.diagonal().add_(degree)
    if NCut:
        inv_sqrt_degree = degree.rsqrt()
        lap = lap * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(lap)

    _, indices = torch.topk(eigenvalues, k, largest=True, sorted=True)
    top_k_eigenvectors = eigenvectors[:, indices]
    top_k_eigenvectors = F.normalize(top_k_eigenvectors, p=2, dim=1)

    P = KMeans(n_clusters=k, n_init=KMrep,
               init='k-means++').fit(top_k_eigenvectors.detach().cpu().numpy()).labels_
    return P


def graph_regalizaried(lap, top_k_eigenvectors):
    return torch.trace(torch.mm(torch.mm(top_k_eigenvectors.T, lap), top_k_eigenvectors))


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    x = torch.sum(X * X, dim=0).unsqueeze(1)
    y = torch.sum(Y * Y, dim=0).unsqueeze(0)
    result = x + y - 2 * torch.mm(X.t(), Y)
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def build_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return: Graph 
    """
    size = X.shape[1]
    num_neighbors = min(num_neighbors, size - 1)
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    # Only the nearest neighbours are needed.  topk avoids sorting every row
    # of the full distance matrix, which is substantially cheaper for large n.
    nearest_distances, _ = torch.topk(
        distances, k=num_neighbors + 1, dim=1, largest=False, sorted=True)
    top_k = nearest_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(nearest_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).to(X.device)
        weights += torch.eye(size).to(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to(X.device)
    weights = weights.to(X.device)
    return weights, raw_weights


def JSD_loss(Z, AZ):
    """
        Loss of JSD (Z, AZ)
    """
    P = F.softmax(Z, dim=1)
    Q = F.softmax(AZ, dim=1)
    M = ((P + Q) / 2).log()
    loss = (F.kl_div(M, P, reduction='mean', size_average=True) +
            F.kl_div(M, Q, reduction='mean', size_average=True)) / 2
    return loss


def add_noise(X, ratio, mu=0, sigma=1):
    n = X[0].shape[0]
    views = len(X)
    num_noise = int(n * ratio)
    for i in range(0, views):
        noise_indice = torch.randperm(X[i].shape[0])[:num_noise]
        noise = mu + sigma * torch.randn(X[i].shape)
        mask = torch.zeros((X[i].shape))
        mask[noise_indice, :] = 1
        noise = noise * mask
        X[i] = X[i] + noise

    return X


def GL_Loss(A, X, par=2e+3):
    """
        Loss of Graph Learning
    """
    degree = A.sum(dim=1)
    # tr(X^T (D-A) X) without materializing the dense diagonal D.
    laplacian_quadratic = torch.sum(degree.unsqueeze(1) * X.square()) - \
        torch.sum(X * torch.mm(A, X))
    loss = laplacian_quadratic + (par / 2) * torch.norm(A)**2
    return loss.mean()


def Calinski_score(z, label, unique=False):
    """ 
        Calinski-Harabasz Index via Pytorch 
    """
    if unique:
        label = KMeans(n_clusters=len(np.unique(label)),
                       init='k-means++').fit(z.detach().cpu().numpy()).labels_

    unique_label = np.unique(label)
    samples = z.shape[0]
    n_clusters = len(np.unique(label))

    over_mean = torch.mean(z, dim=0)
    extra_dis, intra_dis = 0, 0
    for current_label in unique_label:
        cluster_points = z[label == current_label]
        mean_cluster = torch.mean(cluster_points, dim=0)
        extra_dis += len(cluster_points) * \
            torch.sum((mean_cluster - over_mean) ** 2)
        intra_dis += torch.sum((cluster_points - mean_cluster) ** 2)

    ch_index = extra_dis * (samples - n_clusters) / \
        (intra_dis * (n_clusters - 1))

    return ch_index


def ICVI(z, label):
    ICVI = 0
    for i in range(0, z.shape[0]):
        current_label = label[i]
        cluster_points = z[label == current_label]
        extra_dis = distance(z[i].reshape(z.shape[1], 1), cluster_points.T)
        ICVI += torch.sum(extra_dis) / (cluster_points.shape[0] - 1)

        other_points = z[label != current_label]
        intra_dis = distance(z[i].reshape(z.shape[1], 1), other_points.T)
        ICVI -= torch.sum(intra_dis) / other_points.shape[0]

    return ICVI


def GCL(z, S):
    GCL = 0
    thr = 0
    for i in range(0, z.shape[0]):
        cluster_points = z[S[i] > thr]
        extra_dis = distance(z[i].reshape(z.shape[1], 1), cluster_points.T)
        GCL += torch.sum(extra_dis) / (cluster_points.shape[0] - 1)

        other_points = z[S[i] <= thr]
        intra_dis = distance(z[i].reshape(z.shape[1], 1), other_points.T)
        GCL -= torch.sum(intra_dis) / other_points.shape[0]

    return GCL

    return GCL
>>>>>>> 3695aa0d5f1ff49ddeae83ed76f104d5baaf29b7

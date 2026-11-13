<<<<<<< HEAD
import torch
import utils
import torch.nn.functional as F


import torch.nn as nn
import warnings
import math
import time
from model import TestModel2, TestModel
from torch.optim import Adam
from sklearn.cluster import KMeans
from evaluation import eva
from sklearn import metrics
import random
import numpy as np


def setup_seed(seed):
    if seed == 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    setup_seed(args.seed)

    X, Y = utils.load_data(args.name, args.views)

    args.samples = X[0].shape[0]
    args.n_clusters = len(np.unique(Y))
    args.num_features = []
    neighbor = args.neighbor[:]
    upper_neighbor = math.floor(args.samples / args.n_clusters)

    print('---------Dataset Details---------')
    print('name: ' + args.name)
    print(f'samples: {args.samples:.0f}')
    print(f'cluster: {args.n_clusters:.0f}')
    print(f'views: {args.views:.0f}')

    print(f'dimension: {X[0].shape[1]:.0f} ', end='')
    args.num_features.append(X[0].shape[1])
    for i in range(1, args.views):
        print(X[i].shape[1], end=' ')
        args.num_features.append(X[i].shape[1])
    print('\n---------------------------------')

    A = []
    for i in range(0, args.views):
        X[i] = F.normalize(X[i], p=2, dim=1).to(device)
        tmp, _ = utils.build_CAN(X[i].T, args.neighbor[i])
        A.append(utils.graph_normalize(tmp.to(device)))

    curModel = TestModel(num_features=args.num_features,
                         hidden_size=args.hidden_size,
                         emblem_size=args.emblem_size,
                         views=args.views,
                         samples=args.samples).to(device)

    optimizer = Adam(curModel.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    print('---------Start Train---------')
    retScore = [0, 0, 0, 0]
    for epoch in range(0, args.max_epoch):
        curModel.train()
        Z, S, G, A_rec, P = curModel(
            X, A, neighbor, args.views, args.n_clusters)

        l1, l2, l3, l4 = 0, 0, 0, 0
        for i in range(0, args.views):
            l1 += F.binary_cross_entropy(
                A_rec[i].contiguous().view(-1), A[i].contiguous().view(-1))
            l2 += utils.GL_Loss(S[i].to_dense(), Z[i])
            l3 += nn.functional.kl_div(F.softmax(S[i].to_dense()).log(),
                                       F.softmax(G), reduction='sum')
            l4 += -utils.Calinski_score(Z[i].to_dense(), P)
        loss = l1 + l2 + args.alpha*l3 + args.beta*l4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Z, _, G, _, _ = curModel(
                X, A, neighbor, args.views, args.n_clusters)

            label = utils.spectral_clustering(
                G.cpu().numpy(), args.n_clusters, NCut=True, KMrep=10)

            acc, nmi, ari, f1 = eva(Y, label)

            retScore[0] = max(retScore[0], acc)
            retScore[1] = max(retScore[1], nmi)
            retScore[2] = max(retScore[2], ari)
            retScore[3] = max(retScore[3], f1)
            print(
                f'epoch {epoch}: acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')

            if (epoch + 1) % args.update_interval == 0:
                stop = True
                for i in range(0, args.views):
                    if neighbor[i] < upper_neighbor:
                        stop = False
                        # break
                if not stop:
                    for i in range(0, args.views):
                        neighbor[i] = min(
                            neighbor[i] + args.neighbor[i], upper_neighbor) 

    print("---------------------")
    print(
        f'final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')
    print(
        f'best  acc: {retScore[0]:.4f}, nmi: {retScore[1]:.4f}, ari: {retScore[2]:.4f}, f1: {retScore[3]:.4f}'
    )
=======
import torch
import utils
import torch.nn.functional as F


import torch.nn as nn
import warnings
import math
import time
from model import TestModel2, TestModel
from torch.optim import Adam
from sklearn.cluster import KMeans
from evaluation import eva
from sklearn import metrics
import random
import numpy as np


def setup_seed(seed):
    if seed == 0:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    setup_seed(args.seed)

    X, Y = utils.load_data(args.name, args.views)

    args.samples = X[0].shape[0]
    args.n_clusters = len(np.unique(Y))
    args.num_features = []
    neighbor = args.neighbor[:]
    upper_neighbor = math.floor(args.samples / args.n_clusters)

    print('---------Dataset Details---------')
    print('name: ' + args.name)
    print(f'samples: {args.samples:.0f}')
    print(f'cluster: {args.n_clusters:.0f}')
    print(f'views: {args.views:.0f}')

    print(f'dimension: {X[0].shape[1]:.0f} ', end='')
    args.num_features.append(X[0].shape[1])
    for i in range(1, args.views):
        print(X[i].shape[1], end=' ')
        args.num_features.append(X[i].shape[1])
    print('\n---------------------------------')

    A = []
    for i in range(0, args.views):
        X[i] = F.normalize(X[i], p=2, dim=1).to(device)
        tmp, _ = utils.build_CAN(X[i].T, args.neighbor[i])
        A.append(utils.graph_normalize(tmp.to(device)))

    curModel = TestModel(num_features=args.num_features,
                         hidden_size=args.hidden_size,
                         emblem_size=args.emblem_size,
                         views=args.views,
                         samples=args.samples).to(device)

    optimizer = Adam(curModel.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    print('---------Start Train---------')
    retScore = [0, 0, 0, 0]
    for epoch in range(0, args.max_epoch):
        curModel.train()
        Z, S, G, A_rec, P = curModel(
            X, A, neighbor, args.views, args.n_clusters)

        l1, l2, l3, l4 = 0, 0, 0, 0
        for i in range(0, args.views):
            l1 += F.binary_cross_entropy(
                A_rec[i].contiguous().view(-1), A[i].contiguous().view(-1))
            l2 += utils.GL_Loss(S[i].to_dense(), Z[i])
            l3 += nn.functional.kl_div(F.softmax(S[i].to_dense()).log(),
                                       F.softmax(G), reduction='sum')
            l4 += -utils.Calinski_score(Z[i].to_dense(), P)
        loss = l1 + l2 + args.alpha*l3 + args.beta*l4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Z, _, G, _, _ = curModel(
                X, A, neighbor, args.views, args.n_clusters)

            label = utils.spectral_clustering(
                G.cpu().numpy(), args.n_clusters, NCut=True, KMrep=10)

            acc, nmi, ari, f1 = eva(Y, label)

            retScore[0] = max(retScore[0], acc)
            retScore[1] = max(retScore[1], nmi)
            retScore[2] = max(retScore[2], ari)
            retScore[3] = max(retScore[3], f1)
            print(
                f'epoch {epoch}: acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')

            if (epoch + 1) % args.update_interval == 0:
                stop = True
                for i in range(0, args.views):
                    if neighbor[i] < upper_neighbor:
                        stop = False
                        # break
                if not stop:
                    for i in range(0, args.views):
                        neighbor[i] = min(
                            neighbor[i] + args.neighbor[i], upper_neighbor) 

    print("---------------------")
    print(
        f'final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1: {f1:.4f}')
    print(
        f'best  acc: {retScore[0]:.4f}, nmi: {retScore[1]:.4f}, ari: {retScore[2]:.4f}, f1: {retScore[3]:.4f}'
    )
>>>>>>> 3695aa0d5f1ff49ddeae83ed76f104d5baaf29b7

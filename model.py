<<<<<<< HEAD
import torch
import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class TestModel(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, views, samples):
        super(TestModel, self).__init__()
        self.gcn1_l1 = []
        self.gcn1_l2 = []
        self.mlp = []

        for i in range(0, views):
            # GCN Encoder
            self.gcn1_l1.append(
                nn.Linear(num_features[i], hidden_size[i], bias=False))
            self.gcn1_l2.append(
                nn.Linear(hidden_size[i], emblem_size[i], bias=False))

            # Graph Decoder
            self.mlp.append(nn.Linear(samples, samples, bias=False))

        self.gcn1_l1 = nn.ModuleList(self.gcn1_l1)
        self.gcn1_l2 = nn.ModuleList(self.gcn1_l2)
        self.mlp = nn.ModuleList(self.mlp)

        self.fusion = []

        for _ in range(samples):
            tmp = nn.Linear(views, 1, bias=False)
            nn.init.constant_(tmp.weight, float(1 / views))
            self.fusion.append(tmp)

        self.fusion = nn.ModuleList(self.fusion)

    def forward(self, X, A, neighbor, views, n_clusters):
        # Graph Encoder
        Z = []
        S = []
        for i in range(0, views):
            tmp_z = F.relu(self.gcn1_l1[i](torch.mm(A[i], X[i])))
            tmp_z = F.relu(self.gcn1_l2[i](torch.mm(A[i], tmp_z)))
            tmp_z = F.normalize(tmp_z, p=2, dim=1)
            tmp_s, _ = utils.build_CAN(tmp_z.T, neighbor[i])

            Z.append(tmp_z)
            S.append(tmp_s)

        # Graph Fusion
        stacked_S = torch.stack(S, dim=2)
        with torch.no_grad():
            for layer in self.fusion:
                layer.weight.copy_(F.normalize(layer.weight, p=1, dim=1))
        fusion_weight = torch.stack(
            [layer.weight.squeeze(0) for layer in self.fusion], dim=0)
        G = torch.sigmoid(torch.einsum('abv,bv->ab', stacked_S, fusion_weight))

        G = (G + G.T) / 2

        # Compute Pseudo-Label
        P = utils.spectral_clustering_GPU(
            G, n_clusters, NCut=True, KMrep=1)

        # Graph Decoder
        A_rec = []
        for i in range(0, views):
            tmp = F.sigmoid(self.mlp[i](G))
            tmp = (tmp + tmp.T) / 2
            A_rec.append(utils.graph_normalize(tmp))

        return Z, S, G, A_rec, P


class TestModel2(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, views, samples):
        super(TestModel2, self).__init__()
        self.gcn1_l1 = []
        self.gcn1_l2 = []
        self.mlp = []

        for i in range(0, views):
            # GCN Encoder
            self.gcn1_l1.append(
                nn.Linear(num_features[i], hidden_size[i], bias=False))
            self.gcn1_l2.append(
                nn.Linear(hidden_size[i], emblem_size[i], bias=False))

            # Graph Decoder
            self.mlp.append(nn.Linear(samples, samples, bias=False))

        self.gcn1_l1 = nn.ModuleList(self.gcn1_l1)
        self.gcn1_l2 = nn.ModuleList(self.gcn1_l2)
        self.mlp = nn.ModuleList(self.mlp)

        tmp = nn.Linear(views, 1, bias=False)
        nn.init.constant_(tmp.weight, float(1 / views))
        self.fusion = tmp

    def forward(self, X, A, neighbor, views, n_clusters):
        # Graph Encoder
        Z = []
        S = []
        for i in range(0, views):
            tmp_z = F.relu(self.gcn1_l1[i](torch.mm(A[i], X[i])))
            tmp_z = F.relu(self.gcn1_l2[i](torch.mm(A[i], tmp_z)))
            tmp_z = F.normalize(tmp_z, p=2, dim=1)
            tmp_s, _ = utils.build_CAN(tmp_z.T, neighbor[i])

            Z.append(tmp_z)
            S.append(tmp_s)

        # Graph Fusion
        stacked_S = torch.stack(S, dim=2)
        fusion_weight = F.normalize(self.fusion.weight.squeeze(0), p=1, dim=0)
        G = torch.sigmoid(torch.einsum('abv,v->ab', stacked_S, fusion_weight))

        G = (G + G.T) / 2

        # Compute Pseudo-Label
        P = utils.spectral_clustering_GPU(
            G, n_clusters, NCut=True, KMrep=1)

        # Graph Decoder
        A_rec = []
        for i in range(0, views):
            tmp = F.sigmoid(self.mlp[i](G))
            tmp = (tmp + tmp.T) / 2
            A_rec.append(utils.graph_normalize(tmp))

        return Z, S, G, A_rec, P
=======
import torch
import utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


class TestModel(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, views, samples):
        super(TestModel, self).__init__()
        self.gcn1_l1 = []
        self.gcn1_l2 = []
        self.mlp = []

        for i in range(0, views):
            # GCN Encoder
            self.gcn1_l1.append(
                nn.Linear(num_features[i], hidden_size[i], bias=False))
            self.gcn1_l2.append(
                nn.Linear(hidden_size[i], emblem_size[i], bias=False))

            # Graph Decoder
            self.mlp.append(nn.Linear(samples, samples, bias=False))

        self.gcn1_l1 = nn.ModuleList(self.gcn1_l1)
        self.gcn1_l2 = nn.ModuleList(self.gcn1_l2)
        self.mlp = nn.ModuleList(self.mlp)

        self.fusion = []

        for _ in range(samples):
            tmp = nn.Linear(views, 1, bias=False)
            nn.init.constant_(tmp.weight, float(1 / views))
            self.fusion.append(tmp)

        self.fusion = nn.ModuleList(self.fusion)

    def forward(self, X, A, neighbor, views, n_clusters):
        # Graph Encoder
        Z = []
        S = []
        for i in range(0, views):
            tmp_z = F.relu(self.gcn1_l1[i](torch.mm(A[i], X[i])))
            tmp_z = F.relu(self.gcn1_l2[i](torch.mm(A[i], tmp_z)))
            tmp_z = F.normalize(tmp_z, p=2, dim=1)
            tmp_s, _ = utils.build_CAN(tmp_z.T, neighbor[i])

            Z.append(tmp_z)
            S.append(tmp_s)

        # Graph Fusion
        stacked_S = torch.stack(S, dim=2)
        with torch.no_grad():
            for layer in self.fusion:
                layer.weight.copy_(F.normalize(layer.weight, p=1, dim=1))
        fusion_weight = torch.stack(
            [layer.weight.squeeze(0) for layer in self.fusion], dim=0)
        G = torch.sigmoid(torch.einsum('abv,bv->ab', stacked_S, fusion_weight))

        G = (G + G.T) / 2

        # Compute Pseudo-Label
        P = utils.spectral_clustering_GPU(
            G, n_clusters, NCut=True, KMrep=1)

        # Graph Decoder
        A_rec = []
        for i in range(0, views):
            tmp = F.sigmoid(self.mlp[i](G))
            tmp = (tmp + tmp.T) / 2
            A_rec.append(utils.graph_normalize(tmp))

        return Z, S, G, A_rec, P


class TestModel2(nn.Module):
    def __init__(self, num_features, hidden_size, emblem_size, views, samples):
        super(TestModel2, self).__init__()
        self.gcn1_l1 = []
        self.gcn1_l2 = []
        self.mlp = []

        for i in range(0, views):
            # GCN Encoder
            self.gcn1_l1.append(
                nn.Linear(num_features[i], hidden_size[i], bias=False))
            self.gcn1_l2.append(
                nn.Linear(hidden_size[i], emblem_size[i], bias=False))

            # Graph Decoder
            self.mlp.append(nn.Linear(samples, samples, bias=False))

        self.gcn1_l1 = nn.ModuleList(self.gcn1_l1)
        self.gcn1_l2 = nn.ModuleList(self.gcn1_l2)
        self.mlp = nn.ModuleList(self.mlp)

        tmp = nn.Linear(views, 1, bias=False)
        nn.init.constant_(tmp.weight, float(1 / views))
        self.fusion = tmp

    def forward(self, X, A, neighbor, views, n_clusters):
        # Graph Encoder
        Z = []
        S = []
        for i in range(0, views):
            tmp_z = F.relu(self.gcn1_l1[i](torch.mm(A[i], X[i])))
            tmp_z = F.relu(self.gcn1_l2[i](torch.mm(A[i], tmp_z)))
            tmp_z = F.normalize(tmp_z, p=2, dim=1)
            tmp_s, _ = utils.build_CAN(tmp_z.T, neighbor[i])

            Z.append(tmp_z)
            S.append(tmp_s)

        # Graph Fusion
        stacked_S = torch.stack(S, dim=2)
        fusion_weight = F.normalize(self.fusion.weight.squeeze(0), p=1, dim=0)
        G = torch.sigmoid(torch.einsum('abv,v->ab', stacked_S, fusion_weight))

        G = (G + G.T) / 2

        # Compute Pseudo-Label
        P = utils.spectral_clustering_GPU(
            G, n_clusters, NCut=True, KMrep=1)

        # Graph Decoder
        A_rec = []
        for i in range(0, views):
            tmp = F.sigmoid(self.mlp[i](G))
            tmp = (tmp + tmp.T) / 2
            A_rec.append(utils.graph_normalize(tmp))

        return Z, S, G, A_rec, P
>>>>>>> 3695aa0d5f1ff49ddeae83ed76f104d5baaf29b7

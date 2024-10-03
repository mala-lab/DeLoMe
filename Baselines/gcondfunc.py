import numpy as np
import scipy.sparse as sp
import torch
from deeprobust.graph.data import Dataset
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.utils import *
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Data
from collections import Counter
import dgl
import ipdb

def constructDGL(cond_adj, cond_feat, cond_label):
    adj_sparse = sp.csr_matrix(cond_adj.cpu())
    graph = dgl.from_scipy(adj_sparse)
    graph.ndata['feat'] = cond_feat.cpu()
    graph.ndata['label'] = cond_label.cpu()
    return graph


def Gprocess(graph, train_ids, valid_ids, test_ids, ids_per_cls):
    nodes_data = graph.ndata['feat'].cpu()
    edges_src, edges_dst = graph.edges()
    label = graph.dstdata['label'].cpu()
    dpr_data = Pyg2Dpr(edges_src.cpu().numpy(), edges_dst.cpu().numpy(), nodes_data, label, train_ids, valid_ids, test_ids, ids_per_cls)
    data = Transd2Ind(dpr_data)
    return data


class Pyg2Dpr(Dataset):
    def __init__(self, edges_src, edges_dst, nodes_data, label, train_ids, valid_ids, test_ids, ids_per_cls, **kwargs):
        num_nodes = nodes_data.shape[0]
        self.adj = sp.csr_matrix((np.ones(len(edges_src)), (edges_src, edges_dst)), shape=(num_nodes, num_nodes))
        self.features = nodes_data.numpy()
        self.labels = label.numpy()
        self.train_ids = train_ids
        self.valid_ids = valid_ids
        self.test_ids = test_ids
        self.ids_per_cls = ids_per_cls

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

class Transd2Ind:
    # transductive setting to inductive setting
    def __init__(self, dpr_data):
        idx_train = dpr_data.train_ids
        idx_valid = dpr_data.valid_ids
        idx_test = dpr_data.test_ids
        self.labels_train = dpr_data.labels[idx_train]
        self.adj_full, self.feat_full, self.labels_full = dpr_data.adj, dpr_data.features, dpr_data.labels
        self.idx_train = np.array(idx_train)
        self.idx_valid = np.array(idx_valid)
        self.idx_test = np.array(idx_test)
        nclass = Counter(self.labels_full)
        self.nclass = list(nclass.items())

        self.class_dict = None
        self.samplers = None
        self.class_dict2 = None

    def retrieve_class(self, c):
        if self.class_dict is None:
            self.class_dict = {}
            for i, (j, num) in enumerate(self.nclass):
                self.class_dict['class_%s'%j] = (self.labels_train == j)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)

    def retrieve_class_sampler(self, c, adj, transductive, nlayers):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i, (j, _) in enumerate(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == j]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==j]
                self.class_dict2[j] = idx

        if nlayers == 1:
            sizes = [15]
        if nlayers == 2:
            sizes = [10, 5]
        if nlayers == 3:
            sizes = [15, 10, 5]
        if nlayers == 4:
            sizes = [15, 10, 5, 5]
        if nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        # ipdb.set_trace()
        if self.samplers is None:
            self.samplers = []
            for i, (j, _) in enumerate(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[j])
                # ipdb.set_trace()
                self.samplers.append(NeighborSampler(adj, node_idx=node_idx, sizes=sizes, return_e_id=False, num_nodes=adj.size(0)))
        batch = np.random.permutation(self.class_dict2[c])
        for ix, (j, _) in enumerate(self.nclass):
            if c == j:
                index_sampler = ix
        out = self.samplers[index_sampler].sample(batch)
        return out

    def retrieve_class_multi_sampler(self, c, adj, transductive, nlayers, num=256):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i, (j, num) in enumerate(self.nclass):
                if transductive:
                    idx = self.idx_train[self.labels_train == j]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==j]
                self.class_dict2[j] = idx

        if self.samplers is None:
            self.samplers = []
            for l in range(2):
                layer_samplers = []
                sizes = [15] if l == 0 else [10, 5]
                for i, (j, num) in enumerate(self.nclass):
                    node_idx = torch.LongTensor(self.class_dict2[j])
                    layer_samplers.append(NeighborSampler(adj, node_idx=node_idx, sizes=sizes, batch_size=num, num_workers=12, return_e_id=False, num_nodes=adj.size(0), shuffle=True))
                self.samplers.append(layer_samplers)
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[nlayers-1][c].sample(batch)
        return out


def match_loss(gw_syn, gw_real, dis_metric, device):
    dis = torch.tensor(0.0).to(device)
    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)
    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    else:
        exit('DC error: unknown distance function')
    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape
    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def maxdegree(adj):
    n = adj.shape[0]
    return F.relu(max(adj.sum(1))/n - 0.5)

def sparsity2(adj):
    n = adj.shape[0]
    loss_degree = - torch.log(adj.sum(1)).sum() / n
    loss_fro = torch.norm(adj) / n
    return 0 * loss_degree + loss_fro

def sparsity(adj):
    n = adj.shape[0]
    thresh = n * n * 0.01
    return F.relu(adj.sum()-thresh)

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat

def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

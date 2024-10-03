import torch
import torch_sparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import product
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import edge_softmax
from dgl.utils import expand_as_pair
import dgl.function as fn
from deeprobust.graph import utils
import torch.optim as optim
import math
from copy import deepcopy
from .gcondfunc import row_normalize_tensor


class PGE(nn.Module):
    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None):
        super(PGE, self).__init__()
        nhid = 128
        nlayers = 3
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0

    def forward(self, x, inference=False):
        edge_index = self.edge_index
        edge_embed = torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=1)
        for ix, layer in enumerate(self.layers):
            edge_embed = layer(edge_embed)
            if ix != len(self.layers) - 1:
                edge_embed = self.bns[ix](edge_embed)
                edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)
        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        # adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SGC(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cuda'):

        super(SGC, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.conv = GraphConvolution(nfeat, nclass, with_bias=with_bias)
        self.nlayers = nlayers
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = False
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        weight = self.conv.weight
        bias = self.conv.bias
        x = torch.mm(x, weight)
        for i in range(self.nlayers):
            x = torch.spmm(adj, x)
        x = x + bias
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        weight = self.conv.weight
        bias = self.conv.bias
        x = torch.mm(x, weight)
        for ix, (adj, _, size) in enumerate(adjs):
            x = torch_sparse.matmul(adj, x)
        x = x + bias
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        weight = self.conv.weight
        bias = self.conv.bias
        x = torch.mm(x, weight)
        for ix, (adj) in enumerate(adjs):
            if type(adj) == torch.Tensor:
                x = adj @ x
            else:
                x = torch_sparse.matmul(adj, x)
        x = x + bias
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        self.conv.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

class GCNpyg(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, with_bn=False, device=None):
        super(GCNpyg, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.layers = nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, labels, offset1, offset2, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels
        self._train_with_val(labels, offset1, offset2, data, train_iters, verbose)

    def _train_with_val(self, labels, offset1, offset2, data, train_iters, verbose):
        feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_full = torch.LongTensor(data.labels_full).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0
        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output[:, offset1:offset2], labels-offset1)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)
                loss_val = F.nll_loss(output[data.idx_valid, offset1:offset2], labels_full[data.idx_valid]-offset1)
                acc_val = utils.accuracy(output[data.idx_valid, offset1:offset2], labels_full[data.idx_valid]-offset1)
                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)
            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        graph = graph.local_var().to('cuda:{}'.format(feat.get_device()))
        h = self.linear(feat)
        graph.ndata['h'] = h
        graph.update_all(gcn_msg, gcn_reduce)
        h = graph.ndata['h']

        graph.apply_edges(lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(graph.edata.pop('e'))

        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        return h, elist

    def forward_batch(self, block, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        feat_src, feat_dst = expand_as_pair(feat)
        h = self.linear(feat_src)
        block.srcdata['h'] = h
        block.update_all(gcn_msg, gcn_reduce)
        h = block.dstdata['h']

        block.apply_edges(lambda edges: {'e': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
        e = self.leaky_relu(block.edata.pop('e'))

        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        return h, elist

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

class GCNdgl(nn.Module):
    def __init__(self,
                 args):
        super(GCNdgl, self).__init__()
        dims = [args.d_data] + args.GCN_args['h_dims'] + [args.n_cls]
        self.dropout = args.GCN_args['dropout']
        self.gat_layers = nn.ModuleList()
        for l in range(len(dims)-1):
            self.gat_layers.append(GCNLayer(dims[l], dims[l+1]))

    def forward(self, g, features):
        e_list = []
        h = features
        for layer in self.gat_layers[:-1]:
            h, e = layer(g, h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1](g, h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = F.relu(h)
            e_list = e_list + e
            h = F.dropout(h, p=self.dropout, training=self.training)
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()
    
    def fit_with_val(self, cond_g, orig_g, offset1, offset2, idx_valid, args, train_iters=200):
        # self.reset_params()
        # ipdb.set_trace()
        self.opt = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        full_feat = orig_g.ndata['feat']
        full_label = orig_g.ndata['label'].squeeze() - offset1

        cond_feat = cond_g.ndata['feat']
        cond_label = cond_g.ndata['label'] - offset1
        best_val_acc = 0
        for i in range(train_iters):
            self.train()
            self.opt.zero_grad()
            cond_output, _ = self.forward(cond_g, cond_feat)
        
            loss = torch.nn.functional.cross_entropy(cond_output[:, offset1:offset2], cond_label)

            loss.backward()
            self.opt.step()

            if i % 10 == 0:
                with torch.no_grad():
                    self.eval()
                    orig_output, _ = self.forward(orig_g, full_feat)
                    val_acc = utils.accuracy(orig_output[idx_valid, offset1:offset2], full_label[idx_valid])
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        weights = deepcopy(self.state_dict())
        self.load_state_dict(weights)
        return best_val_acc
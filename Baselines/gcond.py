import torch
from .gcondfunc import *
from collections import Counter
import random
import torch.nn as nn
from torch_sparse import SparseTensor
from Backbones.gnns import SGC
from deeprobust.graph import utils
import matplotlib.pyplot as plt
class GCond:
    def __init__(self, args, g, t, train_ids, valid_ids, ids_per_cls, offset1, offset2, device='cuda', **kwargs):
        self.args =args
        self.graph = g
        self.task = t
        self.train_ids = train_ids
        self.valid_ids = valid_ids
        self.ids_per_cls = ids_per_cls
        self.offset1 = offset1
        self.offset2 = offset2
        self.device = device
        self.budget = int(self.args.DeLoMe_args['budget'])

        self.features = self.graph.dstdata['feat']
        self.labels = self.graph.dstdata['label']
        counter = Counter(self.labels.squeeze().cpu().numpy())
        self.nclass = list(counter.items())
        self.epochs = 900
        self.feat_lr = 0.0001
        self.layers = 2
        self.ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]

        self.nnodes_syn = 0
        for i in range(len(self.nclass)):
            self.nnodes_syn = self.nnodes_syn + min(self.budget, len(self.ids_per_cls_train[i]))
        self.feat_syn = nn.Parameter(torch.FloatTensor(self.nnodes_syn, self.features.shape[1]).to(device))

        self.labels_syn = torch.LongTensor(self.generate_labels_syn()).to(device)
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=self.feat_lr)
        self.reset_parameters()
        self.set_seed(42)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))
    
    def set_seed(self, seed):   
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dgl.random.seed(seed)

    def generate_labels_syn(self):
        num_class_dict = {}
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(self.nclass):
            num_class_dict[c] = min(self.budget, len(self.ids_per_cls_train[ix]))
            self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
            labels_syn += [c] * num_class_dict[c]
        self.num_class_dict = num_class_dict
        return labels_syn

    def get_condg(self, plot=False):
        cond_feat, cond_adj, cond_label = self.train()
        graph = constructDGL(cond_adj, cond_feat, cond_label)
        if plot:
            self.plot(cond_adj)
        return graph

    def train(self, verbose=True, sampler=True):
        
        if self.layers == 1:
            sizes = [15]
        if self.layers == 2:
            sizes = [10, 25]
        if self.layers == 3:
            sizes = [15, 10, 5]

        nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(sizes) if sampler else dgl.dataloading.MultiLayerFullNeighborSampler(self.layers)
        Igraph = dgl.graph(([i for i in range(self.nnodes_syn)], [i for i in range(self.nnodes_syn)])).to(self.device)
        syn_class_indices = self.syn_class_indices
        feat_sub = self.get_sub_adj_feat(self.features)
        self.feat_syn.data.copy_(feat_sub)
        
        outer_loop = 1
        val_acc = 0
        for it in range(self.epochs+1):
            loss_avg = 0
            model = SGC(self.args).to(self.device)
            model.reset_params()
            model_parameters = list(model.parameters())
            model.train()

            for ol in range(outer_loop):
                for ix, (c, _) in enumerate(self.nclass):
                    if c not in self.num_class_dict:
                        continue

                    train_ids = self.ids_per_cls_train[ix]
                    dataloader = dgl.dataloading.DataLoader(self.graph.cpu(), train_ids, nb_sampler, batch_size=2000, shuffle=True, drop_last=False)
                    for input_nodes, output_nodes, blocks in dataloader:
                        model.zero_grad()
                        blocks = [b.to(device='cuda:{}'.format(self.args.gpu)) for b in blocks]
                        input_features = blocks[0].srcdata['feat']
                        output_labels = blocks[-1].dstdata['label'].squeeze()
                        output_predictions, _ = model.forward_batch(blocks, input_features)
                        loss_real = F.nll_loss(output_predictions[:, self.offset1:self.offset2], output_labels-self.offset1)

                        gw_real = torch.autograd.grad(loss_real, model_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))
                        
                        ind = syn_class_indices[c]
                        output_syn, _ = model(Igraph, self.feat_syn)
                        loss_syn = F.nll_loss(output_syn[ind[0]:ind[1], self.offset1:self.offset2], self.labels_syn[ind[0]: ind[1]]-self.offset1)
                        gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)

                        coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                        loss = coeff * match_loss(gw_syn, gw_real, dis_metric='mse', device=self.device)

                        self.optimizer_feat.zero_grad()
                        loss.backward()
                        self.optimizer_feat.step()

            if it == self.epochs:
                out_feat_syn, out_labels_syn = self.feat_syn.detach(), self.labels_syn
                out_adj_syn = torch.eye(self.nnodes_syn).to(self.device)

        return out_feat_syn, out_adj_syn, out_labels_syn


    def get_sub_adj_feat(self, features):
        idx_selected = []
        for c in range(len(self.nclass)):
            tmp = random.sample(self.ids_per_cls_train[c], min(self.budget, len(self.ids_per_cls_train[c])))
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]
        return features


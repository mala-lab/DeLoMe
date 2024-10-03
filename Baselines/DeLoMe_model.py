import torch
import copy
from .ergnn_utils import *
import pickle
from .gcond import *
import ipdb

class NET(torch.nn.Module):

    def __init__(self, model, task_manager, args):
        super(NET, self).__init__()

        self.task_manager = task_manager
        # setup network
        self.net = model
        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup losses
        self.ce = torch.nn.functional.cross_entropy
        # setup memories
        self.current_task = -1
        self.cond_num = {}
        self.budget = int(args.DeLoMe_args['budget'])
        self.tro = args.DeLoMe_args['tro']
        self.aux_g = []
        self.adjustments = 0
        self.aux_loss_w_ = []
    
    def observe_task_IL(self, args, g, features, labels, t, train_ids, valid_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the task-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class.
        :param dataset: The entire dataset.

        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]         

        if t!=self.current_task:
            self.current_task = t
            if args.cls_balance == 'logita':
                label_freq = np.array(list(self.cond_num.values()))
                label_current = np.array([len(id) for id in ids_per_cls_train])
                label_freq_array = np.concatenate((label_freq, label_current), axis=0)
                label_freq_array = label_freq_array / label_freq_array.sum()
                adjustments = np.log(label_freq_array ** self.tro + 1e-12)
                adjustments = torch.from_numpy(adjustments)
                self.adjustments = adjustments.to(device='cuda:{}'.format(args.gpu))

            gcond = GCond(args, g, t, train_ids, valid_ids, ids_per_cls, offset1, offset2)
            condg = gcond.get_condg()
            print('Task:', self.current_task, 'condensed graph size:', condg.num_nodes(), 'Edges:', condg.num_edges())
            self.aux_g.append(condg.to(device='cuda:{}'.format(args.gpu)))
            labels_condg = condg.ndata['label']
            for j  in range(args.n_cls):
                if (labels_condg == j).sum() != 0:
                    self.cond_num[j] = (labels_condg == j).sum()
            
            if args.cls_balance == 'lossa':
                n_per_cls = [(labels_condg == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)

        self.net.train()
        self.net.zero_grad()
       
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]
        if args.cls_balance == 'logita':
            loss = self.ce(output[train_ids, offset1:offset2] + self.adjustments[offset1:offset2], output_labels-offset1)
        elif args.cls_balance == 'lossa':
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls] 
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])
        elif args.cls_balance == 'none':
            loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1)

        if t!=0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                if args.cls_balance == 'logita': 
                    loss_aux = self.ce(output[:, o1:o2] + self.adjustments[o1:o2], aux_labels - o1)
                else:
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1:o2])
                loss = loss + loss_aux

        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, valid_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the task-IL setting with mini-batch training.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param dataloader: The data loader for mini-batch training
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
        :param dataset: The entire dataset (currently not in use).

        """
        
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        
        if t != self.current_task:
            self.current_task = t
            if args.cls_balance == 'logita':
                label_frep = np.array(list(self.cond_num.values()))
                label_current = np.array([len(ids) for ids in ids_per_cls_train])
                label_frep_array = np.concatenate((label_frep, label_current), axis=0)
                label_frep_array = label_frep_array / label_frep_array.sum()
                adjustments = np.log(label_frep_array ** self.tro + 1e-12)
                adjustments = torch.from_numpy(adjustments)
                self.adjustments = adjustments.to(device='cuda:{}'.format(args.gpu))

            gcond = GCond(args, g, t, train_ids, valid_ids, ids_per_cls, offset1, offset2)
            condg = gcond.get_condg()
            print('Task:', self.current_task, 'condensed graph size:', condg.num_nodes(), 'Edges:', condg.num_edges())
            self.aux_g.append(condg.to(device='cuda:{}'.format(args.gpu)))
            labels_condg = condg.ndata['label']
            for j in range(args.n_cls):
                if (labels_condg == j).sum() != 0:
                    self.cond_num[j] = (labels_condg == j).sum()
            
            if args.cls_balance == 'lossa':
                n_per_cls = [(labels_condg == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
        
        self.net.train()
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions,_ = self.net.forward_batch(blocks, input_features)

            if args.cls_balance == 'logita':
                loss = self.ce(output_predictions[:, offset1:offset2] + self.adjustments[offset1:offset2], output_labels - offset1)
            elif args.cls_balance == 'lossa':
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                loss = self.ce(output[:, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                loss = self.ce(output[:, offset1:offset2], output_labels-offset1)

            if t != 0:
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    if args.cls_balance == 'logita': 
                        loss_aux = self.ce(output[:, o1:o2] + self.adjustments[o1:o2], aux_labels - o1)
                    else:
                        loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1: o2])
                    loss = loss + loss_aux
            loss.backward()
            self.opt.step()


    def observe(self, args, g, features, labels, t, train_ids, valid_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class.
        :param dataset: The entire dataset.

        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        offset1, offset2 = self.task_manager.get_label_offset(t)

        if t!=self.current_task:
            self.current_task = t
            if args.cls_balance == 'logita':
                label_freq = np.array(list(self.cond_num.values()))
                label_current = np.array([len(ids) for ids in ids_per_cls_train])
                label_freq_array = np.concatenate((label_freq, label_current), axis=0)
                label_freq_array = label_freq_array / label_freq_array.sum()
                adjustments = np.log(label_freq_array ** self.tro + 1e-12)
                adjustments = torch.from_numpy(adjustments)
                self.adjustments = adjustments.to(device='cuda:{}'.format(args.gpu))

            gcond = GCond(args, g, t, train_ids, valid_ids, ids_per_cls, offset1, offset2)
            condg = gcond.get_condg()
            print('Task:', self.current_task, 'condensed graph size:', condg.num_nodes(), 'Edges:', condg.num_edges())
            self.aux_g.append(condg.to(device='cuda:{}'.format(args.gpu)))
            labels_condg = condg.ndata['label']
            for j  in range(args.n_cls):
                if (labels_condg == j).sum() != 0:
                    self.cond_num[j] = (labels_condg == j).sum()

            if args.cls_balance == 'lossa':
                n_per_cls = [(labels_condg == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)

        self.net.train()
        self.net.zero_grad()
        
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]
        if args.cls_balance == 'logita':
            if args.classifier_increase:
                loss = self.ce(output[train_ids,offset1:offset2] + self.adjustments[offset1:offset2], labels[train_ids])
            else:
                loss = self.ce(output[train_ids] + self.adjustments, labels[train_ids])
        elif args.cls_balance == 'lossa':
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            if args.classifier_increase:
                loss = self.ce(output[train_ids,offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
            else:
                loss = self.ce(output[train_ids], labels[train_ids], weight=loss_w_)
        elif args.cls_balance == 'none':
            loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
            if args.classifier_increase:
                loss = self.ce(output[train_ids,offset1:offset2], labels[train_ids])
            else:
                loss = self.ce(output[train_ids], labels[train_ids])

        if t!=0:
            for oldt in range(t):
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                if args.cls_balance == 'logita':
                    if args.classifier_increase:
                        loss_aux = self.ce(output[:, offset1:offset2]+ self.adjustments[offset1:offset2], aux_labels)
                    else:
                        loss_aux = self.ce(output + self.adjustments, aux_labels)
                else:
                    if args.classifier_increase:
                        loss_aux = self.ce(output[:, offset1:offset2], aux_labels, weight=self.aux_loss_w_[oldt][offset1: offset2])
                    else:
                        loss_aux = self.ce(output, aux_labels, weight=self.aux_loss_w_[oldt])

                loss = loss + loss_aux

        loss.backward()
        self.opt.step()

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, train_ids, valid_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting with mini-batch training.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param dataloader: The data loader for mini-batch training
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
        :param dataset: The entire dataset (currently not in use).

        """
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        offset1, offset2 = self.task_manager.get_label_offset(t)

        if t != self.current_task:
            self.current_task = t
            if args.cls_balance == 'logita':
                label_frep = np.array(list(self.cond_num.values()))
                label_current = np.array([len(id) for id in ids_per_cls_train])
                label_frep_array = np.concatenate((label_frep, label_current), axis=0)
                label_frep_array = label_frep_array / label_frep_array.sum()
                adjustments = np.log(label_frep_array ** self.tro + 1e-12)
                adjustments = torch.from_numpy(adjustments)
                self.adjustments = adjustments.to(device='cuda:{}'.format(args.gpu))

            gcond = GCond(args, g, t, train_ids, valid_ids, ids_per_cls, offset1, offset2)
            condg = gcond.get_condg()
            print('Task:', self.current_task, 'condensed graph size:', condg.num_nodes(), 'Edges:', condg.num_edges())
            self.aux_g.append(condg.to(device='cuda:{}'.format(args.gpu)))
            labels_condg = condg.ndata['label']
            for j in range(args.n_cls):
                if (labels_condg == j).sum() != 0:
                    self.cond_num[j] = (labels_condg == j).sum()
            
            if args.cls_balance == 'lossa':
                n_per_cls = [(labels_condg == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                self.aux_loss_w_.append(loss_w_)
              
        self.net.train()
        
        for input_nodes, output_nodes, blocks in dataloader:
            self.net.zero_grad()
            blocks = [b.to(device='cuda:{}'.format(args.gpu)) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()
            output_predictions,_ = self.net.forward_batch(blocks, input_features)

            if args.cls_balance == 'logita':
                if args.classifier_increase:
                    loss = self.ce(output_predictions[:, offset1:offset2] + self.adjustments[offset1:offset2], output_labels)
                else:
                    loss = self.ce(output_predictions + self.adjustments, output_labels)
            elif args.cls_balance == 'lossa':
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                if args.classifier_increase:
                    loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])
                else:
                    loss = self.ce(output_predictions, output_labels, weight=loss_w_)
            elif args.cls_balance == 'none':
                loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))
                if args.classifier_increase:
                    loss = self.ce(output_predictions[:, offset1:offset2], output_labels)
                else:
                    loss = self.ce(output_predictions, output_labels)   
            
            if t != 0:
                for oldt in range(t):
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    if args.cls_balance == 'logita':
                        if args.classifier_increase:
                            loss_aux = self.ce(output[:, offset1:offset2]+ self.adjustments[offset1:offset2], aux_labels)
                        else:
                            loss_aux = self.ce(output + self.adjustments, aux_labels)
                    else:
                        if args.classifier_increase:
                            loss_aux = self.ce(output[:, offset1:offset2], aux_labels, weight=self.aux_loss_w_[oldt][offset1: offset2])
                        else:
                            loss_aux = self.ce(output, aux_labels, weight=self.aux_loss_w_[oldt])
                    loss = loss + loss_aux
            loss.backward()
            self.opt.step()





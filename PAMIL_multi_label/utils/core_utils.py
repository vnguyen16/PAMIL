import numpy as np
import torch
from utils.utils import *
from utils.train_utils import *
import os
import torch.nn as nn
from datasets.dataset_generic import save_splits
from datasets.dataset_generic_npy import get_split_loader
from models.model_protoattention import PAMIL, push
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F

# torch.multiprocessing.set_start_method('spawn', force=True)

configs = get_settings()

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count":0, "correct":0, "R_count":0, "R_correct":0} for i in range(self.n_classes)]  # get the recall
    
    def log(self, Y_hat, Y):  # log for multi-label
        # Y_hat [2,3]
        # Y  [1,1,0]
        Y = Y.squeeze()
        Y_hat = Y_hat.squeeze()
        for i in range(self.n_classes):
            true = int(Y[i])
            i = int(i)
            self.data[i]["count"] += 1
            if true:
                self.data[i]["R_count"] += 1
                if i in Y_hat:
                    self.data[i]["correct"] += 1
                    self.data[i]["R_correct"] += 1
            else:
                if i not in Y_hat:
                    self.data[i]["correct"] += 1
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        R_count = self.data[c]["R_count"]
        R_correct = self.data[c]["R_correct"]
        
        if count == 0: 
            acc = None
            recall = None
        else:
            acc = float(correct) / count
            if R_count == 0:
                recall = None
            else:
                recall = float(R_correct) / R_count
        
        return acc, correct, count, recall, R_correct, R_count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None
        
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'bce':
        loss_fn = nn.functional.binary_cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    

    if args.model_type in ['PAMIL']:
        model_dict.update({'n_protos': args.num_protos})
        model_dict.update({'proto_path': os.path.join(args.proto_path, f'{args.task}_{args.num_protos}_{cur}', f'train_instance_feats_proto.npy')})
        model_dict.update({'proto_pred': args.proto_pred})
        model_dict.update({'proto_weight': args.proto_weight})
        model_dict.update({'inst_pred': args.inst_pred})
        model_dict.update({'k_sample': args.k_sample})
        model = PAMIL(**model_dict)
        
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    if args.model_type in ['PAMIL']:
        # init the mode for training
        # mode = TrainMode.WARM
        # for epoch in range(args.max_epochs):
        #     if mode == TrainMode.WARM:
        #         warm_only(model)
        #         if epoch >= 40:
        #             mode = TrainMode.JOINT
        #     elif mode == TrainMode.JOINT:
        #         joint(model)
        #     train_loop_pamil(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn)
        #     stop = validate_pamil(cur, epoch, model, val_loader, args.n_classes, args,
        #         early_stopping, writer, loss_fn, args.results_dir)
        #     if stop: 
        #         break
        # init the mode
        mode = TrainMode.WARM
        # mode = TrainMode.PUSH
        epoch = 0
        iteration = 0
        warm_only(model)
        while True:
            if mode == TrainMode.WARM:
                train_loop_pamil(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn)
                stop = validate_pamil(cur, epoch, model, val_loader, args.n_classes, args,
                    early_stopping, writer, loss_fn, args.results_dir)
                epoch += 1
                if epoch >= configs.num_warm_epochs:
                    mode = TrainMode.JOINT
                    joint(model)
            elif mode == TrainMode.JOINT:
                train_loop_pamil(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn)
                stop = validate_pamil(cur, epoch, model, val_loader, args.n_classes, args,
                    early_stopping, writer, loss_fn, args.results_dir)
                epoch += 1
                if epoch in configs.push_epochs and epoch >= configs.push_start:
                    mode = TrainMode.PUSH
            elif mode == TrainMode.PUSH:
                push(train_loader, model, epoch, save_proto = args.save_proto, 
                     result_dir=args.results_dir, cur=cur)
                mode = TrainMode.LAST
                last_only(model)
            elif mode == TrainMode.LAST:
                train_loop_pamil(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn)
                stop = validate_pamil(cur, epoch, model, val_loader, args.n_classes, args,
                    early_stopping, writer, loss_fn, args.results_dir)
                epoch += 1
                iteration += 1
                if iteration >= configs.num_last_layer_iterations:
                    mode = TrainMode.JOINT
                    iteration = 0
                
            if epoch >= configs.num_train_epochs:
            # if epoch >= configs.num_train_epochs or stop:
                break
    # save the final results
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_final_checkpoint.pt".format(cur)))
    
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, test_iauc, acc_logger = summary(model, test_loader, args)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    
    total_correct = 0
    total_count = 0
    for i in range(args.n_classes):
        acc, correct, count, recall, r_correct, r_count = acc_logger.get_summary(i)
        total_correct += r_correct
        total_count += r_count        
        print(f'class {i}: acc {acc}, correct {correct}/{count}, recall {recall} = {r_correct}/{r_count}')
        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)
            writer.add_scalar('final/test_{}_recall'.format(i), recall, 0)
            
    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_iauc', val_iauc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_iauc', test_iauc, 0)
        writer.close()
    test_recall = total_correct / total_count
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, test_recall, test_iauc, val_iauc


def train_loop_pamil(epoch, model, loader, optimizer, n_classes, args, writer = None, loss_fn=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_er_loss = 0.
    train_inst_loss = 0.
    inst_count = 0
    all_inst_label = []
    all_inst_score = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader), n_classes))
    print('\n')
    for batch_idx, (data, label, cors, inst_label, _) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=args.inst_pred, label=label)
        Y_prob = Y_prob.squeeze(0)
        inst_label = inst_label[0]
        score = instance_dict['A_raw'].T
        loss_er = instance_dict['loss_er']
        loss_clst = instance_dict['loss_clst']
        index_label = torch.nonzero(label.squeeze(0)).to(device)
                
        
        # loss function
        loss_cls = loss_fn(Y_prob, label.squeeze().float())
        
        # # === get the instance loss, v15 ===
        # loss_inst = 0.
        # sample_num = 5
        # inst_loss_fn = nn.functional.binary_cross_entropy
        # inst_score = F.softmax(score, dim=1)  # softmax over the dim of classes
        # for c in range(n_classes):
        #     sample_score, sample_index = torch.topk(inst_score[:, c], sample_num)
        #     inst_label = torch.tensor([label[0, c] for _ in range(sample_num)]).cuda()
        #     loss_inst += inst_loss_fn(sample_score, inst_label.float()) / sample_num
        # loss_inst = loss_inst / n_classes
        # train_inst_loss += loss_inst.item()  # v15
        
        # loss
        total_loss = loss_cls + args.w_er * loss_er + args.w_clst * loss_clst
        if args.inst_pred:
            loss_inst = instance_dict['loss_inst']
            train_inst_loss += loss_inst.item()
            total_loss += args.w_inst * loss_inst  # use the inst loss like clam，v16
        elif args.attention_er:
            loss_att_er = instance_dict['loss_att_er']
            total_loss += args.w_att_er * loss_att_er  # v18
        elif args.w_proto_clst != 0:
            loss_proto_clst = instance_dict['loss_proto_clst']
            total_loss += args.w_proto_clst * loss_proto_clst  # v21
        
        loss_value = total_loss.item()
        train_loss += loss_value
        
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        # inst-level evaluation
        if inst_label != []:
            all_inst_label += inst_label
            inst_score = score.detach().cpu().numpy()

            if args.task == 'camelyon':
                label_hot = np.array([[label, 1 - label] for label in inst_label])
            else:
                label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
            inst_probs_tmp = [] 
            inst_preds_tmp = []
            
            for class_idx in range(n_classes):
                if class_idx not in index_label:
                    inst_score[:, class_idx] = [-1] * len(inst_label)
                    continue
                
                inst_score_one_class = inst_score[:, class_idx]
                # if len(set(inst_score_one_class))>1:
                inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                            max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                inst_score_one_class = list(inst_score_one_class)
                inst_score[:, class_idx] = inst_score_one_class

                inst_probs[class_idx] += inst_score_one_class
                inst_probs_tmp.append(inst_score_one_class)

                inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])
                
            if inst_preds_tmp:
                inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
            else:
                inst_preds_tmp = [0]*len(inst_label)
            inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
            inst_preds[n_classes] += inst_preds_tmp
            inst_binary_labels[n_classes] += list(label_hot[:,n_classes])
            
            if inst_probs_tmp:
                neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
                # if len(set(neg_score))>1:
                neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
                # neg_score = list(neg_score)
            else:
                neg_score = [0]*len(inst_label)
            
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score
            
        # wsi-level evaluation
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().detach().numpy()
        Y_hat_one_hot = np.zeros(n_classes)
        Y_hat_one_hot[Y_hat.cpu().numpy()]=1

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        all_preds[batch_idx] = Y_hat_one_hot

        error = calculate_error(Y_hat, label)
        train_error += error
    del data
    
    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_inst_loss /= len(loader)  # v15
    train_error /= len(loader)
    
    # =================================================================
    # calculate inst_auc and inst_acc
    all_inst_score = np.concatenate(all_inst_score, axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    inst_accs = []
    for i in range(n_classes+1):
        if len(inst_binary_labels[i])==0:
            continue
        inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
        inst_acc = np.mean(inst_accs)
        print('class {}'.format(str(i)))
        print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))

    # get inst_auc
    inst_aucs = []
    for class_idx in range(n_classes):
        inst_score_sub = inst_probs[class_idx]
        if len(inst_score_sub)==0:
            continue
        fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
        inst_aucs.append(calc_auc(fpr, tpr))
    fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
    inst_aucs.append(calc_auc(fpr, tpr))
    inst_auc = np.nanmean(np.array(inst_aucs))

    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc_score = np.nanmean(np.array(aucs))
    # ================================


    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count, recall, r_correct, r_count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}, recall {recall} = {r_correct}/{r_count}')
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
            writer.add_scalar('train/class_{}_recall'.format(i), recall, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)
        

def validate_pamil(cur, epoch, model, loader, n_classes, args, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader), n_classes))
    print('\n')
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label, _) in enumerate(loader):
            data, label = data.to(device), label.to(device)      

            logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=False)
            Y_prob=Y_prob.squeeze(0)
            inst_label = inst_label[0]
            score = instance_dict['A_raw'].T
            loss_er = instance_dict['loss_er']
            loss_clst = instance_dict['loss_clst']
            index_label = torch.nonzero(label.squeeze(0)).to(device)
            
            loss_cls = loss_fn(Y_prob, label.squeeze().float())
            loss = loss_cls + args.w_er * loss_er + args.w_clst * loss_clst

            val_loss += loss.item()
            
            # =============================
            # inst-level evaluation
            if inst_label != []:
                all_inst_label += inst_label
                inst_score = score.detach().cpu().numpy()

                if args.task == 'camelyon':
                    label_hot = np.array([[label, 1 - label] for label in inst_label])
                else:
                    label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
                inst_probs_tmp = [] 
                inst_preds_tmp = []
                
                for class_idx in range(n_classes):
                    if class_idx not in index_label:
                        inst_score[:, class_idx] = [-1] * len(inst_label)
                        continue
                    
                    inst_score_one_class = inst_score[:, class_idx]
                    # if len(set(inst_score_one_class))>1:
                    inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                                max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                    inst_score_one_class = list(inst_score_one_class)
                    inst_score[:, class_idx] = inst_score_one_class

                    inst_probs[class_idx] += inst_score_one_class
                    inst_probs_tmp.append(inst_score_one_class)

                    inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                    inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                    inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])
                    
                if inst_preds_tmp:
                    inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
                else:
                    inst_preds_tmp = [0]*len(inst_label)
                inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
                inst_preds[n_classes] += inst_preds_tmp
                inst_binary_labels[n_classes] += list(label_hot[:,n_classes])
                
                if inst_probs_tmp:
                    neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
                    # if len(set(neg_score))>1:
                    neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
                    # neg_score = list(neg_score)
                else:
                    neg_score = [0]*len(inst_label)
                
                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
                
            # wsi-level evaluation
            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().detach().numpy()
            Y_hat_one_hot = np.zeros(n_classes)
            Y_hat_one_hot[Y_hat.cpu().numpy()]=1

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            all_preds[batch_idx] = Y_hat_one_hot

            error = calculate_error(Y_hat, label)
            val_error += error
            # ================================================================
    val_error /= len(loader)
    val_loss /= len(loader)
    
    # =================================================================
    # calculate inst_auc and inst_acc
    all_inst_score = np.concatenate(all_inst_score, axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    inst_accs = []
    for i in range(n_classes+1):
        if len(inst_binary_labels[i])==0:
            continue
        inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
        inst_acc = np.mean(inst_accs)
        print('class {}'.format(str(i)))
        print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))

    # get inst_auc
    inst_aucs = []
    for class_idx in range(n_classes):
        inst_score_sub = inst_probs[class_idx]
        if len(inst_score_sub)==0:
            continue
        fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
        inst_aucs.append(calc_auc(fpr, tpr))
    fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
    inst_aucs.append(calc_auc(fpr, tpr))
    inst_auc = np.nanmean(np.array(inst_aucs))

    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc_score = np.nanmean(np.array(aucs))
    # ================================


    # return the metric 
    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.format(val_loss, val_error, auc_score, inst_auc))
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc_score, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)


    for i in range(n_classes):
        acc, correct, count, recall, r_correct, r_count = acc_logger.get_summary(i)
        print(f'class {i}: acc {acc}, correct {correct}/{count}, recall {recall} = {r_correct}/{r_count}')
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
            writer.add_scalar('train/class_{}_recall'.format(i), recall, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, args):
    
    n_classes = args.n_classes
    model_type = args.model_type
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_silde_ids = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader), n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, cors, inst_label, _) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        inst_label = inst_label[0]
        index_label = torch.nonzero(label.squeeze(0)).to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, instance_dict = model(data, inst_pred=False)
            Y_prob=Y_prob.squeeze(0)
        if model_type in ['clam_sb', 'clam_mb', 'PAMIL']:
            score = instance_dict['A_raw'].T
            # score = F.softmax(score,dim=1)
        elif model_type in ['wsod']:
            score = logits
            
            
        # inst-level evaluation
        if inst_label != []:
            all_inst_label += inst_label
            inst_score = score.detach().cpu().numpy()

            if args.task == 'camelyon':
                label_hot = np.array([[label, 1 - label] for label in inst_label])
            else:
                label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
            inst_probs_tmp = [] 
            inst_preds_tmp = []
            
            for class_idx in range(n_classes):
                if class_idx not in index_label:
                    inst_score[:, class_idx] = [-1] * len(inst_label)
                    continue
                
                inst_score_one_class = inst_score[:, class_idx]
                # if len(set(inst_score_one_class))>1:
                inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                            max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                inst_score_one_class = list(inst_score_one_class)
                inst_score[:, class_idx] = inst_score_one_class

                inst_probs[class_idx] += inst_score_one_class
                inst_probs_tmp.append(inst_score_one_class)

                inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])
                
            if inst_preds_tmp:
                inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
            else:
                inst_preds_tmp = [0]*len(inst_label)
            inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
            inst_preds[n_classes] += inst_preds_tmp
            inst_binary_labels[n_classes] += list(label_hot[:,n_classes])
            
            if inst_probs_tmp:
                neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
                # if len(set(neg_score))>1:
                neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
                # neg_score = list(neg_score)
            else:
                neg_score = [0]*len(inst_label)
            
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score
            
        # wsi-level evaluation
        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().detach().numpy()
        Y_hat_one_hot = np.zeros(n_classes)
        Y_hat_one_hot[Y_hat.cpu().numpy()]=1

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        all_preds[batch_idx] = Y_hat_one_hot
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                           'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    del data

    test_error /= len(loader)
    
    # calculate inst_auc and inst_acc
    
    all_inst_score = np.concatenate(all_inst_score, axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    inst_accs = []
    for i in range(n_classes+1):
        if len(inst_binary_labels[i])==0:
            continue
        inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
        inst_acc = np.mean(inst_accs)
        print('class {}'.format(str(i)))
        print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))
    
    # get inst_auc
    inst_aucs = []
    for class_idx in range(n_classes):
        inst_score_sub = inst_probs[class_idx]
        if len(inst_score_sub)==0:
            continue
        fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
        inst_aucs.append(calc_auc(fpr, tpr))
    fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
    inst_aucs.append(calc_auc(fpr, tpr))
    inst_auc = np.nanmean(np.array(inst_aucs))


    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc_score = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc_score, inst_auc, acc_logger


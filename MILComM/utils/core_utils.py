import numpy as np
import pandas as pd
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, ADD_MIL
from models.model_dsmil import DS_MIL
from models.model_transmil import TransMIL
from models.model_dtfd import DTFD_MIL_tier1, DTFD_MIL_tier2
from models.model_nicwss import NICWSS
# from models.model_addmil import AdditiveClassifier, DefaultAttentionModule, DefaultMILGraph
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
import utils.wsi_seam_utils as visualization

# torch.multiprocessing.set_start_method('spawn', force=True)

def cal_pos_acc(pos_label,pos_score,inst_rate):
    df_score = pd.DataFrame([pos_label,pos_score]).T
    df_score = df_score.sort_values(by=1)
    df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
    total_num_pos = df_score_top.shape[0]
    if total_num_pos:
        return df_score_top[0].sum()/total_num_pos
    else:
        return float('nan')

def cal_neg_acc(neg_label, neg_score, inst_rate):
    df_score = pd.DataFrame([neg_label,neg_score]).T
    df_score = df_score.sort_values(by=1)
    df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]
    total_num_neg = df_score_down.shape[0]
    if total_num_neg:
        return df_score_down[0].sum()/total_num_neg
    else:
        return float('nan')
    
def instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label):
    # for binary
    if n_classes == 1:
        label_mid = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
        label_hot = np.column_stack((1-label_mid, label_mid))
    else:
        label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
    inst_probs_tmp = [] 
    inst_preds_tmp = []

    inst_score = score.detach().cpu().numpy() # Nxcls 不含正常

    # max-min standard all_inst_score and all_inst_score_pos
    for class_idx in range(n_classes):

        if class_idx not in index_label:
            inst_score[:,class_idx] = [0]*len(inst_label)
            continue

        inst_score_one_class = inst_score[:,class_idx]

        inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                    max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))

        inst_probs_tmp.append(inst_score_one_class)    

        inst_score[:,class_idx] = inst_score_one_class

        inst_probs[class_idx] += inst_score_one_class

        inst_preds[class_idx] += [0 if i<0.5 else 1 for i in inst_score_one_class]
        inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

        inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])

        pos_accs_each_wsi[class_idx].append(cal_pos_acc(label_hot[:,class_idx], inst_score_one_class, inst_rate))

    if inst_preds_tmp:
        inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0) # Nx1
    else:
        inst_preds_tmp = [0]*len(inst_label)
    inst_preds_neg = [1 if i==0 else 0 for i in inst_preds_tmp]
    inst_preds[n_classes] += inst_preds_neg
    inst_binary_labels[n_classes]+=list(label_hot[:,n_classes])

    if inst_probs_tmp:
        neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #n类平均，越低越是neg
        neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
    else:
        neg_score = [0]*len(inst_label)

    neg_accs_each_wsi.append(cal_neg_acc(label_hot[:,n_classes], neg_score, inst_rate))


    return inst_score, neg_score       
    
def instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, all_inst_score, all_inst_score_neg,
                    pos_accs_each_wsi, neg_accs_each_wsi, inst_rate):
    
    all_inst_score = np.concatenate(all_inst_score, axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    inst_accs = []
    
    for i in range(n_classes+1):
        if len(inst_binary_labels[i])==0:
            continue
        inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
        print('class {}'.format(str(i)))
        print(classification_report(inst_binary_labels[i], inst_preds[i], zero_division=1))
        
    inst_acc = np.mean(inst_accs)

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
    
    # calculate pos acc and neg acc
    pos_accs = [sum(i)/len(i) if len(i) else 0 for i in pos_accs_each_wsi]
    pos_acc = np.nanmean(np.array(pos_accs))
    pos_acc_str = "seleted pos %f acc: "%(inst_rate)
    for i in range(n_classes):
        if i<n_classes-1:
            pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])+', '
        else:
            pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])
    print(pos_acc_str)
    neg_acc = sum(neg_accs_each_wsi)/len(neg_accs_each_wsi)
    print("seleted neg %f acc: %f" % (inst_rate, neg_acc))
    
    return inst_auc, inst_acc, pos_accs, neg_acc

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        # Y_hat输出的是值的形式而非向量，Y也要值的形式
        Y = np.atleast_1d(Y.squeeze().cpu().numpy())
        Y_hat = Y_hat.squeeze().cpu().numpy()
        for true in Y:
            self.data[true]["count"] += 1
            if true in Y_hat:
                self.data[true]["correct"] += 1
    
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
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

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

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if isinstance(model, list):
            model_tier1, model_tier2 = model
            state_dicts = {"model_tier1": model_tier1.state_dict(), "model_tier2": model_tier2.state_dict()}
            torch.save(state_dicts, ckpt_name)
        else:
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

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'bce':
        if args.model_type == 'ds_mil':
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.functional.binary_cross_entropy
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'fea_dim': args.fea_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    elif args.model_type == 'add_mil':
        model = ADD_MIL(**model_dict)
    elif args.model_type == 'trans_mil':
        model = TransMIL(**model_dict)
    elif args.model_type == 'ds_mil':
        model = DS_MIL(**model_dict, dropout_node=0, non_linearity=1)
        state_dict_weights = torch.load('./init.pth')
        try:
            milnet.load_state_dict(state_dict_weights, strict=False)
        except:
            del state_dict_weights['b_classifier.v.1.weight']
            del state_dict_weights['b_classifier.v.1.bias']
            model.load_state_dict(state_dict_weights, strict=False)
    elif args.model_type == 'dtfd_mil':
        model_tier1 = DTFD_MIL_tier1(**model_dict)
        model_tier2 = DTFD_MIL_tier2(**model_dict)
        model = [model_tier1, model_tier2]
    elif args.model_type in ['nic', 'nicwss']:
        model_dict.update({'only_cam': args.only_cam})
        model = NICWSS(**model_dict)
    else:
        raise NotImplementedError
    
    if args.model_type == 'dtfd_mil':
        for model_temp in model:
            model_temp.relocate()
    else:
        model.relocate()
    print('Done!')
    if args.model_type == 'dtfd_mil':
        for model_temp in model:
            print_network(model_temp)
    else:
        print_network(model)

    print('\nInit optimizer ...', end=' ')
    if args.model_type == 'dtfd_mil':
        optimizer1 = get_optim(model_tier1, args)
        optimizer2 = get_optim(model_tier2, args)
        optimizer = [optimizer1, optimizer2]
    else:
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

    
    _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args)
    
    for epoch in range(args.max_epochs):
        if args.model_type in ['clam_sb', 'clam_mb']:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn,
                            args.inst_rate, not args.no_inst_cluster)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                                 early_stopping, writer, loss_fn, args.results_dir, args.inst_rate, not args.no_inst_cluster)
        elif args.model_type in ['add_mil']:
            train_loop_addmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.inst_rate)
            stop = validate_addmil(cur, epoch, model, val_loader, args.n_classes, 
                                   early_stopping, writer, loss_fn, args.results_dir, args.inst_rate)
        elif args.model_type in ['ds_mil']:
            train_loop_dsmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.inst_rate)
            stop = validate_dsmil(cur, epoch, model, val_loader, args.n_classes, 
                                  early_stopping, writer, loss_fn, args.results_dir, args.inst_rate)
        elif args.model_type in ['trans_mil']:
            train_loop_transmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.inst_rate)
            stop = validate_transmil(cur, epoch, model, val_loader, args.n_classes, 
                                     early_stopping, writer, loss_fn, args.results_dir, args.inst_rate)
        elif args.model_type in ['dtfd_mil']:
            train_loop_dtfdmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.inst_rate)
            stop = validate_dtfdmil(cur, epoch, model, val_loader, args.n_classes, 
                                    early_stopping, writer, loss_fn, args.results_dir, args.inst_rate)
        elif args.model_type in ['nic', 'nicwss']:
            train_loop_nicwss(epoch, model, train_loader, optimizer, args.n_classes, args.only_cam, args.b_rv, args.w_cls, args.w_er, args.w_ce,
                    writer, args.inst_rate)
            stop = validate_nicwss(cur, epoch, model, val_loader, args.n_classes, early_stopping, args.only_cam, args.b_rv, args.w_cls, args.w_er, args.w_ce,
                                   writer, args.inst_rate, args.task, args.results_dir)
        else:
            raise NotImplementedError
        
        if stop: 
            break

    if args.early_stopping:
        if args.model_type == 'dtfd_mil':
            loaded_state_dicts = torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
            model[0].load_state_dict(loaded_state_dicts["model_tier1"])
            model[1].load_state_dict(loaded_state_dicts["model_tier2"])
        else:
            model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, test_iauc, acc_logger = summary(model, test_loader, args)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_iauc', val_iauc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_iauc', test_iauc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error , test_iauc, val_iauc

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None,
                    inst_rate=0.1, instance_eval= True):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []
    
    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data = data.to(device)
        label = label.to(device)
        index_label = torch.nonzero(label.squeeze()).to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
        
        total_loss = 0
        total_loss_value = 0
        
        logits, Y_prob, Y_hat, score, instance_dict = model(data, label = label, instance_eval = instance_eval)
            
        acc_logger.log(Y_hat, index_label)

        loss = loss_fn(Y_prob.squeeze().float(), label.squeeze().float())

        loss_value = loss.item()
        
        if instance_eval:
        
            instance_loss = instance_dict['instance_loss']
            if instance_loss!=0:
                inst_count+=1
                instance_loss_value = instance_loss.item()
                train_inst_loss += instance_loss_value

            total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 
            
            inst_preds_clam = instance_dict['inst_preds']
            inst_labels_clam = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds_clam, inst_labels_clam)
        else:
            total_loss = loss

        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        score = score.T
        inst_label = inst_label[0]
        
        if inst_label!=[]:
            inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score
        
        

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
        
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
            
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def train_loop_dsmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data = data.to(device)
        label = label.to(device)
        index_label = torch.nonzero(label.squeeze()).to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
        
        logits, Y_prob, Y_hat, score, patch_logits = model(data)
        
        max_logits, _ = torch.max(patch_logits, 0)
        
        bag_loss = loss_fn(logits.squeeze().float(), label.squeeze().float())
        max_loss = loss_fn(max_logits.squeeze().float(), label.squeeze().float())
        
        loss = 0.5*bag_loss + 0.5*max_loss
        
        acc_logger.log(Y_hat, index_label)

        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        score = score.T
        inst_label = inst_label[0]
        
        if inst_label!=[]:
            inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
    

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)
        
def train_loop_transmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data = data.to(device)
        label = label.to(device)
        index_label = torch.nonzero(label.squeeze()).to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
        
        logits, Y_prob, Y_hat, score, _ = model(data)
        
        acc_logger.log(Y_hat, index_label)
        
        loss = loss_fn(Y_prob.squeeze().float(), label.squeeze().float())
        
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        score = score.T
        inst_label = inst_label[0]
        
        if inst_label!=[]:
            inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)  
        
def train_loop_addmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, inst_rate=0.1):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data = data.to(device)
        label = label.to(device)
        index_label = torch.nonzero(label.squeeze()).to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label

        logits, Y_prob, Y_hat, score, _ = model(data)
        
        acc_logger.log(Y_hat, index_label)
        
        loss = loss_fn(Y_prob.squeeze().float(), label.squeeze().float())
        
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        score = score.T
        inst_label = inst_label[0]
        
        if inst_label!=[]:
            inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)  
        
def train_loop_dtfdmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_tier1, model_tier2 = model
    
    model_tier1.train()
    model_tier2.train()
    
    optimizer1, optimizer2 = optimizer
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data = data.to(device)
        label = label.to(device)
        index_label = torch.nonzero(label.squeeze()).to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
            
        slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
        
        slide_sub_preds = torch.sigmoid(slide_sub_preds)
        
        loss1 = loss_fn(slide_sub_preds.squeeze().float(), slide_sub_labels.squeeze().float())
        
        optimizer1.zero_grad()
        torch.nn.utils.clip_grad_norm_(model_tier1.parameters(), 5)
        
        ## optimization for the second tier
        logits = model_tier2(slide_pseudo_feat)
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        
        Y_prob = torch.sigmoid(logits)

        loss2 = loss_fn(Y_prob.squeeze().float().T, label.squeeze().float())

        optimizer2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()

        torch.nn.utils.clip_grad_norm_(model_tier2.parameters(), 5)

        optimizer1.step()
        optimizer2.step()

        acc_logger.log(Y_hat, index_label)
        
        loss = loss1 + loss2
        
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        score = score.T
        
        inst_label = inst_label[0]
        
        if inst_label!=[]:
            inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                                      pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)  

def train_loop_nicwss(epoch, model, loader, optimizer, n_classes, for_cam=False, b_rv=0.5, w_cls=1.0, w_er=1.0, w_ce=1.0,
                      writer = None, inst_rate = 0.01): 
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_cls_loss = 0.
    train_er_loss = 0.
    train_ce_loss = 0.
    train_error = 0.

    all_inst_score = []    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        label = label.to(device)
        label_bn = label.unsqueeze(2).unsqueeze(3)
        if n_classes == 1:  # for binary
            index_label = torch.nonzero(label.squeeze(0)).to(device)
        else:
            index_label = torch.nonzero(label.squeeze()).to(device)
        
        # data = data[0]
        data = data.to(device)
        
        if for_cam:  # for nic
            cam = model(data, masked=True, train=True)
            label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
            loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
            cam_raw = visualization.max_norm(cam)
            cam = visualization.max_norm(cam) * label_bn
            Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
            
            # total loss
            loss = loss_cls1
            
        else:  # for nicwss
            cam, cam_rv, valid_pixels = model(data, masked=True, train=True)
            label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
            label_pred_cam_rv = F.adaptive_avg_pool2d(cam_rv, (1, 1))
            loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
            loss_cls2 = F.multilabel_soft_margin_loss(label_pred_cam_rv, label_bn)
            cam_raw = visualization.max_norm(cam)
            cam = visualization.max_norm(cam) * label_bn
            cam_rv = visualization.max_norm(cam_rv) * label_bn
            # Equivarinat loss
            loss_er = torch.mean(torch.abs(cam - cam_rv))
            
            cond_entropy = -((valid_pixels * (cam_rv * torch.log(cam + 1e-10))).sum(1))
            cond_entropy = cond_entropy.sum(dim=(1, 2))
            cond_entropy /= valid_pixels.squeeze(1).sum(dim=(1, 2))
            cond_entropy = cond_entropy.mean(0)
            
            Y_prob = (label_pred_cam_or * b_rv + label_pred_cam_rv * (1-b_rv)).squeeze(3).squeeze(2)
            
            # total loss
            loss_cls = loss_cls1 * b_rv + loss_cls2 * (1-b_rv)
            loss = w_cls * loss_cls + w_er * loss_er + w_ce * cond_entropy
            train_cls_loss += loss_cls.item()
            train_er_loss += loss_er.item()
            train_ce_loss += cond_entropy.item()

        logits_sigmoid = Y_prob.detach().cpu().numpy()
        Y_hat = torch.from_numpy(np.where(logits_sigmoid>0.5)[0]).cuda().unsqueeze(0)  # for binary
        
        acc_logger.log(Y_hat, index_label)

        error = calculate_error(Y_hat, label)  # for binary
        train_error += error
        
        loss_value = loss.item()
        
        train_loss += loss_value

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        # instance calculation part
        inst_label = inst_label[0]
        if inst_label!=[]:  # instance score
            mask = cors[1]
            f_h, f_w = np.where(mask==1)
            cam_n = cam_raw[0,:,...] # for binary
            inst_score = cam_n[:, f_h, f_w].T
            inst_score, neg_score = instance_analysis(n_classes, inst_score, inst_label, inst_probs, inst_preds, inst_binary_labels, 
                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate, index_label)
        
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_cls_loss /= len(loader)
    train_er_loss /= len(loader)
    train_ce_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
    
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'\
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)
        if not for_cam:
            writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
            writer.add_scalar('train/er_loss', train_er_loss, epoch)
            writer.add_scalar('train/ce_loss', train_ce_loss, epoch)


def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, 
                  results_dir = None, inst_rate=0.1, instance_eval=True):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    val_inst_loss = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))
    
    with torch.no_grad():
        print('\n')
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):

            data = data.to(device)
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)

            if inst_label!=[]:
                all_inst_label += inst_label

            total_loss = 0
            total_loss_value = 0

            logits, Y_prob, Y_hat, score, instance_dict = model(data, label=label, instance_eval=instance_eval)

            acc_logger.log(Y_hat, index_label)

            loss = loss_fn(Y_prob.squeeze(), label.squeeze().float())

            val_loss += loss.item()
            
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            if instance_eval:

                instance_loss = instance_dict['instance_loss']

                if instance_loss!=0:
                    inst_count+=1
                    instance_loss_value = instance_loss.item()
                    val_inst_loss += instance_loss_value

                inst_preds_clam = instance_dict['inst_preds']
                inst_labels_clam = instance_dict['inst_labels']
                inst_logger.log_batch(inst_preds_clam, inst_labels_clam)
            
            error = calculate_error(Y_hat, label)
            val_error += error
            
            score = score.T
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
        
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))
            
    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, val_loss, val_error, auc, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_dsmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, 
                  loss_fn = None, results_dir=None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)
            
            if inst_label!=[]:
                all_inst_label += inst_label

            logits, Y_prob, Y_hat, score, patch_logits = model(data)
            max_logits, _ = torch.max(patch_logits, 0)    

            bag_loss = loss_fn(logits.squeeze().float(), label.squeeze().float())
            max_loss = loss_fn(max_logits.squeeze().float(), label.squeeze().float())

            loss = 0.5*bag_loss + 0.5*max_loss

            acc_logger.log(Y_hat, index_label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
    
            score = score.T
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))
    
    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, val_loss, val_error, auc, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_transmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                      results_dir=None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
            data = data.to(device)
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)

            if inst_label!=[]:
                all_inst_label += inst_label

            logits, Y_prob, Y_hat, score, _ = model(data)

            acc_logger.log(Y_hat, index_label)
            
            loss = loss_fn(Y_prob.squeeze(), label.squeeze().float())
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            score = score.T
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
        
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))
            
    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, val_loss, val_error, auc, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_addmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    inst_count = 0
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
            data = data.to(device)
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)

            if inst_label!=[]:
                all_inst_label += inst_label

            logits, Y_prob, Y_hat, score, _ = model(data)
            
            acc_logger.log(Y_hat, index_label)
            
            loss = loss_fn(Y_prob.squeeze(), label.squeeze().float())
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            score = score.T
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
        
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))
            
    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, val_loss, val_error, auc, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_dtfdmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None, inst_rate=0.1):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_tier1, model_tier2 = model
    
    model_tier1.eval()
    model_tier2.eval()
    
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    val_loss = 0.
    val_error = 0.
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
            data = data.to(device)
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)

            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
            
            slide_sub_preds = torch.sigmoid(slide_sub_preds)
        
            loss1 = loss_fn(slide_sub_preds.squeeze().float(), slide_sub_labels.squeeze().float())

            ## optimization for the second tier
            logits = model_tier2(slide_pseudo_feat)

            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            
            Y_prob = torch.sigmoid(logits)
            
            loss2 = loss_fn(Y_prob.squeeze().float().T, label.squeeze().float())

            acc_logger.log(Y_hat, index_label)

            loss = loss1 + loss2
            
            loss_value = loss.item()
        
            val_loss += loss_value

            error = calculate_error(Y_hat, label)
            val_error += error

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            score = score.T
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                inst_score, neg_score = instance_analysis(n_classes, score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)
        
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))
            
    print('Epoch: {}, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'
          .format(epoch, val_loss, val_error, auc, inst_auc, inst_acc))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def validate_nicwss(cur, epoch, model, loader, n_classes, early_stopping = None, for_cam=False, b_rv=0.5, w_cls=1.0, w_er=1.0, w_ce=1.0,
                    writer = None, inst_rate=0.01, task='camelyon', results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
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

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
            data, label = data.to(device), label.to(device)
            label_bn = label.unsqueeze(2).unsqueeze(3)
            
            if n_classes == 1:  # for binary
                index_label = torch.nonzero(label.squeeze(0)).to(device)
            else:
                index_label = torch.nonzero(label.squeeze()).to(device)            
            if for_cam:
                cam = model(data, masked=True, train=False)
                label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
                loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
                cam_raw = visualization.max_norm(cam)
                cam = visualization.max_norm(cam)  * label_bn
                Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
                # Classification loss
                loss = loss_cls1
            else:
                cam, cam_rv = model(data, masked=True, train=False)
                cam_raw = visualization.max_norm(cam)
                
                label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
                label_pred_cam_rv = F.adaptive_avg_pool2d(cam_rv, (1, 1))

                loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
                loss_cls2 = F.multilabel_soft_margin_loss(label_pred_cam_rv, label_bn)
                
                cam = visualization.max_norm(cam) * label_bn  # v8 changed
                cam_rv = visualization.max_norm(cam_rv) * label_bn
                
                Y_prob = (label_pred_cam_or * b_rv + label_pred_cam_rv * (1-b_rv)).squeeze(3).squeeze(2)
                # Classification loss
                loss_cls = loss_cls1 * b_rv + loss_cls2 * (1-b_rv)
                loss = loss_cls
            
            logits_sigmoid = Y_prob.detach().cpu().numpy()
            Y_hat = torch.from_numpy(np.where(logits_sigmoid>0.5)[0]).cuda().unsqueeze(0)  # for binary
            

            acc_logger.log(Y_hat, index_label) 

            error = calculate_error(Y_hat, label)  # for binary

            val_error += error

            loss_value = loss.item()

            val_loss += loss_value

            # instance calculation part
            inst_label = inst_label[0]
            
            if inst_label!=[]:
                mask = cors[1]
                f_h, f_w = np.where(mask==1)
                # cam_n = cam_raw[0,:,...].squeeze(0)
                cam_n = cam_raw[0,:,...]
                inst_score = cam_n[:, f_h, f_w].T
                inst_score, neg_score = instance_analysis(n_classes, inst_score, inst_label, inst_probs, inst_preds, 
                                                        inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                        inst_rate, index_label)

                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
                
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                        all_inst_score, all_inst_score_neg, 
                                                        pos_accs_each_wsi, neg_accs_each_wsi, inst_rate)


    # for binary
    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc = np.nanmean(np.array(aucs))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)
        

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.format(val_loss, val_error, auc, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    
    if args.model_type == 'dtfd_mil':
        model_tier1, model_tier2 = model
        model_tier1.eval()
        model_tier2.eval()
    else:
        model.eval()
        
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(args.n_classes+1)]
    inst_preds = [[] for _ in range(args.n_classes+1)]
    inst_binary_labels = [[] for _ in range(args.n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(args.n_classes)]
    neg_accs_each_wsi = []

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros((len(loader), args.n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):

        data, label = data.to(device), label.to(device)
        
        if args.n_classes == 1:  # for binary
            index_label = torch.nonzero(label.squeeze(0)).to(device)
        else:
            index_label = torch.nonzero(label.squeeze()).to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        
        if args.model_type in ['nic', 'nicwss']:
            label_bn = label.unsqueeze(2).unsqueeze(3)
        
  
        with torch.no_grad():

            if args.model_type == 'dtfd_mil':
                slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
                logits = model_tier2(slide_pseudo_feat)
                Y_hat = torch.topk(logits, 1, dim = 1)[1]
                Y_prob = torch.sigmoid(logits)
                score = score.T
            elif args.model_type in ['nic', 'nicwss']:
                if args.only_cam:
                    cam = model(data, masked=True, train=False)
                    label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
                    
                    Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
                    cam_raw = visualization.max_norm(cam)
                    cam = visualization.max_norm(cam)  * label_bn
                else:
                    cam, cam_rv = model(data, masked=True, train=False)
                    label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
                    label_pred_cam_rv = F.adaptive_avg_pool2d(cam_rv, (1, 1))
                    Y_prob = (label_pred_cam_or * args.b_rv + label_pred_cam_rv * (1-args.b_rv)).squeeze(3).squeeze(2)
                    
                    cam_raw = visualization.max_norm(cam)
                    cam = visualization.max_norm(cam)
                    cam_rv = visualization.max_norm(cam_rv)
                logits_sigmoid = Y_prob.cpu().numpy()
                Y_hat = torch.from_numpy(np.where(logits_sigmoid>0.5)[0]).cuda().unsqueeze(0)  # for binary
                
            else:
                logits, Y_prob, Y_hat, score, _ = model(data)

        inst_label = inst_label[0]
        
        if inst_label!=[]:
            if args.model_type in ['nic', 'nicwss']:
                mask = cors[1]
                f_h, f_w = np.where(mask==1)
                # cam_n = cam_raw[0,:,...].squeeze()
                cam_n = cam_raw[0,:,...]  # for binary
                inst_score = cam_n[:, f_h, f_w].T
                inst_score, neg_score = instance_analysis(args.n_classes, inst_score, inst_label, inst_probs, inst_preds, 
                                                          inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                          args.inst_rate, index_label)
        
            else:
                inst_score, neg_score = instance_analysis(args.n_classes, score, inst_label, inst_probs, inst_preds, 
                                                        inst_binary_labels, pos_accs_each_wsi, neg_accs_each_wsi, 
                                                        args.inst_rate, index_label)

            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

        acc_logger.log(Y_hat, index_label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                           'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label)  # for binary
        test_error += error

    test_error /= len(loader)
    
    inst_auc, inst_acc, pos_accs, neg_acc = instance_report(args.n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, args.inst_rate)
    
    # calculate inst_auc and inst_acc   
    
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg]  # 转化为是normal类的概率 越高越好

    aucs = []
    binary_labels = all_labels
    for class_idx in range(args.n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, inst_auc, acc_logger

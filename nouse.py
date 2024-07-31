import numpy as np
import torch
from utils.utils import *
import os
import pandas as pd
from datasets.dataset_generic import save_splits
from models.model_nicwss import NICWSS
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc

import utils.wsi_seam_utils as visualization

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
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

    if args.bag_loss == 'bce':
        loss_fn = nn.functional.binary_cross_entropy
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'fea_dim': args.fea_dim}
    # model_dict.update({"size_arg": args.model_size})
            
    if args.model_type in ['nicwss']:
        model = NICWSS(**model_dict)
    else:
        raise NotImplementedError

    
    model.relocate()
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
        early_stopping = EarlyStopping(patience = 40, stop_epoch=100, verbose = True)
    else:
        early_stopping = None
    print('Done!')
    
    #  _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args.n_classes, args.task)
    
    ref_start = False
    
    for epoch in range(args.max_epochs):
        if args.model_type in ['nicwss']:
            train_loop_nicwss(epoch, model, train_loader, optimizer, args.n_classes, args.only_cam, args.b_rv, args.w_cls, args.w_er, args.w_ce,
                              writer, args.inst_rate)
            stop = validate_nicwss(cur, epoch, model, val_loader, args.n_classes, early_stopping, args.only_cam, args.b_rv, args.w_cls, args.w_er, args.w_ce,
                                   writer, args.inst_rate, args.task, args.results_dir)
        else:
            raise NotImplementedError
        
        if (stop and not ref_start and args.inst_refinement) or (epoch > args.ref_start_epoch and args.inst_refinement):
            ref_start = True
            early_stopping = EarlyStopping(patience = 10, stop_epoch=100, verbose = True)
        elif stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args.n_classes, args.task)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, test_iauc, acc_logger = summary(model, test_loader, args.n_classes, args.task)
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

    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        label = label.to(device)
        label_bn = torch.zeros(n_classes)
        label_bn[label.long()] = 1
        label_bn = label_bn.to(device)
        
        label_bn = label_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if inst_label!=[] and sum(inst_label)!=0:  # instance label
            inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label]  # 单标签多分类只考虑 normal vs cancer
            all_inst_label += inst_label
        
        # data = data[0]
        data = data.to(device)
        mask = cors[1]
        
        cam = model(data, masked=True, train=True)

        label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))
        
        loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
        
        # Equivarinat loss
        cam_raw = visualization.max_norm(cam)
        cam = visualization.max_norm(cam) * label_bn
        

        Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
        Y_hat = torch.argmax(Y_prob)
        # Total loss
        loss = loss_cls1
        
        acc_logger.log(Y_hat, label)

        error = calculate_error(Y_hat, label)
        train_error += error
        
        loss_value = loss.item()
        
        train_loss += loss_value

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        # instance calculation part
        
        if inst_label!=[] and sum(inst_label)!=0:  # instance score
            f_h, f_w = np.where(mask==1)
            cam_n = cam_raw[0,label,...].squeeze(0).data.cpu().numpy()
            inst_score = list(cam_n[f_h, f_w])
            inst_pred = [1 if i>0.5 else 0 for i in inst_score]  # use threshold

            all_inst_score += inst_score
            all_inst_pred += inst_pred

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_cls_loss /= len(loader)
    train_er_loss /= len(loader)
    train_ce_loss /= len(loader)
    train_error /= len(loader)
    
    # excluding -1 
    all_inst_label_sub = np.array(all_inst_label)
    all_inst_score_sub = np.array(all_inst_score)
    all_inst_pred_sub = np.array(all_inst_pred)
    
    all_inst_score_sub = all_inst_score_sub[all_inst_label_sub!=-1]
    all_inst_pred_sub = all_inst_pred_sub[all_inst_label_sub!=-1]
    all_inst_label_sub = all_inst_label_sub[all_inst_label_sub!=-1] 

    inst_auc = roc_auc_score(all_inst_label_sub, all_inst_score_sub)
    df_score = pd.DataFrame([all_inst_label_sub, all_inst_score_sub]).T
    
    df_score = df_score.sort_values(by=1)
    df_score[1] = df_score[1].apply(lambda x: 1 if x>0.5 else 0)

    df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
    df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]

    pos_acc = (df_score_top[0].sum()/df_score_top.shape[0])
    neg_acc = (1-df_score_down[0].sum()/df_score_down.shape[0])
    # print("seleted pos all acc: %f" % pos_acc_all)
    print("seleted pos %f acc: %f" % (inst_rate, pos_acc))
    print("seleted neg %f acc: %f" % (inst_rate, neg_acc))
    
    # all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label_sub, all_inst_pred_sub)
    print(classification_report(all_inst_label_sub, all_inst_pred_sub, zero_division=1))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'\
          .format(epoch, train_loss, train_error, inst_auc, inst_acc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/pos_acc', pos_acc, epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)
        if not for_cam:
            writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
            writer.add_scalar('train/er_loss', train_er_loss, epoch)
            writer.add_scalar('train/ce_loss', train_ce_loss, epoch)
        

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
    
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
            label = label.to(device)
            label_bn = torch.zeros(n_classes)
            label_bn[label.long()] = 1
            label_bn = label_bn.to(device)

            label_bn = label_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            if inst_label!=[] and sum(inst_label)!=0:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                all_inst_label += inst_label

            data = data.to(device)
            mask = cors[1]

            cam = model(data, masked=True, train=False)

            label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))

            loss_cls1 = F.multilabel_soft_margin_loss(label_pred_cam_or, label_bn)
            
            cam_raw = visualization.max_norm(cam)
            cam = visualization.max_norm(cam)  * label_bn
            
            Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
            Y_hat = torch.argmax(Y_prob)
            # Classification loss
            loss = loss_cls1

            acc_logger.log(Y_hat, label) 

            error = calculate_error(Y_hat, label)

            val_error += error

            loss_value = loss.item()

            val_loss += loss_value

            # instance calculation part
            if inst_label!=[] and sum(inst_label)!=0:
                f_h, f_w = np.where(mask==1)
                cam_n = cam_raw[0,label,...].squeeze(0).data.cpu().numpy()
                inst_score = list(cam_n[f_h, f_w])
                inst_pred = [1 if i>0.5 else 0 for i in inst_score]

                all_inst_score += inst_score
                all_inst_pred += inst_pred
            
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    # excluding -1 
    all_inst_label_sub = np.array(all_inst_label)
    all_inst_score_sub = np.array(all_inst_score)
    all_inst_pred_sub = np.array(all_inst_pred)
    
    all_inst_score_sub = all_inst_score_sub[all_inst_label_sub!=-1]
    all_inst_pred_sub = all_inst_pred_sub[all_inst_label_sub!=-1]
    all_inst_label_sub = all_inst_label_sub[all_inst_label_sub!=-1] 
    
    inst_auc = roc_auc_score(all_inst_label_sub, all_inst_score_sub)

    inst_acc = accuracy_score(all_inst_label_sub, all_inst_pred_sub)
    print(classification_report(all_inst_label_sub, all_inst_pred_sub, zero_division=1))


    if task == 'camelyon':
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        if n_classes == 2:
            binary_labels = np.hstack((1-binary_labels, binary_labels))
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

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

def summary(model, loader, n_classes, task):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        label = label.to(device)
        label_bn = torch.zeros(n_classes)
        label_bn[label.long()] = 1
        label_bn = label_bn.to(device)

        label_bn = label_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if inst_label!=[] and sum(inst_label)!=0:
            inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
            all_inst_label += inst_label

        data = data.to(device)
        mask = cors[1]
        
        with torch.no_grad():
            cam = model(data, masked=True, train=False)

        label_pred_cam_or = F.adaptive_avg_pool2d(cam, (1, 1))

        Y_prob = label_pred_cam_or.squeeze(3).squeeze(2)
        Y_hat = torch.argmax(Y_prob)
        
        cam_raw = visualization.max_norm(cam)
        cam = visualization.max_norm(cam)  * label_bn

        acc_logger.log(Y_hat, label) 

        error = calculate_error(Y_hat, label)

        test_error += error

        # instance calculation part

        if inst_label!=[] and sum(inst_label)!=0:
            f_h, f_w = np.where(mask==1)
            cam_n = cam_raw[0, label,...].squeeze(0).data.cpu().numpy()
            inst_score = list(cam_n[f_h, f_w])
            inst_pred = [1 if i>0.5 else 0 for i in inst_score]

            all_inst_score += inst_score
            all_inst_pred += inst_pred
        
        slide_id = slide_ids.iloc[batch_idx]
            
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})


    test_error /= len(loader)

    # excluding -1 
    all_inst_label_sub = np.array(all_inst_label)
    all_inst_score_sub = np.array(all_inst_score)
    all_inst_pred_sub = np.array(all_inst_pred)
    
    all_inst_score_sub = all_inst_score_sub[all_inst_label_sub!=-1]
    all_inst_pred_sub = all_inst_pred_sub[all_inst_label_sub!=-1]
    all_inst_label_sub = all_inst_label_sub[all_inst_label_sub!=-1] 

    inst_auc = roc_auc_score(all_inst_label_sub, all_inst_score_sub)
    inst_acc = accuracy_score(all_inst_label_sub, all_inst_pred_sub)
    print(classification_report(all_inst_label_sub, all_inst_pred_sub, zero_division=1))

    if task == 'camelyon':
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        if n_classes == 2:
            binary_labels = np.hstack((1-binary_labels, binary_labels))
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, inst_auc, acc_logger

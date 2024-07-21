import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, ADD_MIL
from models.model_dsmil import DS_MIL
from models.model_transmil import TransMIL
from models.model_dtfd import DTFD_MIL_tier1, DTFD_MIL_tier2
# from models.model_addmil import AdditiveClassifier, DefaultAttentionModule, DefaultMILGraph
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc

# torch.multiprocessing.set_start_method('spawn', force=True)

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
#         model = DefaultMILGraph(
#         pointer=DefaultAttentionModule(hidden_activation = nn.LeakyReLU(0.2), hidden_dims=[512, 256], input_dims=1024),
#         classifier=AdditiveClassifier(hidden_dims=[512, 256], input_dims=1024, output_dims=args.n_classes)
#     )
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
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.model_type == 'dtfd_mil':
        model_tier1 = DTFD_MIL_tier1(**model_dict)
        model_tier2 = DTFD_MIL_tier2(**model_dict)
        model = [model_tier1, model_tier2]
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    if args.model_type == 'dtfd_mil':
#         model_tier1.relocate()
#         model_tier2.relocate()
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
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn, args.task)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        elif args.model_type in ['add_mil']:
            train_loop_addmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.task)
            stop = validate_addmil(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        elif args.model_type in ['ds_mil']:
            train_loop_dsmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.task)
            stop = validate_dsmil(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        elif args.model_type in ['trans_mil']:
            train_loop_transmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.task)
            stop = validate_transmil(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        elif args.model_type in ['dtfd_mil']:
            train_loop_dtfdmil(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.task)
            stop = validate_dtfdmil(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, args.task)
            stop = validate_loop(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir, args.task)
        
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

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, task='camelyon'):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        logits, Y_prob, Y_hat, score, _ = model(data)
        
        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        
        score = score.T
        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:,0]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/(inst_score.max()-inst_score.min()))
            all_inst_label += inst_label
            all_inst_score += inst_score
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        # if (batch_idx + 1) % 20 == 0:
        #     print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None, task='camelyon'):
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

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
    # for batch_idx, (data, label) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, score, instance_dict = model(data, label=label, instance_eval=True)
    
        
        # loss = loss_fn(logits, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss
        
        # inst_label = [[i%2 for i in range(score.shape[-1])]] ### 临时代码
        
        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        
        score = score.T

        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/(inst_score.max()-inst_score.min()))
            all_inst_label += inst_label
            all_inst_score += inst_score

        acc_logger.log(Y_hat, label)


        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))
    
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def train_loop_dsmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        wsi_label = torch.zeros(n_classes)
        wsi_label[label.long()] = 1
        wsi_label = wsi_label.to(device)
        
        logits, Y_prob, Y_hat, score, patch_logits = model(data)
        max_logits, _ = torch.max(patch_logits, 0)    
        
        bag_loss = loss_fn(logits.view(1, -1), wsi_label.view(1, -1))
        max_loss = loss_fn(max_logits.view(1, -1), wsi_label.view(1, -1))
        
        loss = 0.5*bag_loss + 0.5*max_loss
        
        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        score = score.T
        
        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
                
            elif task == "camelyon":
                
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score
        
        acc_logger.log(Y_hat, label)

        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)
        
def train_loop_transmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        wsi_label = torch.zeros(n_classes)
        wsi_label[label.long()] = 1
        wsi_label = wsi_label.to(device)
        
        logits, Y_prob, Y_hat, score, _ = model(data)
        
        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        score = score.T
        
        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)  
        
def train_loop_addmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, task='camelyon'):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        wsi_label = torch.zeros(n_classes)
        wsi_label[label.long()] = 1
        wsi_label = wsi_label.to(device)

        logits, Y_prob, Y_hat, score, _ = model(data)
        
        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        
        score = score.T

        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif task == "camelyon":
                
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
          
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, inst_acc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc, inst_acc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_acc', inst_acc, epoch)
        
def train_loop_dtfdmil(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    model_tier1, model_tier2 = model
    
    model_tier1.train()
    model_tier2.train()
    
    optimizer1, optimizer2 = optimizer

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        # logits, Y_prob, Y_hat, score, _ = model(data)

        slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
        
        loss1 = loss_fn(slide_sub_preds, slide_sub_labels).mean()
        
        optimizer1.zero_grad()
        torch.nn.utils.clip_grad_norm_(model_tier1.parameters(), 5)
        
        ## optimization for the second tier
        logits = model_tier2(slide_pseudo_feat)

        loss2 = loss_fn(logits, label).mean()

        optimizer2.zero_grad()

        loss1.backward(retain_graph=True)
        loss2.backward()

        torch.nn.utils.clip_grad_norm_(model_tier2.parameters(), 5)

        optimizer1.step()
        optimizer2.step()
        
        Y_hat = torch.topk(logits, 1, dim = 1)[1]

        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        score = score.T
        
        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score
        
        acc_logger.log(Y_hat, label)
        loss = loss1 + loss2
        loss_value = loss.item()
        
        train_loss += loss_value
        
        error = calculate_error(Y_hat, label)
        train_error += error

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def validate_loop(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, 
                  loss_fn = None, results_dir=None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, score, _ = model(data)
            
            inst_label = inst_label[0]
            inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
            
            score = score.T
            if inst_label!=[]:
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
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

def validate_dsmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, 
                  loss_fn = None, results_dir=None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            wsi_label = torch.zeros(n_classes)
            wsi_label[label.long()] = 1
            wsi_label = wsi_label.to(device)

            logits, Y_prob, Y_hat, score, patch_logits = model(data)
            max_logits, _ = torch.max(patch_logits, 0)    

            bag_loss = loss_fn(logits.view(1, -1), wsi_label.view(1, -1))
            max_loss = loss_fn(max_logits.view(1, -1), wsi_label.view(1, -1))

            loss = 0.5*bag_loss + 0.5*max_loss

            inst_label = inst_label[0]
            score = score.T
            if inst_label!=[]:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score

            acc_logger.log(Y_hat, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    
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

def validate_addmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            wsi_label = torch.zeros(n_classes)
            wsi_label[label.long()] = 1
            wsi_label = wsi_label.to(device)

            logits, Y_prob, Y_hat, score, _ = model(data)
            
            inst_label = inst_label[0]
            score = score.T
            if inst_label!=[]:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        # prob[:,0] = 1-(prob[:,1:].sum(axis=1))
        # auc = roc_auc_score(labels, prob, multi_class='ovr')
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
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

def validate_transmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            wsi_label = torch.zeros(n_classes)
            wsi_label[label.long()] = 1
            wsi_label = wsi_label.to(device)

            logits, Y_prob, Y_hat, score, _ = model(data)
            
            inst_label = inst_label[0]
            score = score.T
            if inst_label!=[]:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        # prob[:,0] = 1-(prob[:,1:].sum(axis=1))
        # auc = roc_auc_score(labels, prob, multi_class='ovr')
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
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

def validate_dtfdmil(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None,
                    results_dir=None, task='camelyon'):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tier1, model_tier2 = model
    model_tier1.eval()
    model_tier2.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
        
            loss1 = loss_fn(slide_sub_preds, slide_sub_labels).mean()

            ## optimization for the second tier
            logits = model_tier2(slide_pseudo_feat)

            loss2 = loss_fn(logits, label).mean()

            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            
            Y_prob = torch.softmax(logits, dim=1)
            
            inst_label = inst_label[0]
            score = score.T
            if inst_label!=[]:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        # prob[:,0] = 1-(prob[:,1:].sum(axis=1))
        # auc = roc_auc_score(labels, prob, multi_class='ovr')
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)
        

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'
          .format(val_loss, val_error, auc, inst_auc))
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

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, 
                  results_dir = None, task='camelyon'):
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
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
#         for batch_idx, (data, label) in enumerate(loader):
        
            data, label = data.to(device), label.to(device)      
            
            logits, Y_prob, Y_hat, score, instance_dict = model(data, label=label, instance_eval=True)
            
#             inst_label = [[i%2 for i in range(score.shape[-1])]] ### 临时代码
            inst_label = inst_label[0]
            inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer

            score = score.T
            if inst_label!=[]:
                inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
                if score.shape[-1]==1:
                    inst_score = score.detach().cpu().numpy()[:,0]
                elif task == "camelyon":
                    inst_score = score[:,1].detach().cpu().numpy()[:]
                else:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
                all_inst_label += inst_label
                all_inst_score += inst_score
                
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.format(val_loss, val_error, auc, inst_auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)


    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        data, label = data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        
        with torch.no_grad():
            if args.model_type == 'dtfd_mil':
                slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
                logits = model_tier2(slide_pseudo_feat)
                Y_hat = torch.topk(logits, 1, dim = 1)[1]
                Y_prob = torch.softmax(logits, dim=1)
            else:
                logits, Y_prob, Y_hat, score, _ = model(data)

        inst_label = inst_label[0]
        inst_label = [1 if patch_label!=0 else 0 for patch_label in inst_label] # 单标签多分类只考虑 normal vs cancer
        score = score.T

        if inst_label!=[]:
            if score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif args.task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_score)
    print(classification_report(all_inst_label, all_inst_score))

    if args.n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, inst_auc, acc_logger

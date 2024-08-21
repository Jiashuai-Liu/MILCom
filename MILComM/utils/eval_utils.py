import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, ADD_MIL
from models.model_dsmil import DS_MIL
from models.model_transmil import TransMIL
from models.model_dtfd import DTFD_MIL_tier1, DTFD_MIL_tier2
# from models.model_addmil import AdditiveClassifier, DefaultAttentionModule, DefaultMILGraph
from models.model_nicwss import NICWSS

import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
import matplotlib.pyplot as plt
import utils.wsi_seam_utils as visualization

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

    label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
    inst_probs_tmp = [] 
    inst_preds_tmp = []

    inst_score = score.detach().cpu().numpy() # Nxcls 不含正常

    # max-min standard all_inst_score and all_inst_score_pos
    for class_idx in range(n_classes):

        if class_idx not in index_label:
            inst_score[:,class_idx] = [-1]*len(inst_label)
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
#     pos_accs = [sum(i)/len(i) if len(i) else 0 for i in pos_accs_each_wsi]
#     pos_acc = np.nanmean(np.array(pos_accs))
#     pos_acc_str = "seleted pos %f acc: "%(inst_rate)
#     for i in range(n_classes):
#         if i<n_classes-1:
#             pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])+', '
#         else:
#             pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])
#     print(pos_acc_str)
#     neg_acc = sum(neg_accs_each_wsi)/len(neg_accs_each_wsi)
#     print("seleted neg %f acc: %f" % (inst_rate, neg_acc))
    
    return inst_auc, inst_acc #, pos_accs, neg_acc

def initiate_model(args, ckpt_path):
    print('Init Model')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, 'fea_dim': args.fea_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict)
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

    elif args.model_type == 'dtfd_mil':
        model_tier1 = DTFD_MIL_tier1(**model_dict)
        model_tier2 = DTFD_MIL_tier2(**model_dict)
        model = [model_tier1, model_tier2]
    elif args.model_type in ['nic', 'nicwss']:
        model_dict.update({'only_cam': args.only_cam})
        model = NICWSS(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
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

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    if args.model_type == 'dtfd_mil':
        model[0].load_state_dict(ckpt_clean["model_tier1"])
        model[1].load_state_dict(ckpt_clean["model_tier2"])
    else:
        model.load_state_dict(ckpt_clean, strict=True)
        model.relocate()
        model.eval()
    return model

def eval_(dataset, args, ckpt_path):
    if not os.path.exists(ckpt_path):
        ckpt_path=ckpt_path.replace('_best.pt','.pt')
    print(ckpt_path)
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _, df_inst = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df, df_inst

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
    all_preds = np.zeros((len(loader), args.n_classes))
    
    slide_ids = loader.dataset.slide_data['slide_id']
    all_silde_ids=[]
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):

        data, label = data.to(device), label.to(device)
        
        if args.n_classes == 1:  # for binary
            index_label = torch.nonzero(label.squeeze(0)).to(device)
        else:
            index_label = torch.nonzero(label.squeeze()).to(device)
            
        if inst_label!=[]:
            all_inst_label += list(inst_label[0])
        
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
                score = score.T

        inst_label = inst_label[0]
        
        
        
        if inst_label!=[]:
            if args.model_type in ['nic', 'nicwss']:
                mask = cors[1]
                f_h, f_w = np.where(mask==1)
                # cam_n = cam_raw[0,:,...].squeeze(0)
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
            if type(cors[0][0]) is np.ndarray:
                cors = [["{}_{}_{}.png".format(str(cors[0][i][0]), str(cors[0][i][1]), args.p_size) for i in range(len(cors[0]))]]
                
            all_silde_ids += [os.path.join(slide_ids[batch_idx], cors[0][i]) for i in range(len(cors[0]))]

            
        acc_logger.log(Y_hat, index_label)
        probs = Y_prob.cpu().numpy()
        Y_hat_one_hot = np.zeros(args.n_classes)
        Y_hat_one_hot[Y_hat.cpu().numpy()]=1
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        all_preds[batch_idx] = Y_hat_one_hot
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                           'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label) # for binary
        test_error += error

    test_error /= len(loader)
    
    inst_auc, inst_acc = instance_report(args.n_classes, inst_binary_labels, inst_preds, inst_probs, 
                                                            all_inst_score, all_inst_score_neg, 
                                                            pos_accs_each_wsi, neg_accs_each_wsi, args.inst_rate)
    
    # calculate inst_auc and inst_acc   
    
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好

    aucs = []
    binary_labels = all_labels
    for class_idx in range(args.n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids}
    for c in range(args.n_classes):
        results_dict.update({'label_{}'.format(c): all_labels[:,c]})
        results_dict.update({'prob_{}'.format(c): all_probs[:,c]})
        results_dict.update({'pred_{}'.format(c): all_preds[:,c]})
    df = pd.DataFrame(results_dict)
    if all_inst_score.shape[1]<args.n_classes+1:
        all_inst_score = np.insert(all_inst_score, args.n_classes, values=all_normal_score, axis=1)
    inst_results_dict = {'filename':all_silde_ids,'label':all_inst_label}
    # print(len(all_silde_ids),len(all_inst_label),all_inst_score.shape)
    for c in range(args.n_classes+1):
        inst_results_dict.update({'prob_{}'.format(c): all_inst_score[:,c]})
    df_inst = pd.DataFrame(inst_results_dict)
    
    return patient_results, test_error, [auc_score,inst_auc,inst_acc], df, acc_logger, df_inst

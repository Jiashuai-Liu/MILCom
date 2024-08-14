import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB, ADD_MIL
from models.model_dsmil import DS_MIL
from models.model_transmil import TransMIL
from models.model_dtfd import DTFD_MIL_tier1, DTFD_MIL_tier2
from models.model_nicwss import NICWSS
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import auc as calc_auc
import utils.wsi_seam_utils as visualization


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
    #     model = DefaultMILGraph(
    #     pointer=DefaultAttentionModule(hidden_activation = nn.LeakyReLU(0.2), hidden_dims=[512, 256], input_dims=1024),
    #     classifier=AdditiveClassifier(hidden_dims=[512, 256], input_dims=1024, output_dims=args.n_classes)
    # )
    
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
        # model_tier1.relocate()
        # model_tier2.relocate()
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
    print('auc: ', auc[0])
    print('inst_auc: ', auc[1])
    return model, patient_results, test_error, auc, df, df_inst

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    
    if args.model_type == 'dtfd_mil':
        model_tier1, model_tier2 = model
        model_tier1.eval()
        model_tier2.eval()
    else:
        model.eval()
    #     model.cpu()
        
    # if args.model_type == 'trans_mil':
    #     # 指定量化配置，使用默认的动态量化配置
    #     model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    #     # 准备模型进行动态量化
    #     model_fp32_prepared = torch.quantization.prepare(model, inplace=False)
    #     # 对模型进行动态量化
    #     model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
    #     model_int8.cpu()
        
        
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    
    patient_results = {}
    
    all_silde_ids = []
    all_inst_score = []
    all_inst_label = []
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        data, label = data.cuda(), label.cuda()
        inst_label = inst_label[0]
        
        slide_id = slide_ids.iloc[batch_idx]
        if args.model_type in ['nic', 'nicwss']:
            label_bn = torch.zeros(args.n_classes)
            label_bn[label.long()] = 1
            label_bn = label_bn.to(device)
            label_bn = label_bn.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        
        with torch.no_grad():
            if args.model_type == 'dtfd_mil':
                slide_pseudo_feat, slide_sub_preds, slide_sub_labels, score = model_tier1(data, label)
                logits = model_tier2(slide_pseudo_feat)
                Y_hat = torch.topk(logits, 1, dim = 1)[1]
                Y_prob = torch.softmax(logits, dim=1)
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
                Y_hat = torch.argmax(Y_prob)
            else:
                logits, Y_prob, Y_hat, score, _ = model(data)
                score = score.T
            

        

        if inst_label!=[]:
            if args.model_type in ['nic', 'nicwss']:                
                mask = cors[1]
                f_h, f_w = np.where(mask==1)
                cam_n = cam_raw[0, label,...].squeeze(0).data.cpu().numpy()
                inst_score = cam_n[f_h, f_w]
                inst_pred = [1 if i>0.5 else 0 for i in inst_score]
                
            elif score.shape[-1]==1:
                inst_score = score.detach().cpu().numpy()[:,0]
            elif args.task == "camelyon":
                inst_score = score[:,1].detach().cpu().numpy()[:]
            else:
                inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = inst_score.squeeze()
            inst_score = list((inst_score-inst_score.min())/max(inst_score.max()-inst_score.min(),1e-10))
            all_inst_label += inst_label
            all_inst_score += inst_score

            if type(cors[0][0]) is np.ndarray:
                cors = [["{}_{}_{}.png".format(str(cors[0][i][0]), str(cors[0][i][1]), args.patch_size) for i in range(len(cors[0]))]]

            all_silde_ids += [os.path.join(slide_ids[batch_idx], cors[0][i]) for i in range(len(cors[0]))]
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    
    all_inst_label = [1 if i!=0 else 0 for i in all_inst_label]
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    all_inst_pred = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    print(classification_report(all_inst_label, all_inst_pred))
    
    ### maybe change later for the auc score
    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    df_inst = pd.DataFrame([all_silde_ids, all_inst_score, all_inst_label]).T
    df_inst.columns = ['filename', 'prob', 'label']
    # print(df_inst)
    
    return patient_results, test_error, [auc_score,inst_auc,inst_acc], df, acc_logger, df_inst

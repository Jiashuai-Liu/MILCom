# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code renal_subtyping_AddMIL_res50_1512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_CLAMSB_0512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_CLAMSB_noinst_0512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_DTFDMIL_res50_1512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_subtyping_AddMIL_res50_1512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_subtyping_DSMIL_res50_1512_5fold_s1
# CUDA_VISIBLE_DEVICES=2 python eval.py --models_exp_code camelyon_subtyping_TransMIL_res50_1512_5fold_s1

CUDA_VISIBLE_DEVICES=5 python eval.py --models_exp_code kica_subtyping_nicwss_conch_10x512_5fold_s1 \
    --data_root_dir /data_nas/ljs/KICA/feature_conch/
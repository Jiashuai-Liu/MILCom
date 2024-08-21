# CUDA_VISIBLE_DEVICES=1 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_gastric_subtyping_nicwss.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
    python -u main.py --config configs/config_nicwss.yaml \
        > main_camelyon_subtyping_nicwss.out 2>&1 &
# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_esca_subtyping_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_gastric_staging_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_renal_subtyping_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_lung_subtyping_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_kica_subtyping_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_kica_staging_nicwss.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nicwss.yaml \
#         > main_esca_staging_nicwss.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
    python -u main.py --config configs/config_nicwss.yaml \
        > main_camelyon_nicwss.out 2>&1 &
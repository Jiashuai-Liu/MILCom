# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_esca_subtyping_nic.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_gastric_staging_nic.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_renal_subtyping_nic.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_lung_subtyping_nic.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 \
    python -u main.py --config configs/config_nic.yaml \
        > main_kica_subtyping_nic.out 2>&1 &

# CUDA_VISIBLE_DEVICES=2 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_kica_staging_nic.out 2>&1 &

# CUDA_VISIBLE_DEVICES=0 \
#     python -u main.py --config configs/config_nic.yaml \
#         > main_esca_staging_nic.out 2>&1 &
#! /bin/bash
python create_patches_fp.py --patch --seg --patch_size 512 --step_size 512 --patch_level 0 --save_dir /home5/gzy/Cameylon/0_512
CUDA_VISIBLE_DEVICES=2,7 python extract_features_fp.py --data_h5_dir /home5/gzy/Cameylon/0_512/ --tag _0_512 --csv_path /home5/gzy/Cameylon/0_512/process_list_autogen.csv --feat_dir /home5/gzy/Cameylon/feature_resnet/ --batch_size 64
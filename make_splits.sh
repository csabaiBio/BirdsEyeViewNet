# Script to create all scripts in one go. Only run one time!

# # Yale 
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/Yale_HER2_status.csv' --split_name Yale_10_fold --val_frac 0.1 --test_frac 0.1

# HunCRC 
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/HunCRC_CRC_vs_clear.csv' --split_name HunCRC_10_fold  --val_frac 0.1 --test_frac 0.1

# Ovarian 
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/Ovarian_responce.csv' --split_name Ovarian_10_fold --val_frac 0.1 --test_frac 0.1

# Bracs
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/BRACS_malignancy.csv' --split_name Bracs_10_fold --val_frac 0.1 --test_frac 0.1

# DLBCL
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/DLBCL_6Year_PFS.csv' --split_name DLBCL_10_fold --val_frac 0.1 --test_frac 0.1

# PANDA
python create_splits_seq.py --task task_1_tumor_vs_normal --seed 50 --label_frac 1 --k 10 --data_csv 'data_cleaning/dataset_csv/PANDA_final.csv' --split_name PANDA_10_fold --val_frac 0.1 --test_frac 0.1

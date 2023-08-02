# Script to see how the accuracies vary for different numbers of epocs of training between FlockNet and ImageNet
# conda activate clam

declare -a models=("/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/FlockNet_1X/checkpoint0020.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/FlockNet_1X/checkpoint0040.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/FlockNet_1X/checkpoint0060.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/FlockNet_1X/checkpoint0080.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/FlockNet_1X/checkpoint0099.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/imagenet_full_dino/checkpoint0020.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/imagenet_full_dino/checkpoint0040.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/imagenet_full_dino/checkpoint0060.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/imagenet_full_dino/checkpoint0080.pth" "/mnt/ncshare/ozkilim/lymphoma/CLAM/Geo_CLAM/CNNs/imagenet_full_dino/checkpoint0099.pth")
declare -a model_names=("FlockNet20" "FlockNet40" "FlockNet60" "FlockNet80" "FlockNet100" "ImageNet20" "ImageNet40" "ImageNet60" "ImageNet80" "ImageNet100")

experement_number="2"

length=${#models[@]}

# for ((i = 0; i < length; i++)); 
#     do
#     model="${models[$i]}"
#     model_name="${model_names[$i]}"

#     # # ===== Generate all embeddings ===== 
#     CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir PATCHES/Ovarian_level_4 --data_slide_dir  /tank/WSI_data/Ovarian_combined_tif --csv_path PATCHES/Ovarian_level_4/files.csv --feat_dir /local_storage/CLAM_FEATURES/Ovarian/$model_name --batch_size 1024 --slide_ext .tif --embedder $model &
#     CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir PATCHES/Yale_Her2_level_4 --data_slide_dir /tank/WSI_data/SVS --csv_path PATCHES/Yale_Her2_level_4/files.csv --feat_dir /local_storage/CLAM_FEATURES/Yale/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
#     CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir PATCHES/DLBCL --data_slide_dir /tank/WSI_data/DLBCL_WSIs --csv_path /mnt/ncshare/ozkilim/lymphoma/CLAM/dataset_csv/DLBCL_6Year_PFS.csv --feat_dir /local_storage/CLAM_FEATURES/DLBCL/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
#     wait
#     CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir PATCHES/HunCRC_level_4 --data_slide_dir /tank/WSI_data/scidata --csv_path PATCHES/HunCRC_level_4/files.csv --feat_dir /local_storage/CLAM_FEATURES/HunCRC/$model_name --batch_size 1024 --slide_ext .mrxs --embedder $model  &
#     CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir PATCHES/Bracs_level_4 --data_slide_dir /tank/WSI_data/bracs_icar/BRACS_WSI --csv_path PATCHES/Bracs_level_4/files.csv --feat_dir /local_storage/CLAM_FEATURES/Bracs/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
#     CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir PATCHES/Melanoma --data_slide_dir /tank/WSI_data/visiomel_melanoma_drivendata/slides --csv_path /mnt/ncshare/ozkilim/lymphoma/CLAM/dataset_csv/melanoma_clean.csv --feat_dir /local_storage/CLAM_FEATURES/Melanoma/$model_name --batch_size 1024 --slide_ext .tif --embedder $model &

#     done

# wait

for ((i = 0; i < length; i++)); 
    do
    model="${models[$i]}"
    model_name="${model_names[$i]}"

    # ===== Run all experements =====
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/HunCRC/$model_name --exp_code HunCRC_$model_name"_"$experement_number --split_dir HunCRC_10_fold_100 --data_csv 'dataset_csv/HUnCRC_adenoma_vs_all.csv' --early_stopping  --max_epochs 100 --results_dir ./results_round2/HunCRC  &
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/Yale/$model_name --exp_code Yale_$model_name"_"$experement_number --split_dir Yale_10_fold_100 --data_csv 'dataset_csv/Yale_her2.csv' --early_stopping  --max_epochs 100 --results_dir ./results_round2/Yale  &
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/Ovarian/$model_name --exp_code Ovarian_$model_name"_"$experement_number --split_dir Ovarian_10_fold_100 --data_csv 'dataset_csv/ovarian_clean_seperated_cases.csv' --early_stopping  --max_epochs 100 --results_dir ./results_round2/Ovarian & 
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/Bracs/$model_name --exp_code Bracs_$model_name"_"$experement_number --split_dir Bracs_10_fold_100 --data_csv 'dataset_csv/bracs.csv' --max_epochs 100  --early_stopping   --results_dir ./results_round2/Bracs & 
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/Melanoma/$model_name --exp_code Melanoma_$model_name"_"$experement_number --split_dir Melanoma_10_fold_100_100 --data_csv 'dataset_csv/melanoma_clean.csv' --max_epochs 100  --early_stopping   --results_dir ./results_round2/Melanoma & 
    CUDA_VISIBLE_DEVICES=1,2 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type mil --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/CLAM_FEATURES/DLBCL/$model_name --exp_code DLBCL_$model_name"_"$experement_number --split_dir DLBCL_10_fold_100 --data_csv 'dataset_csv/DLBCL_6Year_PFS.csv' --max_epochs 100  --early_stopping  --results_dir ./results_round2/DLBCL & 
    wait
    done

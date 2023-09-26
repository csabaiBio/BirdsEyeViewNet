# Script to embedd at the datasets with the differnet embedding networks.
# make sure conda activate clam has been run so environment is activated! 

# Define model paths
# declare -a models=("../CNNs/meta_pretrained/dino_rn50_checkpoint.pth" "../CNNs/FlockNet_1X/checkpoint0099.pth" "../CNNs/FlockNet_2x/checkpoint0099.pth" "../CNNs/FlockNet_2x/checkpoint0050.pth" "random" "imagenet_supervised")
# # Define models names

# declare -a models=("../CNNs/meta_pretrained/dino_rn50_checkpoint.pth" "../CNNs/FlockNet_1X/checkpoint0099.pth" "../CNNs/FlockNet_2x/checkpoint0099.pth" "../CNNs/FlockNet_2x/checkpoint0050.pth" "random" "imagenet_supervised")

declare -a model_names=("TCGA")
declare -a models=("TCGA")

length=${#models[@]}

for ((i = 0; i < length; i++)); 
    do
    model="${models[$i]}"
    model_name="${model_names[$i]}"

    # # ===== Generate all embeddings ===== 
    CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir 20X_PATCHES/Ovarian_level_4 --data_slide_dir  /tank/WSI_data/Ovarian_combined_tif --csv_path data_cleaning/dataset_csv/Ovarian_responce.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/Ovarian/$model_name --batch_size 1024 --slide_ext .tif --embedder $model &
    CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir 20X_PATCHES/Yale_Her2_level_4 --data_slide_dir /tank/WSI_data/SVS --csv_path data_cleaning/dataset_csv/Yale_HER2_status.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/Yale/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
    CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir 20X_PATCHES/HunCRC_level_4 --data_slide_dir /tank/WSI_data/scidata --csv_path data_cleaning/dataset_csv/HunCRC_CRC_vs_clear.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/HunCRC/$model_name --batch_size 1024 --slide_ext .mrxs --embedder $model  &
    wait
    CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir 20X_PATCHES/BRACS --data_slide_dir /tank/WSI_data/bracs_icar/BRACS_WSI --csv_path data_cleaning/dataset_csv/BRACS_malignancy.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/Bracs/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
    CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir 20X_PATCHES/DLBCL --data_slide_dir /tank/WSI_data/DLBCL_WSIs --csv_path data_cleaning/dataset_csv/DLBCL_6Year_PFS.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/DLBCL/$model_name --batch_size 1024 --slide_ext .svs --embedder $model &
    CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py --data_h5_dir 20X_PATCHES/PANDA --data_slide_dir /tank/WSI_data/PANDA/1000_subset --csv_path data_cleaning/dataset_csv/PANDA_ISUP.csv --feat_dir /local_storage/FINAL_CLAM_FEATURES/20X/PANDA/$model_name --batch_size 1024 --slide_ext .tiff --embedder $model &

    
done


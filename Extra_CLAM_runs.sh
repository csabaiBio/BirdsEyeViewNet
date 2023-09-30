# Script to run all experements for tranfer learning cross scales with MIL
# make sure conda activate clam has been run so environment is activated! 

# declare -a models=("../CNNs/meta_pretrained/dino_rn50_checkpoint.pth" "..CNNs/FlockNet_1X/checkpoint0099.pth")
# Define models names

declare -a model_names=("FlockNet2X")

# label for the experements incase you need to re-run experiments.
length=${#model_names[@]}

for ((i = 0; i < length; i++)); 
    do
    model_name="${model_names[$i]}"

    experement_number="5"  
    # # ===== Run all experements =====
    # CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-3 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/FINAL_CLAM_FEATURES/20X/Yale/$model_name --exp_code Yale_$model_name"_"$experement_number --split_dir Yale_10_fold_100 --data_csv data_cleaning/dataset_csv/Yale_HER2_status.csv --early_stopping  --max_epochs 100 --results_dir ./heatmaps_extra/results_20X/Yale  &

    # # experement_number="4" 
    # # # ===== Run all experements =====
    # CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-4 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/FINAL_CLAM_FEATURES/20X/Yale/$model_name --exp_code Yale_$model_name"_"$experement_number --split_dir Yale_10_fold_100 --data_csv data_cleaning/dataset_csv/Yale_HER2_status.csv --early_stopping  --max_epochs 100 --results_dir ./heatmaps_extra/results_20X/Yale  &

    # experement_number="5" 
    # # ===== Run all experements =====
    CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-5 --k 10 --exp_code task_1_tumor_vs_normal_CLAM_50 --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type clam_sb --log_data --data_root_dir . --embedding_size 1000 --data_dir /local_storage/FINAL_CLAM_FEATURES/20X/Yale/$model_name --exp_code Yale_$model_name"_"$experement_number --split_dir Yale_10_fold_100 --data_csv data_cleaning/dataset_csv/Yale_HER2_status.csv --early_stopping  --max_epochs 100 --results_dir ./heatmaps_extra/results_20X/Yale  &

    wait
done


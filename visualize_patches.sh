# This script produces a few visual samples of for opensource images at both 10X and 20X resolution.

# 10X 

# DLBCL
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/DLBCL_6Year_PFS.csv --data_slide_dir /tank/WSI_data/DLBCL_WSIs  --data_h5_dir 10X_PATCHES/DLBCL --feat_dir  10X_IMAGES/DLBCL  --save_folder 10X_IMAGES/DLBCL  --slide_ext .svs
# Ovarian 
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/Ovarian_responce.csv --data_slide_dir /tank/WSI_data/Ovarian_combined_tif  --data_h5_dir 10X_PATCHES/Ovarian_level_4 --feat_dir  10X_IMAGES/Ovarian  --save_folder 10X_IMAGES/Ovarian  --slide_ext .tif
# BRACS
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/BRACS_malignancy.csv --data_slide_dir /tank/WSI_data/bracs_icar/BRACS_WSI  --data_h5_dir 10X_PATCHES/BRACS --feat_dir  10X_IMAGES/BRACS  --save_folder 10X_IMAGES/BRACS --slide_ext .svs
# Yale Her2
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/Yale_HER2_status.csv --data_slide_dir /tank/WSI_data/SVS  --data_h5_dir 10X_PATCHES/Yale_Her2_level_4 --feat_dir 10X_IMAGES/Yale  --save_folder 10X_IMAGES/Yale --slide_ext .svs
# HunCRC 
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/HunCRC_CRC_vs_clear.csv --data_slide_dir /tank/WSI_data/scidata  --data_h5_dir 10X_PATCHES/HunCRC_level_4 --feat_dir  10X_IMAGES/HunCRC  --save_folder 10X_IMAGES/HunCRC --slide_ext .mrxs
# PANDA 
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/PANDA_ISUP.csv --data_slide_dir /tank/WSI_data/PANDA/1000_subset  --data_h5_dir 20X_PATCHES/PANDA --feat_dir  20X_IMAGES/PANDA  --save_folder 20X_IMAGES/PANDA --slide_ext .tiff


# 20X 

# DLBCL
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/DLBCL_6Year_PFS.csv --data_slide_dir /tank/WSI_data/DLBCL_WSIs  --data_h5_dir 20X_PATCHES/DLBCL --feat_dir  20X_IMAGES/DLBCL  --save_folder 20X_IMAGES/DLBCL
# Ovarian 
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/Ovarian_responce.csv --data_slide_dir /tank/WSI_data/Ovarian_combined_tif  --data_h5_dir 20X_PATCHES/Ovarian_level_4 --feat_dir  20X_IMAGES/Ovarian  --save_folder 20X_IMAGES/Ovarian  --slide_ext .tif
# BRACS
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/BRACS_malignancy.csv --data_slide_dir /tank/WSI_data/bracs_icar/BRACS_WSI  --data_h5_dir 20X_PATCHES/BRACS --feat_dir  20X_IMAGES/BRACS  --save_folder 20X_IMAGES/BRACS
# Yale Her2
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/Yale_HER2_status.csv --data_slide_dir /tank/WSI_data/SVS  --data_h5_dir 20X_PATCHES/Yale_Her2_level_4 --feat_dir 20X_IMAGES/Yale  --save_folder 20X_IMAGES/Yale
# HunCRC 
python get_visual_samples.py --csv_path data_cleaning/dataset_csv/HunCRC_CRC_vs_clear.csv --data_slide_dir /tank/WSI_data/scidata  --data_h5_dir 20X_PATCHES/HunCRC_level_4 --feat_dir  20X_IMAGES/HunCRC  --save_folder 20X_IMAGES/HunCRC  --slide_ext .mrxs

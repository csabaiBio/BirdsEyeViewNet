# Set up scripts and patch at 10X and 20X magnification levels.
# Here we have 10X magnifications datasets

# DLBCL
# The study cohort consists of patients with de novo, CD20+ DLBCL treated with curative intent with R-CHOP or R-CHOP–like immunochemotherapy
python create_patches_fp.py --source /tank/WSI_data/DLBCL_WSIs --save_dir 10X_PATCHES/DLBCL --patch_size 256 --seg --patch --stitch  --patch_level 2 --preset bwh_biopsy.csv # 40x magnification (0.25 µm per pixel) for level 0 20X for level 1
# Ovarian 
python create_patches_fp.py --source /tank/WSI_data/Ovarian_combined_tif --save_dir 10X_PATCHES/Ovarian_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 2 --preset tcga.csv # 20x objective dataset for level 0 this is half steps! level 2 is 10X
# BRACS
python create_patches_fp.py --source /tank/WSI_data/bracs_icar/BRACS_WSI --save_dir 10X_PATCHES/BRACS --patch_size 256 --seg --patch --stitch  --patch_level 2 --preset tcga.csv # Aperio AT2 scanner at 0.25 µm/pixel for 40× resolution dataset.
# Yale Her2
python create_patches_fp.py --source /tank/WSI_data/SVS --save_dir 10X_PATCHES/Yale_Her2_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 1 --preset tcga.csv #20× magnification dataset
# HunCRC 
python create_patches_fp.py --source  /tank/WSI_data/scidata --save_dir 10X_PATCHES/HunCRC_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 5 --preset tcga.csv #40x magnification, which resulted in 0.1213 μm/pixel resolution half steps!

# PANDA
python create_patches_fp.py --source /tank/WSI_data/PANDA/1000_subset --save_dir 10X_PATCHES/PANDA --patch_size 256 --seg --patch --stitch  --patch_level 1 --preset tcga.csv #  20X dataset.


# Here we have 20X magnifications datasets

# DLBCL
python create_patches_fp.py --source /tank/WSI_data/DLBCL_WSIs --save_dir 20X_PATCHES/DLBCL --patch_size 256 --seg --patch --stitch  --patch_level 1 --preset bwh_biopsy.csv 
# Ovarian 
python create_patches_fp.py --source /tank/WSI_data/Ovarian_combined_tif --save_dir 20X_PATCHES/Ovarian_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset tcga.csv 
# BRACS
python create_patches_fp.py --source /tank/WSI_data/bracs_icar/BRACS_WSI --save_dir 20X_PATCHES/BRACS --patch_size 256 --seg --patch --stitch  --patch_level 1 --preset tcga.csv 
# Yale Her2
python create_patches_fp.py --source /tank/WSI_data/SVS --save_dir 20X_PATCHES/Yale_Her2_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset tcga.csv 
# HunCRC 
python create_patches_fp.py --source  /tank/WSI_data/scidata --save_dir 20X_PATCHES/HunCRC_level_4 --patch_size 256 --seg --patch --stitch  --patch_level 3 --preset tcga.csv 
# PANDA
python create_patches_fp.py --source /tank/WSI_data/PANDA/1000_subset --save_dir 20X_PATCHES/PANDA --patch_size 256 --seg --patch --stitch  --patch_level 0 --preset tcga.csv #  20X dataset.

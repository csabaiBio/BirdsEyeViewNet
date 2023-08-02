for i in $(cat /mnt/ncshare/ozkilim/lymphoma/CLAM/data_cleaning/dataset_csv/PANDA_image_list.csv)
do
  temp="${i%\"}"
  temp="${temp#\"}"

  cp /tank/WSI_data/PANDA/train_images/$temp.tiff /tank/WSI_data/PANDA/1000_subset
done

# copy the subset 500 images to speec up downstream processing!
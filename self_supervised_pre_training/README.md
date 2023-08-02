MULTI GPU webdataset loader: main_dino_webdataset.py

For different data input modify only the urls variable that points to the tar files.

FlockNet 1X satellite 1.2M:

torchrun --nproc_per_node=8 main_dino_webdataset.py --arch resnet50 --optimizer sgd --lr 0.03 --weight_decay 1e-4 --weight_decay_end 1e-4 --global_crops_scale 0.14 1 --local_crops_scale 0.05 0.14 --data_path ' ' --output_dir /data/csabai-group/mixed_1X_output/ --epochs 100 --batch_size 128 --num_workers 6


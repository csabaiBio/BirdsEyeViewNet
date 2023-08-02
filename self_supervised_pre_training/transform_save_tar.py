import os
import glob
import h5py
import webdataset as wds
from itertools import islice
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import argparse


def BEV_transform(image):
    """Increased dynmic range like bird vision"""
    B, G, R = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    image = cv2.merge((B,G,R))

    return image

    
def create_tar_from_hdf5_files(input_files, output_dir, images_per_tar=50000):
    tar_counter = 0
    images_counter = 0
    images_counter_all = 0
    tar_path = os.path.join(output_dir, f'dataset_%04d.tar' % tar_counter)
    
    with wds.TarWriter(tar_path) as sink:
        for f in tqdm(range(input_files.shape[0])):
            index = f
            file_path = input_files[f]
            try:
                with h5py.File(file_path, 'r') as in_h5:
                    # Get the variable-length image data from the HDF5 file
                    image_data = in_h5['imgs']
                    num_images = image_data.shape[0]
                    
                    # Write the data to the tar file
                    for i in range(num_images):
                        image = BEV_transform( image_data[i] )
                        # if not np.mean(image) > 250:
                    
                        # Convert the image to PPM format and compress it
                        sink.write({
                            "__key__": f"sample%09d" % (images_counter_all + i),
                            "ppm": image,
                        })

                        images_counter += 1

                        # Create a new tar file if the number of images exceeds the limit
                        if images_counter >= images_per_tar:
                            tar_counter += 1
                            tar_path = os.path.join(output_dir, f'dataset_%04d.tar' % tar_counter)
                            sink.close()
                            sink = wds.TarWriter(tar_path)
                            images_counter = 0
                        
                    images_counter_all += num_images
            except:
                print("corrupted file?")

# Add arguments
number_of_h5_files = 2580 # each file contains 1000 images. This is for FlockNet2X as there are 2.58 million images.
parser = argparse.ArgumentParser()
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=number_of_h5_files)
args = parser.parse_args()

data_path = '/path/to/h5/files/'
input_dir =  '/path/to/h5/files/'
output_dir = '/path/to/save/tars/'
os.makedirs(output_dir, exist_ok=True)
images_per_tar = 10000

# Get the list of HDF5 files.
hdf5_files = np.array( sorted( glob.glob(os.path.join(input_dir, '*.h5')) ) )

## actual process happens here
# Create tar files from the HDF5 files
create_tar_from_hdf5_files(hdf5_files[args.start_idx:args.end_idx], output_dir, images_per_tar=10000)


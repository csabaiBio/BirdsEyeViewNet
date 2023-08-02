import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import tiffslide
from torchvision import models
import past_work.vision_transformer as vits
from past_work.vision_transformer import DINOHead


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=False, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features) #might need an augmentation on this to amek slides more natural earth colors.

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


def resnet_50_model_dino(pretrained_weights):
	"""Load resnet 50 model given path to pretrained weights from dino."""
	model = models.resnet50(pretrained=False)
	checkpoint_key = 'teacher'
	state_dict = torch.load(pretrained_weights, map_location="cpu")
	if checkpoint_key is not None and checkpoint_key in state_dict:
		print(f"Take key {checkpoint_key} in provided checkpoint dict")
		state_dict = state_dict[checkpoint_key]
	# remove `module.` prefix
	state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
	# remove `backbone.` prefix induced by multicrop wrapper
	state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
	msg = model.load_state_dict(state_dict, strict=False)
	print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

	return model


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--embedder', type=str, default="imagenet")

args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')

	#define embedding type.
	model_type = args.embedder

	if model_type == "satelite":
		checkpoint = torch.load("./satelite_models/B3_rn50_moco_0099_ckpt.pth") #https://github.com/zhu-xlab/SSL4EO-S12/tree/d2868adfada65e40910bfcedfc49bc3b20df2248
		state_dict = checkpoint['state_dict']
		for k in list(state_dict.keys()):
			# retain only encoder up to before the embedding layer
			if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
				#pdb.set_trace()
				# remove prefix
				state_dict[k[len("module.encoder_q."):]] = state_dict[k]
			# delete renamed or unused k
			del state_dict[k]

		model = models.resnet50(pretrained=False)
		msg = model.load_state_dict(state_dict,strict=False) 

	elif model_type == "random":
		model = models.resnet50(pretrained=False)

	elif model_type == "bracs":
		state_dict = torch.load("/mnt/ncshare/ozkilim/nightingale_breast/phase1/bracs/bracks.pth")
		model = models.resnet50(pretrained=False)
		msg = model.load_state_dict(state_dict,strict=False) 


	elif model_type == "imagenet_supervised":
		model = models.resnet50(pretrained=True)
		# model = nn.Sequential(*(list(model.children())[:-2])) # strips off last linear layer.

	else:
		# Load the model path for dino pre-trained model  
		model = resnet_50_model_dino(model_type)



	# elif model_type == "TGCA_VIT16":
	# 	# """ Load pretrained HIPT embedder for comparason aagainst state of the art """
	# 	# model = #load vit16... 
	# 	# model = models.vit_b_16(pretrained=False) # my need to load from HIPT the full model.
	# 	# pretrained_weights = "/mnt/ncshare/ozkilim/dino/saved_models/tgca_HIPT_vit/vit256_small_dino.pth"
	# 	# state_dict = torch.load(pretrained_weights, map_location="cpu")
    #     # if checkpoint_key is not None and checkpoint_key in state_dict:
    #     #     print(f"Take key {checkpoint_key} in provided checkpoint dict")
    #     #     state_dict = state_dict[checkpoint_key]
    #     # # remove `module.` prefix
    #     # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # # remove `backbone.` prefix induced by multicrop wrapper
    #     # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #     # msg = model.load_state_dict(state_dict, strict=False)
    #     # print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
	# 	pass

	# else:
	# 	print("No model chosen to make embeddings. Pick one of random,imagnet,bracs satelite!")

	model = model.to(device)
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):

		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()

		try: 

			if args.slide_ext == ".tiff":
				wsi = tiffslide.open_slide(slide_file_path) 
			else:
				wsi = openslide.open_slide(slide_file_path) 

			output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
			model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))

		except: 
			print("BUG IN SAVING WSI reading.!") #temp fix remove for broken slides after... 


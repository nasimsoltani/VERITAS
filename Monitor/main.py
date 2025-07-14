# import the required packages
import os
import pickle
from tqdm import tqdm
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
#import torch.nn.functional as F
import argparse


from monitor_models import MonitorNet
from monitor_training import train_model
from monitor_dataset import TrainDataset
from monitor_train_dataset_reader import pkl_reader
from monitor_test_model import test_model

parser = argparse.ArgumentParser(description = 'Train and validation pipeline',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu_id', default=0, type=int, help='ID of GPU to be used')
parser.add_argument('--train_dataset_path', default='', type=str, help='path to dataset')
parser.add_argument('--test_dataset_path', default='', type=str, help='path to dataset')
parser.add_argument('--channel_param', default='', type=str, help='Channel_Profile, Tx_Speed, Delay_Spread')
parser.add_argument('--train', action='store_true', help='if set to true, training will be done')
parser.add_argument('--test', action='store_true', help='if set to true, test will be done')
parser.add_argument('--weight_path', default='', type=str, help='path to trained weights for performing model test')

args = parser.parse_args()

# set user parameters
batch_size = 32
save_path = os.path.abspath('results')
# Initial configurations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

pred_pkl_path = os.path.join(save_path, 'preds_' + args.channel_param + '.pkl')

# create the model:
model = MonitorNet()

#model.load_state_dict(torch.load(args.weight_path, weights_only=True))


dummy_input = torch.rand((1,2,90,36))
dummy_output = model(dummy_input)

print(dummy_output.shape)

# print number of parameters in the model
pp=0
for p in list(model.parameters()):
    n=1
    for s in list(p.size()):
        n = n*s
    pp += n
print('This model has ' +str(pp)+ ' parameters')

train_dataset_path = os.path.abspath(args.train_dataset_path) 

if args.train:

	train_cache, val_cache = pkl_reader(train_dataset_path, args.channel_param)

	print(len(train_cache[0]))
	print(len(train_cache[1]))
	print(len(list(train_cache.keys())))

	#training files = 65340  or 98978

	# create the data loader and data generators
	train_dataset = TrainDataset(train_cache)
	val_dataset = TrainDataset(val_cache)
	train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
	val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

	train_model(model, train_dl, val_dl, args.channel_param, save_path)

if args.test:

	test_dataset_path = os.path.abspath(args.test_dataset_path) 
	weight_path = os.path.abspath(args.weight_path) 
	
	if args.channel_param == 'Channel_Profile':
		
		channel_models = ['tdl_a','tdl_b','tdl_c','tdl_d','tdl_e','uniform']
		speed_range = [10]
		# Test the trained model on the test dataset
		print('Testing on the test dataset')
		pred_dict = test_model(model, weight_path, test_dataset_path, channel_models, speed_range)
		print('Testing on the test dataset done')

		# Test the trained model on the training dataset to form in-distribution clusters
		print('Testing on the training dataset')
		channel_models = ['tdl_d','uniform']
		pred_dict_id = test_model(model, weight_path, train_dataset_path, channel_models, speed_range)
		print('Testing on the training dataset done')
		
	if args.channel_param == 'Tx_Speed':
		
		speed_range = [0,1,2,3,4,20]
		channel_models = ['tdl_d']
		# Test the trained model on the test dataset
		print('Testing on the test dataset')
		pred_dict = test_model(model, weight_path, test_dataset_path, channel_models, speed_range)
		print('Testing on the test dataset done')

		# Test the trained model on the training dataset to form in-distribution clusters
		print('Testing on the training dataset')
		speed_range = [0,1,2] # just id classes		
		pred_dict_id = test_model(model, weight_path, train_dataset_path, channel_models, speed_range)
		print('Testing on the training dataset done')
	
	if args.channel_param == 'Delay_Spread':
		
		speed_range = [10.0, 50.0, 80.0, 100.0, 200.0, 400.0]  # these are actually delay values saved as speed variable
		channel_models = ['tdl_b']

		# Test the trained model on the test dataset
		print('Testing on the test dataset')
		pred_dict = test_model(model, weight_path, test_dataset_path, channel_models, speed_range)
		print('Testing on the test dataset done')

		# Test the trained model on the training dataset to form in-distribution clusters
		print('Testing on the training dataset')
		speed_range = [10.0, 50.0, 80.0]    # just id classes
		pred_dict_id = test_model(model, weight_path, train_dataset_path, channel_models, speed_range)
		print('Testing on the training dataset done')
	

	preds = {}
	preds['pred_dict'] = pred_dict
	preds['pred_dict_id'] = pred_dict_id

	with open (pred_pkl_path, 'wb') as handle:
		pickle.dump(preds, handle)

	print('***** Results written in a file in: ' + str(pred_pkl_path) + ' *****')

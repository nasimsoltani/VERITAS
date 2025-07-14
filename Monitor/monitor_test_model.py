from monitor_models import MonitorNet
from monitor_train_dataset_reader import aux_channel_generator

import torch
import numpy as np
import pickle
from tqdm import tqdm
import random


def test_model(model, weight_path, dataset_path, channel_models, speed_range):

	# load the trained weights:
	model.load_state_dict(torch.load(weight_path, weights_only=True))

	# We don't need gradients on to do reporting
	model.train(False)
	model.cuda()
	# model.cpu()
	model.eval()

	SNR_range = list(np.arange(21,step=2))
	multiburst_degree = 3 

	# load the dataset pickle file 
	with open (dataset_path,'rb') as handle:
		pkl_content = pickle.load(handle)

	print('file read')

	# create empty pred_dict

	pred_dict = {}
	for channel in channel_models:
		pred_dict[channel] = {}
		for channel in pred_dict:
			for speed in speed_range:
				pred_dict[channel][speed] = {} 
				for SNR in SNR_range:
					pred_dict[channel][speed][SNR] = [] 

	# test the model
	with torch.no_grad():   
		for channel in channel_models:   
			for speed in tqdm(speed_range):
				for SNR in SNR_range:
					SNR_index = SNR_range.index(SNR)

					if channel in ['uniform','gaussian','shuffled','rayleigh']:
						this_list = pkl_content['tdl_d'][speed][SNR]
						aux_type = channel
					else:	
						this_list = pkl_content[channel][speed][SNR]
						aux_type = None


					random.shuffle(this_list)
					super_list = []
					start_index = 0
					# stack multiburst_degree of these grids together:
					num_super_grids = int(len(this_list)-multiburst_degree+1)
					for _ in range(num_super_grids):
						super_grid = np.empty((0,36,2))
						for i in range(multiburst_degree):

							this_grid = this_list[start_index+i]
							this_grid = aux_channel_generator(this_grid, aux_type)   # this will only modify the grid if channel uniform or gaussian otherwise it sends the grid out without change

							super_grid = np.concatenate((super_grid,this_grid),axis=0)
    
						# now the supergrid is formed, attach it to super_list
						super_list.append(super_grid)
						# start next supergrid with a stride of 1
						start_index += 1 #multiburst_degree
					
					# now super_list is ready, put it in this_list:
					this_list = super_list

					for X in this_list:

						X = np.moveaxis(X, -1, 0)
						X = np.expand_dims(X, axis=0)
						X = torch.from_numpy(X).float().cuda()
						y_hat = model(X)

						pred_dict[channel][speed][SNR].append(y_hat.cpu().detach().numpy())

	return pred_dict

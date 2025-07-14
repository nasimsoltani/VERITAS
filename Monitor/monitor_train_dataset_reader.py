import pickle
import numpy as np
import random

def aux_channel_generator(X, aux_type):
	if aux_type == 'uniform':
		r1 = np.amax(X)
		r2 = np.amin(X)
		y = (r1-r2) * np.random.rand(X.shape[0],X.shape[1],X.shape[2]) + r2
	elif aux_type == 'gaussian':
		mean = np.mean(X)
		std = np.std(X)
		y = np.random.normal(loc=mean, scale=std, size=X.shape)
	elif aux_type == 'shuffled':
		noise_grid = X.reshape(1,X.shape[0]*X.shape[1]*X.shape[2])
		random.shuffle(noise_grid)
		y = noise_grid.reshape(X.shape)
	elif aux_type == 'rayleigh':
		std = np.std(X)
		y = np.random.rayleigh(scale=std, size=X.shape)
	else:
		y = X
	return y


def pkl_reader(dataset_pickle_path, channel_param):

	""" this function reads the training set pickle file and based on the channel parameter
		that we want to train for, outputs a train_cache and val_cache with keys as indexes 
		of values in that channel parameter in-distribution class list """

	# load the pickle file
	with open (dataset_pickle_path,'rb') as handle:
		pkl_content = pickle.load(handle)

	train_cache = {}
	val_cache = {}
	multiburst_degree = 3
	data_portion_used = 1
	SNR_range = list(np.arange(21,step=2))

	if channel_param == 'Channel_Profile':
		print('***** Training for Channel Profile *****')
		channel_models = ['tdl_d','uniform'] #,'gaussian'] 
		speed_range = [10]
		label_range = channel_models

	elif channel_param == 'Tx_Speed':
		print('***** Training for Transmitter Speed *****')
		channel_models = ['tdl_d']
		speed_range = [0,1,2]
		label_range = speed_range

	elif channel_param == 'Delay_Spread':
		print('***** Training for Delay Spread *****')
		channel_models = ['tdl_b']
		speed_range = [10.0, 50.0, 80.0]
		label_range = speed_range


	for label in label_range:
		train_cache[label_range.index(label)], val_cache[label_range.index(label)] = [], []


	# now fill in the caches using pkl dataset content
	for channel_model in channel_models:
		print(channel_model)
		for speed in speed_range:
			for SNR in SNR_range:
				if channel_model in ['uniform','gaussian','shuffled','rayleigh']:
					aux_type = channel_model
					this_list = pkl_content['tdl_d'][speed][SNR]
				else:
					aux_type = None
					this_list = pkl_content[channel_model][speed][SNR]
				
				random.shuffle(this_list)
				this_list = this_list[:int(data_portion_used*len(this_list))]
				super_list = []
				start_index = 0
				# stack multiburst_degree of these grids together:
				num_super_grids = int(len(this_list)-multiburst_degree+1)
				
				for _ in range(num_super_grids):
					super_grid = np.empty((0,36,2))
					for i in range(multiburst_degree):
						this_grid = this_list[start_index+i]
						
						this_grid = aux_channel_generator(this_grid, aux_type)   # this will only modify the grid if channel is uniform or gaussian otherwise it sends the grid out without change
						super_grid = np.concatenate((super_grid,this_grid),axis=0)
					# now the supergrid is formed, attach it to super_list
					super_list.append(super_grid)
					# start next supergrid with a stride of 1
					start_index += 1
				# now super_list is ready, put it in this_list:
				this_list = super_list

				if channel_param == 'Channel_Profile':
					train_cache[label_range.index(channel_model)].extend(this_list[:int(0.9*len(this_list))])
					val_cache[label_range.index(channel_model)].extend(this_list[int(0.9*len(this_list)):])
				elif channel_param == 'Tx_Speed' or channel_param == 'Delay_Spread':
					train_cache[label_range.index(speed)].extend(this_list[:int(0.9*len(this_list))])
					val_cache[label_range.index(speed)].extend(this_list[int(0.9*len(this_list)):])


	return train_cache, val_cache

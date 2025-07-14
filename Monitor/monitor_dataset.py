from torch.utils.data import Dataset
import random
import numpy as np


## Data loader for triplet loss

class TrainDataset(Dataset):
    def __init__(self, data_cache):

        self.data_cache = data_cache

    def __len__(self):
      	# loop over the dataset class as many as we define here
        return int(len(list(self.data_cache.keys()))*len(self.data_cache[0]) * 1.0)


    def __getitem__(self, index):
        #Generate two samples of data (anchor and positive)
        this_random_class = random.sample(list(self.data_cache.keys()), 1)[0]
        anchor_and_positive = random.sample(self.data_cache[this_random_class], 2)
        this_random_grid = anchor_and_positive[0]
        X = np.moveaxis(this_random_grid, -1, 0)
        y = int(this_random_class)     # creating y (output)

        ## load the positive sample:
        positive_grid = anchor_and_positive[1]
        positive_grid = np.moveaxis(positive_grid, -1, 0)
        ## load the negative sample
        negative_class_list = list(self.data_cache.keys())
        negative_class_list.remove(this_random_class)
        negative_class = random.sample(negative_class_list, 1)[0]
        negative_grid = random.sample(self.data_cache[negative_class], 1)[0]
        negative_grid = np.moveaxis(negative_grid, -1, 0)

        # return shuffled version of anchor as the negative grid:
#         negative_grid = X.reshape(1,X.shape[0]*X.shape[1]*X.shape[2])
#         random.shuffle(negative_grid)
#         negative_grid = negative_grid.reshape(X.shape)

        # return noise as the negative grid:
#         r1 = np.amax(X)
#         r2 = np.amin(X)
#         negative_grid = (r1-r2) * np.random.rand(X.shape[0],X.shape[1],X.shape[2]) + r2


        return X, positive_grid, negative_grid, y



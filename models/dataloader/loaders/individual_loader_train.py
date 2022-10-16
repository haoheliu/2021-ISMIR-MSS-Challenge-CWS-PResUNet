
import sys
sys.path.append("/vol/research/dcase2022/project/2021-ISMIR-MSS-Challenge-CWS-PResUNet")

from torch.utils.data import Dataset
from utils._random_torch import random_choose_list, uniform_torch
import os
import numpy as np
import torch
import random


HOURS_FOR_A_EPOCH=100

class INDIVIDUAL_LOADER_train(Dataset):
    '''
        {
            "type-of-source"{               # e.g. vocal, bass
                "dataset-name": "<path to .lst file (a list of path to wav files)>",
            }
            ...
        }
    '''
    def __init__(self):
        """
        :param frame_length: segment length in seconds
        :param sample_rate: sample rate of target dataset
        :param data: a dict object containing the path to .lst file
        :param augmentation(deprecated): Optional
        :param aug_conf: Optional, used to update random server
        :param aug_sources: the type-of-source needed for augmentation, for example, vocal
        :param aug_effects: the effects to take, for example: ['tempo','pitch'].
        """
        self.train_dir = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/train"
        self.train_list = os.listdir(self.train_dir)
        self.segment_length = 300
        self.init_processes = []
        self.data_all = {}


    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def unify_length(self, x):
        length = x.shape[1]
        if(length > self.segment_length):
            start = int(uniform_torch(0, length-self.segment_length))
            return x[:,start:start+self.segment_length]
        else:
            ret = np.zeros((80, 300))
            ret[:,:length] = x
            return ret

    def __getitem__(self, item):
        # [samples, channels], 2**15, int
        data = {}

        if(os.getpid() not in self.init_processes):
            self.init_processes.append(os.getpid())
            self.set_seed(os.getpid())
        train_file = random_choose_list(self.train_list)
        feature = np.load(os.path.join(self.train_dir, train_file), allow_pickle=True)
        x = feature.item()["x"]
        y = feature.item()["y"]
        z = feature.item()["z"]
        x = self.unify_length(x)
        y = self.unify_length(y)
        z = self.unify_length(z)
        return x, y, z, train_file

    def __len__(self):
        # A Epoch = every 100 hours of datas
        return int(len(self.train_list))

if __name__ == "__main__":
    loader = INDIVIDUAL_LOADER_train()
    for x, z in loader:
        print(x.shape, z.shape)
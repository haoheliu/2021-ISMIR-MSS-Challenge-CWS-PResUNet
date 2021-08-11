
import sys

from torch.utils.data import Dataset
from models.dataloader.utils import *
import os

class ALL_LOADER(Dataset):
    '''
        {
            "type-of-source"{               # e.g. vocal, bass
                "dataset-name": "<path to .lst file (a list of path to wav files)>",
            }
            ...
        }
    '''
    def __init__(self,
                 target:str,  # ["vocals","bass","drums","other"]
                 frame_length=3.0,
                 sample_rate = 44100,
                 data = {},
                 overlap_num=1,
                 ):
        """
        :param frame_length: segment length in seconds
        :param sample_rate: sample rate of target dataset
        :param data: a dict object containing the path to .lst file
        :param augmentation(deprecated): Optional
        :param aug_conf: Optional, used to update random server
        :param aug_sources: the type-of-source needed for augmentation, for example, vocal
        :param aug_effects: the effects to take, for example: ['tempo','pitch'].
        """

        self.all_type = ['bass','drums','other','vocals']

        self.init_processes = []
        self.overlap_num = overlap_num
        self.sample_rate = sample_rate
        self.data_all = {}
        for k in data.keys():
            self.data_all[k] = construct_data_folder(data[k])
        self.frame_length = frame_length

    def random_fname(self,type, dataset_name = None):
        if(dataset_name is None):
            dataset_name = get_random_key(self.data_all[type][1],self.data_all[type][2])
        return self.data_all[type][0][dataset_name][random_torch(high=len(self.data_all[type][0][dataset_name]), to_int=True)],dataset_name

    def random_trunk(self, frame_length, type=None):
        # [samples, channel]
        trunk, length, sr = None , 0, self.sample_rate
        while (length-frame_length < -0.1):
            fname, dataset_name = self.random_fname(type=type)
            trunk_length = frame_length - length
            segment, duration, sr = random_chunk_wav_file(fname, trunk_length)
            assert sr == self.sample_rate
            length += duration
            if (trunk is None): trunk = segment
            else: trunk = np.concatenate([trunk, segment], axis=0)
        return trunk

    def set_seed(self,seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def __getitem__(self, item):
        # [samples, channels], 2**15, int
        data = {}

        if(os.getpid() not in self.init_processes):
            self.init_processes.append(os.getpid())
            self.set_seed(os.getpid())

        for k in self.all_type:
            data[k] = self.random_trunk(self.frame_length, type=k)
            data[k] = torch.tensor(data[k].astype(np.float32))
            data[k] = constrain_length_torch(data[k],self.sample_rate*self.frame_length)

        return data

    def __len__(self):
        # A Epoch = every 100 hours of datas
        return int(3600*100 / self.frame_length)

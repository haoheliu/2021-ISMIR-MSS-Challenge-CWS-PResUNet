import pytorch_lightning as pl
from torch.utils.data import DataLoader
from models.dataloader.loaders.individual_loader_train import INDIVIDUAL_LOADER_train
from models.dataloader.loaders.individual_loader_test import INDIVIDUAL_LOADER_test
from models.dataloader.loaders.all_loader import ALL_LOADER
from models.dataloader.loaders.paried_loader import PairedFullLengthDataLoader
from torch.utils.data.distributed import DistributedSampler

class MUSDB18HQDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data,
                 train_loader = "INDIVIDUAL_LOADER", train_type="vocals",
                 overlap_num = 1,
                 distributed = False,
                 batchsize = 12, frame_length=4.0, num_workers = 12, sample_rate = 44100,
                 quiet_threshold = 0
                 ):

        super(MUSDB18HQDataModule, self).__init__()
        self.train_data = train_data
        self.overlap_num = overlap_num
        self.test_data = test_data
        self.distributed=distributed
        self.train_type = train_type
        self.batchsize = batchsize
        self.frame_length = frame_length
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.train_loader = train_loader
        self.quiet_threshold = quiet_threshold

    def setup(self, stage = None):
        if(stage == 'fit' or stage is None):
            self.train = INDIVIDUAL_LOADER_train()

            self.val = INDIVIDUAL_LOADER_test()

        if(stage == "test" or stage is None):
            self.val = INDIVIDUAL_LOADER_test()

    def train_dataloader(self) -> DataLoader:
        if(self.distributed):
            sampler = DistributedSampler(self.train)
            return DataLoader(self.train, sampler = sampler, batch_size=self.batchsize, num_workers=self.num_workers, pin_memory=False)
        else:
            return DataLoader(self.train, batch_size=self.batchsize, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if(self.distributed):
            sampler = DistributedSampler(self.val,shuffle=False)
            return DataLoader(self.val, sampler = sampler, batch_size=1, pin_memory=False)
        else:
            return DataLoader(self.val, batch_size=1, shuffle=False)

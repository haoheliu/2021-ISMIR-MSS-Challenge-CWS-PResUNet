import torchaudio

torchaudio.set_audio_backend("sox_io")
from pytorch_lightning import Trainer
from pynvml import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from model import UNetResComplex_100Mb as MODEL
from models.dataloader.modules.MUSDB18HQDataModule import MUSDB18HQDataModule
from utils.callbacks.base import *
import os
sys.path.append(os.getcwd())
from models.config import Config
import time
from argparse import ArgumentParser
import torch

if __name__ == "__main__":
    ROOT = Config.ROOT

    assert len(Config.TRAIL_NAME) != 0

    if (os.path.exists("path.json")):
        os.remove("path.json")

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--name', type=str, default="four subband resunet model")
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--subband', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--warmup_data', type=float, default=26.6)
    parser.add_argument('--reduce_lr_period', type=int, default=400)
    parser.add_argument('--frame_length', type=float, default=3.0)
    parser.add_argument('--workers', type=int, default=22)

    parser.add_argument('--reload', type=str, default="")

    args = parser.parse_args()
    current = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    name = current + "-" + args.name
    if (len(args.reload) != 0):
        name += "_reload_" + (args.reload).replace("/", ".")

    if (torch.cuda.is_available()):
        nvmlInit()
        gpu_nums = int(nvmlDeviceGetCount())
        accelerator = 'ddp'
        distributed = True
    else:
        gpu_nums = 0
        accelerator = None
        distributed = False

    logger = TensorBoardLogger(save_dir=Config.TRAIL_NAME + "_log", name=name)
    print("DISTRIBUTED", distributed)
    # profiler = AdvancedProfiler(output_filename='_zz.txt', line_count_restriction=1.0)
    ####################################################################################################
    lr=0.0006
    check_val_every_n_epoch = 2
    gamma=args.gamma
    batchsize = args.batchsize
    frame_length = args.frame_length

    if(gpu_nums != 0):
        seconds_per_step = gpu_nums * batchsize * frame_length
    else:
        seconds_per_step = batchsize * frame_length

    warmup_data = args.warmup_data # hours of example
    reduce_lr_period = args.reduce_lr_period # hours of example
    sample_rate = 44100

    model = MODEL(channels=2, stem="all", nsrc=1, subband = args.subband,
                  # training
                  lr=lr,
                  gamma=gamma,
                  batchsize=batchsize,
                  frame_length=frame_length,
                  sample_rate=sample_rate,
                  warm_up_steps=int(warmup_data * 3600 / seconds_per_step), reduce_lr_steps=int(reduce_lr_period * 3600 / seconds_per_step),  # four gpus !!
                  # datas
                  check_val_every_n_epoch=check_val_every_n_epoch)

    # Data Module1
    dm = MUSDB18HQDataModule(
        distributed=distributed, train_loader="ALL_LOADER",train_type="all", overlap_num=1,
        train_data=Config.train_data, test_data=Config.test_data,
        batchsize=batchsize, frame_length=frame_length, num_workers=args.workers, sample_rate=sample_rate,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{val_loss:.4f}',
        monitor="val_loss",
        save_top_k=-1,
        mode='min',
    )

    init_log_dir = initLogDir(current_dir=os.getcwd())

    callbacks = []
    callbacks.append(init_log_dir)
    callbacks.extend([lr_monitor, checkpoint_callback])

    if(gpu_nums > 0):
        trainer = Trainer.from_argparse_args(args,
                                             gpus = list(range(0,gpu_nums)),
                                             max_epochs=5000,
                                             terminate_on_nan=True,
                                             num_sanity_val_steps=2,
                                             resume_from_checkpoint=args.reload if (len(args.reload) != 0) else None,
                                             callbacks=callbacks,
                                             accelerator=accelerator,
                                             sync_batchnorm=True,
                                             replace_sampler_ddp=False,
                                             check_val_every_n_epoch=check_val_every_n_epoch,
                                             checkpoint_callback=True, logger=logger, log_every_n_steps=10,
                                             progress_bar_refresh_rate=1, flush_logs_every_n_steps=200)
    else:
        trainer = Trainer.from_argparse_args(args,
                                             max_epochs=5000,
                                             terminate_on_nan=True,
                                             num_sanity_val_steps=2,
                                             resume_from_checkpoint=args.reload if (len(args.reload) != 0) else None,
                                             callbacks=callbacks,
                                             accelerator=accelerator,
                                             sync_batchnorm=True,
                                             replace_sampler_ddp=False,
                                             check_val_every_n_epoch=check_val_every_n_epoch,
                                             checkpoint_callback=True, logger=logger, log_every_n_steps=10,
                                             progress_bar_refresh_rate=1, flush_logs_every_n_steps=200)
    dm.setup('fit')
    trainer.fit(model, datamodule=dm)
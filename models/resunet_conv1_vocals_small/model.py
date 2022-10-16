import sys
import os
sys.path.append(os.getcwd())
from models.config import Config
from models.resunet_conv8_vocals.modules import *

import torch.utils
import torch.utils.data
import torch.nn.functional as F
from utils.f_helper import FDomainHelper
from torchlibrosa.stft import magphase
import numpy as np
import pytorch_lightning as pl
from torchlibrosa import STFT
from utils.overlapadd import LambdaOverlapAdd

from utils.file_io import *

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()

    def __call__(self, output, target):
        return self.loss(output,target)

class L1_Wav_L1_Sp(nn.Module):
    def __init__(self):
        super(L1_Wav_L1_Sp, self).__init__()
        self.f_helper = FDomainHelper()
        self.window_size = 2048
        hop_size = 441
        center = True
        pad_mode = "reflect"
        window = "hann"

        self.l1 = L1()
        self.stft = STFT(
            n_fft=self.window_size,
            hop_length=hop_size,
            win_length=self.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def __call__(self, output, target, alpha_t=1.0):
        wav_loss = self.l1(output, target)
        if(alpha_t < 1):
            sp_loss = self.l1(
                self.f_helper.wav_to_spectrogram(output, eps=1e-8),
                self.f_helper.wav_to_spectrogram(target, eps=1e-8)
            )
            sp_loss /= math.sqrt(self.window_size)
        else: sp_loss = 0.0
        return alpha_t*wav_loss + (1-alpha_t)*sp_loss

class UNetResComplex_100Mb(pl.LightningModule):
    def __init__(self, channels, target, nsrc=1, subband=4, use_lsd_loss=False,
                 lr=0.002, gamma=0.9,
                 batchsize=None, frame_length=None,
                 sample_rate=None,
                 warm_up_steps=1000, reduce_lr_steps=15000,
                 # datas
                 check_val_every_n_epoch=5,  # inside a validation set, how many samples gonna saved
                 ):
        # sub4 52.041G 66.272M
        super(UNetResComplex_100Mb, self).__init__()
        window_size = 2048
        hop_size = 441
        center = True,
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'relu'
        momentum = 0.01
        freeze_parameters = True
        self.use_lsd_loss = use_lsd_loss
        self.save_hyperparameters()
        self.nsrc = nsrc
        self.subband = subband
        self.channels = channels
        self.lr = lr
        self.gamma = gamma

        self.sample_rate = sample_rate
        self.batchsize = batchsize
        self.frame_length = frame_length

        # self.hparams['channels'] = 2
        self.target = target
        self.wav_spec_loss = L1_Wav_L1_Sp()
        # self.lsd_loss = get_loss_function("lsd")
        self.train_step = 0
        self.val_step = 0
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.val_result_save_dir = None
        self.val_result_save_dir_step = None
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
            subband=self.subband if(self.subband != 1) else None,
            root=Config.ROOT
        )

        self.bn0 = nn.BatchNorm2d(80, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes1(in_channels=1, out_channels=32,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes1(in_channels=32, out_channels=64,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes1(in_channels=64, out_channels=128,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes1(in_channels=128, out_channels=256,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockRes1(in_channels=256, out_channels=384,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockRes1(in_channels=384, out_channels=384,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = EncoderBlockRes1(in_channels=384, out_channels=384,
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockRes1(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockRes1(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes1(in_channels=384, out_channels=256,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes1(in_channels=256, out_channels=128,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes1(in_channels=128, out_channels=64,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes1(in_channels=64, out_channels=32,
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes1(in_channels=32, out_channels=32, downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()
        self.lr_lambda = lambda step: self.get_lr_lambda(step,
                                                        gamma = self.gamma,
                                                        warm_up_steps=warm_up_steps,
                                                        reduce_lr_steps=reduce_lr_steps)
        
        self.x = []

    def get_lr_lambda(self,step, gamma, warm_up_steps, reduce_lr_steps):
        r"""Get lr_lambda for LambdaLR. E.g.,

        .. code-block: python
            lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

            from torch.optim.lr_scheduler import LambdaLR
            LambdaLR(optimizer, lr_lambda)
        """
        if step <= warm_up_steps:
            return step / warm_up_steps
        else:
            return gamma ** (step // reduce_lr_steps)

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)

        # sp, cos_in, sin_in = self.f_helper.wav_to_mag_phase_subband_spectrogram(input)
        y = input.clone()
        

        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        # x = sp.transpose(1, 3)
        # """(batch_size, freq_bins, time_steps, channels_num)"""
        # x = self.bn0(x)  # normalization to freq bins
        # """(batch_size, freq_bins, time_steps, channels_num)"""
        # x = x.transpose(1, 3)
        # """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = y.shape[2]  # time_steps
        pad_len = int(np.ceil(y.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        y = F.pad(y, pad=(0, 0, 0, pad_len))
        # cos_in = F.pad(cos_in, pad=(0, 0, 0, pad_len))
        # sin_in = F.pad(sin_in, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        y = y[..., 0: y.shape[-1] - 1]  # (bs, channels, T, F)

        (N_, C_, T_, F_) = y.shape

        # UNet
        (y1_pool, y1) = self.encoder_block1(y)  # y1_pool: (bs, 32, T / 2, F / 2)
        (y2_pool, y2) = self.encoder_block2(y1_pool)  # y2_pool: (bs, 64, T / 4, F / 4)
        (y3_pool, y3) = self.encoder_block3(y2_pool)  # y3_pool: (bs, 128, T / 8, F / 8)
        (y4_pool, y4) = self.encoder_block4(y3_pool)  # y4_pool: (bs, 256, T / 16, F / 16)
        (y5_pool, y5) = self.encoder_block5(y4_pool)  # y5_pool: (bs, 512, T / 32, F / 32)
        (y6_pool, y6) = self.encoder_block6(y5_pool)  # y6_pool: (bs, 1024, T / 64, F / 64)
        y_center,_ = self.conv_block7(y6_pool)  # (bs, 2048, T / 64, F / 64)
        y7 = self.decoder_block1(y_center, y6)  # (bs, 1024, T / 32, F / 32)
        y8 = self.decoder_block2(y7, y5)  # (bs, 512, T / 16, F / 16)
        y9 = self.decoder_block3(y8, y4)  # (bs, 256, T / 8, F / 8)
        y10 = self.decoder_block4(y9, y3)  # (bs, 128, T / 4, F / 4)
        y11 = self.decoder_block5(y10, y2)  # (bs, 64, T / 2, F / 2)
        y12 = self.decoder_block6(y11, y1)  # (bs, 32, T, F)
        y,_ = self.after_conv_block1(y12)  # (bs, 32, T, F)
        y = self.after_conv2(y)  # (bs, channels, T, F)

        # Recover shape
        y = F.pad(y, pad=(0, 1))
        y = y[:, :, 0: origin_len, :]

        return y, y+input

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, amsgrad=True)
        # StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def preprocess(self, batch, train=False):
            return batch[0].permute(0,2,1).unsqueeze(1).float(), batch[1].permute(0,2,1).unsqueeze(1).float(), batch[2].permute(0,2,1).unsqueeze(1).float(), batch[3]

    def calc_loss(self, output, vocal):
        # l1 = self.wav_spec_loss(output, vocal)
        l1 = torch.mean(torch.abs(output-vocal))
        self.log("loss", l1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return l1

    def training_step(self, batch, batch_nb):
        x, y, z, fname = self.preprocess(batch)
        # self.x.append(z.flatten().detach().cpu().numpy()) # -1.92, 1.12
        # print(np.mean(self.x), np.std(self.x))
        z_est, y_est = self(x)
        loss = self.calc_loss(z_est, z)
        self.train_step += 1
        return {"loss": loss}

    def visualize(self, x, fname):
        import matplotlib.pyplot as plt
        plt.imshow(np.flipud(x[0,0].T), aspect="auto")
        plt.savefig(fname)

    def validation_step(self, batch, batch_nb):
        x, y, z, fname = self.preprocess(batch)
        fname = fname[0]
        z_est, y_est = self(x)
        os.makedirs(os.path.join(self.val_result_save_dir_step),exist_ok=True)
        
        if(batch_nb < 10):
            self.visualize(tensor2numpy(y_est), fname=os.path.join(self.val_result_save_dir_step, "y_est"+str(fname)+".png"))
            self.visualize(tensor2numpy(z_est), fname=os.path.join(self.val_result_save_dir_step, "z_est"+str(fname)+".png"))
            self.visualize(tensor2numpy(y), fname=os.path.join(self.val_result_save_dir_step, "y"+str(fname)+".png"))
            
        np.save(os.path.join(self.val_result_save_dir_step, "z_est"+str(fname)), tensor2numpy(z_est))
        np.save(os.path.join(self.val_result_save_dir_step, "y_est"+str(fname)), tensor2numpy(y_est))
        loss = self.calc_loss(z_est, z)
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        # Use the default log function to gather info from gpus
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)



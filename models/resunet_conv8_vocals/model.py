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

        if (subband == 8): self.bn0 = nn.BatchNorm2d(129, momentum=momentum)
        elif (subband == 4):
            self.bn0 = nn.BatchNorm2d(257, momentum=momentum)
        elif (subband == 2):
            self.bn0 = nn.BatchNorm2d(513, momentum=momentum)
        else:
            self.bn0 = nn.BatchNorm2d(1025, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes8(in_channels=channels * nsrc * subband, out_channels=32,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes8(in_channels=32, out_channels=64,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes8(in_channels=64, out_channels=128,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes8(in_channels=128, out_channels=256,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockRes8(in_channels=256, out_channels=384,
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockRes8(in_channels=384, out_channels=384,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = EncoderBlockRes8(in_channels=384, out_channels=384,
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.conv_block8 = EncoderBlockRes8(in_channels=384, out_channels=384,
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.conv_block9 = EncoderBlockRes8(in_channels=384, out_channels=384,
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.conv_block10 = EncoderBlockRes8(in_channels=384, out_channels=384,
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockRes8(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockRes8(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes8(in_channels=384, out_channels=256,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes8(in_channels=256, out_channels=128,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes8(in_channels=128, out_channels=64,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes8(in_channels=64, out_channels=32,
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes4(in_channels=32, out_channels=32, downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=channels * nsrc * 4 * subband,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()
        self.lr_lambda = lambda step: self.get_lr_lambda(step,
                                                        gamma = self.gamma,
                                                        warm_up_steps=warm_up_steps,
                                                        reduce_lr_steps=reduce_lr_steps)

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

        sp, cos_in, sin_in = self.f_helper.wav_to_mag_phase_subband_spectrogram(input)

        """(batch_size, channels_num, time_steps, freq_bins)"""

        # Batch normalization
        x = sp.transpose(1, 3)
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = self.bn0(x)  # normalization to freq bins
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        cos_in = F.pad(cos_in, pad=(0, 0, 0, pad_len))
        sin_in = F.pad(sin_in, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        (N_, C_, T_, F_) = x.shape

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center,_ = self.conv_block7(x6_pool)  # (bs, 2048, T / 64, F / 64)
        x_center,_ = self.conv_block8(x_center)  # (bs, 2048, T / 64, F / 64)
        x_center,_ = self.conv_block9(x_center)  # (bs, 2048, T / 64, F / 64)
        x_center,_ = self.conv_block10(x_center)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x,_ = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]

        cos_in = cos_in[:, :, 0:origin_len, :]
        sin_in = sin_in[:, :, 0:origin_len, :]

        mag, real,imag, residual, cos,sin,cos_out,sin_out = [],[],[],[],[],[],[], []

        sub_channels = self.subband*self.channels*self.nsrc

        for i in range(self.subband*self.channels*self.nsrc):
            # print(i + sub_channels, i + 1 + sub_channels)
            real.append(x[:,i+sub_channels:i+1+sub_channels,:,:])
            # print(i + sub_channels*2, i + 1 + sub_channels*2)
            imag.append(x[:, i + sub_channels*2:i + 1 + sub_channels*2, :, :])
            # print(i + sub_channels * 3, i + 1 + sub_channels * 3)
            residual.append(x[:, i + sub_channels*3:i + 1 + sub_channels*3, :, :])
            mag.append(torch.relu(torch.sigmoid(x[:, i:i + 1, :, :]) * sp[:, i:i + 1, :, :] + residual[-1]))
            (_, sub_cos, sub_sin) = magphase(real[-1],imag[-1])
            cos.append(sub_cos)
            sin.append(sub_sin)
            cos_out.append(cos_in[:,i:i+1,:,:]*sub_cos-sin_in[:,i:i+1,:,:]*sub_sin)
            sin_out.append(sin_in[:,i:i+1,:,:]*sub_cos+cos_in[:,i:i+1,:,:]*sub_sin)

        length = input.shape[2] // self.subband
        mag = torch.cat(mag,dim=1)
        cos_out = torch.cat(cos_out, dim=1)
        sin_out = torch.cat(sin_out, dim=1)

        wav_out = self.f_helper.mag_phase_subband_spectrogram_to_wav(sps=mag,coss = cos_out,sins=sin_out,length=length)

        pad_tail = input.size()[-1]-wav_out.size()[-1]
        wav_out = torch.nn.functional.pad(wav_out,(0,pad_tail))

        output_dict = {'wav': wav_out}
        return output_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        # StepLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    def preprocess(self, batch, train=False):
        if (train):
            vocal = batch['front']
            acc = batch['background']
            vocal, acc = vocal.float().permute(0, 2, 1), acc.float().permute(0, 2, 1)
            mixture = vocal + acc
            return vocal, acc, mixture
        else:  # during test or validaton
            vocal = batch[self.target]
            if(self.target == "bass"): acc = batch['no_bass']
            elif (self.target == "other"): acc = batch['no_other']
            elif (self.target == "drums"): acc = batch['no_drums']
            elif (self.target == "vocals"): acc = batch['acc']
            else: acc = None
            vocal, acc = vocal.float().permute(0, 2, 1), acc.float().permute(0, 2, 1)
            if(acc.size()[-1] != vocal.size()[-1]):
                min_length = min(acc.size()[-1], vocal.size()[-1])
                acc = acc[...,:min_length]
                vocal = vocal[..., :min_length]
            mixture = acc + vocal
            return vocal, acc, mixture, batch['fname'][0]  # a sample for a batch

    def calc_loss(self, output, vocal):
        l1 = self.wav_spec_loss(output, vocal)
        self.log("loss", l1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return l1

    def training_step(self, batch, batch_nb):
        vocal, acc, mixture = self.preprocess(batch, train=True)
        output = self(mixture)['wav']
        loss = self.calc_loss(output, vocal)
        self.train_step += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        vocal, acc, mixture, fname = self.preprocess(batch)
        continuous_nnet = LambdaOverlapAdd(
            nnet=self,
            n_src=self.channels * self.nsrc,
            window_size=self.sample_rate * 20,
            in_margin =int(self.sample_rate*1.5),
            window="boxcar",
            reorder_chunks=False,
            enable_grad=False,
            device=self.device
        )
        output = continuous_nnet.forward(mixture)  # [bs, samples, channels]
        loss = self.calc_loss(output, vocal)
        output = torch.transpose(output,2,1)
        os.makedirs(os.path.join(self.val_result_save_dir_step, str(fname)),exist_ok=True)
        save_wave((tensor2numpy(output) * 2 ** 15).astype(np.short),
                  fname=os.path.join(self.val_result_save_dir_step, str(fname), self.target + ".wav"))
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        # Use the default log function to gather info from gpus
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)



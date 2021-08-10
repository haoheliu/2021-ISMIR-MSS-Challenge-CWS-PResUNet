import sys
sys.path.append("/Users/admin/Documents/projects/music-demixing-challenge-starter-kit")

from models.no_v_kqq_multihead_v2_conv4.modules import *

import torch.utils
import torch.utils.data
import torch.nn.functional as F
from utils.f_helper import FDomainHelper
from torchlibrosa.stft import magphase
import numpy as np
import pytorch_lightning as pl
from torchlibrosa import STFT
import time
from models.config import Config
class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.loss = torch.nn.L1Loss()

    def __call__(self, output, target):
        return self.loss(output,target)

class BN_GRU(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,layer=1, bidirectional=False, batchnorm=True, dropout=0.0):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if(batchnorm):self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self,inputs):
        # (batch, 1, seq, feature)
        if(inputs.size()[1] != 1):
            inputs = inputs[:,0:1,...]
        if(self.batchnorm):inputs = self.bn(inputs)
        out,_ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)




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

    def __call__(self, output, target, alpha_t=0.85):
        wav_loss = self.l1(output, target)

        sp_loss = self.l1(
            self.f_helper.wav_to_spectrogram(output, eps=1e-8),
            self.f_helper.wav_to_spectrogram(target, eps=1e-8)
        )

        sp_loss /= math.sqrt(self.window_size)

        return alpha_t*wav_loss + (1-alpha_t)*sp_loss

class UNetResComplex_100Mb(pl.LightningModule):
    def __init__(self, channels, stem="", nsrc=1,subband=4, use_lsd_loss=False,
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
        center = True
        pad_mode = 'reflect'
        window = 'hann'
        activation = 'relu'
        momentum = 0.01
        freeze_parameters = True
        self.hop_size = 441
        self.center = True
        self.pad_mode = 'reflect'
        self.window_size = 2048
        self.window = 'hann'
        self.activation = 'relu'
        self.momentum = 0.01
        self.freeze_parameters = True
        self.use_lsd_loss = use_lsd_loss
        self.save_hyperparameters()
        self.nsrc = nsrc
        self.channels = channels
        self.lr = lr
        self.gamma = gamma
        self.subband = subband
        self.sample_rate = sample_rate
        self.batchsize = batchsize
        self.frame_length = frame_length

        # self.hparams['channels'] = 2
        # self.lsd_loss = get_loss_function("lsd")
        self.train_step = 0
        self.val_step = 0
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.val_result_save_dir = None
        self.val_result_save_dir_step = None
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}
        self.stem = stem
        self.f_helper = FDomainHelper(
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            pad_mode=pad_mode,
            window=window,
            freeze_parameters=freeze_parameters,
            subband=self.subband,
            root=Config.ROOT
        )

        if (subband == 8):
            self.bn0 = nn.BatchNorm2d(129, momentum=momentum)
        elif (subband == 4):
            self.bn0 = nn.BatchNorm2d(257, momentum=momentum)
        elif (subband == 2):
            self.bn0 = nn.BatchNorm2d(513, momentum=momentum)
        else:
            self.bn0 = nn.BatchNorm2d(1025, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes4(in_channels=channels * nsrc * subband, out_channels=32,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes4(in_channels=32, out_channels=64,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes4(in_channels=64, out_channels=128,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes4(in_channels=128, out_channels=256,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockRes4(in_channels=256, out_channels=384,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockRes4(in_channels=384, out_channels=384,
                                               downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = EncoderBlockRes4(in_channels=384, out_channels=384,
                                            downsample=(1, 1), activation=activation, momentum=momentum)
        self.conv_block8 = EncoderBlockRes4(in_channels=384, out_channels=384,
                                            downsample=(1, 1), activation=activation, momentum=momentum)
        self.conv_block9 = EncoderBlockRes4(in_channels=384, out_channels=384,
                                            downsample=(1, 1), activation=activation, momentum=momentum)
        self.conv_block10 = EncoderBlockRes4(in_channels=384, out_channels=384,
                                             downsample=(1, 1), activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockRes4(in_channels=384, out_channels=384,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockRes4(in_channels=384, out_channels=384,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes4(in_channels=384, out_channels=256,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes4(in_channels=256, out_channels=128,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes4(in_channels=128, out_channels=64,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes4(in_channels=64, out_channels=32,
                                               stride=(2, 2), activation=activation, momentum=momentum)
        self.bass_blocks = self.get_after_conv_block()
        self.drums_blocks = self.get_after_conv_block()
        self.other_blocks = self.get_after_conv_block()

        self.init_weights()
        self.lr_lambda = lambda step: self.get_lr_lambda(step,
                                                         gamma=self.gamma,
                                                         warm_up_steps=warm_up_steps,
                                                         reduce_lr_steps=reduce_lr_steps)
    def get_after_conv_block(self):
        li = nn.ModuleList()
        li.append(EncoderBlockRes4(in_channels=32, out_channels=32, downsample=(1, 1),
                                   activation=self.activation, momentum=self.momentum))
        li.append(nn.Conv2d(in_channels=32, out_channels=self.channels * self.nsrc * 4 * self.subband,
                            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
        return li

    def divide(self, data):
        """
        :param data: [batchsize, 8, samples]
        :return: Vocals: [batchsize, 2, samples], Bass: [batchsize, 2, samples], Drums, Others
        """
        return data[:,0:2,:], data[:,2:4,:], data[:,4:6,:]

    def after_conv_block_forward(self, blocks: nn.ModuleList, x):
        block1, block2 = blocks[0], blocks[1]
        x,_ = block1(x)
        return block2(x)

    def get_lr_lambda(self, step, gamma, warm_up_steps, reduce_lr_steps):
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
        after_conv_blocks = [self.bass_blocks, self.drums_blocks, self.other_blocks]
        init_bn(self.bn0)
        for block in after_conv_blocks:
            for b in block:
                if(type(b) == nn.Conv2d): init_layer(b)

    def forward(self, input, stem=None):
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

        res = []
        if(self.stem == "bass"):
            after_conv_blocks = [self.bass_blocks]
        elif (self.stem == "drums"):
            after_conv_blocks = [self.drums_blocks]
        elif (self.stem == "other"):
            after_conv_blocks = [self.other_blocks]
        else:
            after_conv_blocks = [self.bass_blocks, self.drums_blocks, self.other_blocks]

        for block in after_conv_blocks:
            x = self.after_conv_block_forward(block,x12)

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
            wav_out = F.pad(wav_out,(0,pad_tail))

            res.append(wav_out)
        res = torch.cat(res, dim=1)
        output_dict = {'wav': res}
        return output_dict


if __name__ == "__main__":
    # 1.02s
    model = UNetResComplex_100Mb(channels=2)
    wav = torch.randn((1, 2, 44100))
    start = time.time()
    out = model(wav)['wav']
    print(time.time() - start)
    print(out.size())
    print(out)






import sys

# import multiprocessing_logging

sys.path.append("/Users/admin/Documents/projects/arnold_workspace/src")
sys.path.append("/opt/tiger/lhh_arnold_base/arnold_workspace/src")

from task_music_source_separation.joint_separation.modules import *
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F

from torchlibrosa.stft import  magphase
from tools.pytorch.losses import get_loss_function
from tools.dsp.overlapadd_boxcar import LambdaOverlapAdd

from callbacks.base import *
from tools.file.wav import *
from tools.pytorch.pytorch_util import *
from tools.pytorch.losses import *
from task_music_source_separation.joint_separation.config import Config

class UNetResComplex_100Mb(pl.LightningModule):
    def __init__(self, channels, nsrc=1, use_lsd_loss=False,
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
        self.channels = channels
        self.lr = lr
        self.gamma = gamma

        self.sample_rate = sample_rate
        self.batchsize = batchsize
        self.frame_length = frame_length

        # self.hparams['channels'] = 2
        self.l1loss = get_loss_function("l1")
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
            subband=None,
            root=Config.ROOT
        )
        self.time_downsample_ratio = 2 ** 5
        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes4B(in_channels=channels, out_channels=32,
                                                kernel_size=(3, 3), downsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.encoder_block2 = EncoderBlockRes4B(in_channels=32, out_channels=64,
                                                kernel_size=(3, 3), downsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.encoder_block3 = EncoderBlockRes4B(in_channels=64, out_channels=128,
                                                kernel_size=(3, 3), downsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.encoder_block4 = EncoderBlockRes4B(in_channels=128, out_channels=256,
                                                kernel_size=(3, 3), downsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.encoder_block5 = EncoderBlockRes4B(in_channels=256, out_channels=384,
                                                kernel_size=(3, 3), downsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.encoder_block6 = EncoderBlockRes4B(in_channels=384, out_channels=384,
                                                kernel_size=(3, 3), downsample=(1, 2), activation=activation,
                                                momentum=momentum)
        self.conv_block7a = EncoderBlockRes4B(in_channels=384, out_channels=384,
                                              kernel_size=(3, 3), downsample=(1, 1), activation=activation,
                                              momentum=momentum)
        self.conv_block7b = EncoderBlockRes4B(in_channels=384, out_channels=384,
                                              kernel_size=(3, 3), downsample=(1, 1), activation=activation,
                                              momentum=momentum)
        self.conv_block7c = EncoderBlockRes4B(in_channels=384, out_channels=384,
                                              kernel_size=(3, 3), downsample=(1, 1), activation=activation,
                                              momentum=momentum)
        self.conv_block7d = EncoderBlockRes4B(in_channels=384, out_channels=384,
                                              kernel_size=(3, 3), downsample=(1, 1), activation=activation,
                                              momentum=momentum)
        self.decoder_block1 = DecoderBlockRes4B(in_channels=384, out_channels=384,
                                                kernel_size=(3, 3), upsample=(1, 2), activation=activation,
                                                momentum=momentum)
        self.decoder_block2 = DecoderBlockRes4B(in_channels=384, out_channels=384,
                                                kernel_size=(3, 3), upsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.decoder_block3 = DecoderBlockRes4B(in_channels=384, out_channels=256,
                                                kernel_size=(3, 3), upsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.decoder_block4 = DecoderBlockRes4B(in_channels=256, out_channels=128,
                                                kernel_size=(3, 3), upsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.decoder_block5 = DecoderBlockRes4B(in_channels=128, out_channels=64,
                                                kernel_size=(3, 3), upsample=(2, 2), activation=activation,
                                                momentum=momentum)
        self.decoder_block6 = DecoderBlockRes4B(in_channels=64, out_channels=32,
                                                kernel_size=(3, 3), upsample=(2, 2), activation=activation,
                                                momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes2(in_channels=32, out_channels=32,
                                                    downsample=(1, 1), activation=activation,
                                                   momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=channels * 4 * 4,
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
        r"""
        Args:
            input: (batch_size, channels_num, segment_samples)

        Outputs:
            output_dict: {
                'wav': (batch_size, channels_num, segment_samples),
                'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)
        # shapes: (batch_size, channels_num, time_steps, freq_bins)

        # batch normalization
        x = sp.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        # (batch_size, chanenls, time_steps, freq_bins)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = int(np.ceil(x.shape[2] / self.time_downsample_ratio)) \
                  * self.time_downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        # (batch_size, channels, padded_time_steps, freq_bins)

        # Let frequency bins be evenly divided by 2, e.g., 1025 -> 1024.
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        (N_, C_, T_, F_) = x.shape

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 384, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7a(x6_pool)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7b(x_center)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7c(x_center)  # (bs, 384, T / 32, F / 64)
        (x_center, _) = self.conv_block7d(x_center)  # (bs, 384, T / 32, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        (x, _) = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels * 3, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))  # Pad frequency, e.g., 1024 -> 1025.
        x = x[:, :, 0: origin_len, :]  # (bs, channels * 3, T, F)

        res = []
        for i in range(3):
            mask_mag1 = torch.sigmoid(x[:, i*8: i*8+2, :, :])
            _mask_real = x[:, i*8+2: i*8+4, :, :]
            _mask_imag = x[:, i*8+4: i*8+6, :, :]
            _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)

            mask_mag2 = x[:, i*8+6: i*8+8, :, :]

            # e^{jX + jM}
            out_cos = cos_in * mask_cos - sin_in * mask_sin
            out_sin = sin_in * mask_cos + cos_in * mask_sin

            # out_mag = sp * mask_mag
            out_mag = F.relu_(sp * mask_mag1 + mask_mag2)
            out_real = out_mag * out_cos
            out_imag = out_mag * out_sin

            length = input.shape[2]

            wav_out = torch.stack((
                self.f_helper.istft(out_real[:, 0: 1, :, :], out_imag[:, 0: 1, :, :], length),
                self.f_helper.istft(out_real[:, 1: 2, :, :], out_imag[:, 1: 2, :, :], length)),
                dim=1
            )
            res.append(wav_out)
        res = torch.cat(res,dim=1)
        output_dict = {'wav': res}
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

    def divide(self, data):
        """
        :param data: [batchsize, 8, samples]
        :return: Vocals: [batchsize, 2, samples], Bass: [batchsize, 2, samples], Drums, Others
        """
        return data[:,0:2,:], data[:,2:4,:], data[:,4:6,:]

    def preprocess(self, batch, train=False):
        if (train):
            bass = batch['bass'].float().permute(0, 2, 1)
            drums = batch['drums'].float().permute(0, 2, 1)
            other = batch['other'].float().permute(0, 2, 1)
            mixture = bass + drums + other
            return  bass, drums, other, mixture
        else:  # during test or validaton
            bass = batch['bass'].float().permute(0, 2, 1)
            drums = batch['drums'].float().permute(0, 2, 1)
            other = batch['other'].float().permute(0, 2, 1)
            mixture = bass + drums + other
            return  bass, drums, other, mixture, batch['fname'][0]  # a sample for a batch

    def info(self,string:str):
        lg.info("On trainer-" + str(self.trainer.global_rank) + ": " + string)

    def calc_loss(self, output, vocal, name: str):
        l1 = self.l1loss(output, vocal)
        self.log(name, l1, on_step=True, on_epoch=False, logger=True, sync_dist=True, prog_bar=True)
        return l1

    def training_step(self, batch, batch_nb):
        bass, drums, other, mixture = self.preprocess(batch, train=True)
        est_bass, est_drums, est_other = self.divide(self(mixture)['wav'])
        loss = self.calc_loss(est_other, other, "o")
        loss = loss + self.calc_loss(est_bass, bass, "b")
        loss = loss + self.calc_loss(est_drums, drums, "d")
        if(self.train_step > 10000):
            loss = loss + self.calc_loss(est_bass + est_drums + est_other, mixture, "m")
        self.log("All-loss", loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.train_step += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        bass, drums, other, mixture, fname = self.preprocess(batch)
        continuous_nnet = LambdaOverlapAdd(
            nnet=self,
            n_src=self.channels * self.nsrc * 3,
            window_size=self.sample_rate * 20,
            in_margin =int(self.sample_rate*1.5),
            window="boxcar",
            reorder_chunks=False,
            enable_grad=False,
            device=self.device
        )
        est_bass, est_drums, est_other = self.divide(continuous_nnet.forward(mixture))

        loss = self.calc_loss(est_other, other, "o")
        loss = loss + self.calc_loss(est_bass, bass, "b")
        loss = loss + self.calc_loss(est_drums, drums, "d")
        loss = loss + self.calc_loss(est_bass + est_drums + est_other, mixture, "m")

        est_bass = torch.transpose(est_bass,2,1)
        est_other = torch.transpose(est_other,2,1)
        est_drums = torch.transpose(est_drums,2,1)

        save_wave((tensor2numpy(est_bass) * 2 ** 15).astype(np.short),
                  fname=os.path.join(self.val_result_save_dir_step, str(fname) + "bass.wav"))
        save_wave((tensor2numpy(est_drums) * 2 ** 15).astype(np.short),
                  fname=os.path.join(self.val_result_save_dir_step, str(fname) + "drums.wav"))
        save_wave((tensor2numpy(est_other) * 2 ** 15).astype(np.short),
                  fname=os.path.join(self.val_result_save_dir_step, str(fname) + "other.wav"))
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        # Use the default log function to gather info from gpus
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)


if __name__ == "__main__":
    # 1.02s
    model = UNetResComplex_100Mb(channels=2)
    wav = torch.randn((1, 2, 44100))
    start = time.time()
    out = model(wav)['wav']
    print(time.time() - start)
    print(out.size())
    print(out)



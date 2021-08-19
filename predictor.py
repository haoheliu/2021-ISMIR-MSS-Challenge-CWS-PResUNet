import concurrent.futures
import os
import time

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from models.resunet_conv8_vocals.model import UNetResComplex_100Mb as Conv8Res
from models.resunet_joint_training_other.model import UNetResComplex_100Mb as NO_V_multihead_Conv4
from demucs_predictor import DemucsPredictor
from utils.overlapadd_singlethread_exclude_vocal import LambdaOverlapAdd as Exclude_Vocal_LambdaOverlapAdd
from utils.overlapadd_singlethread import LambdaOverlapAdd
from utils.filtering import delete_band
MARGIN = int(44100*1.5)

def divide_stems(data):
    """
    :param data: [batchsize, 8, samples]
    :return: Vocals: [batchsize, 2, samples], Bass: [batchsize, 2, samples], Drums, Others
    """
    return data[:,0:2], data[:,2:4], data[:,4:6]

def pre(x):
    x = torch.FloatTensor(x)
    return x.permute(1, 0)[None, ...]

def post(y):
    if(y is None): return None
    if(y.is_cuda):
        y = y.detach().cpu()
    return y[0, ...].permute(1, 0).numpy()

class SubbandResUNetPredictor():
    """Lower baseline of using `1/4 * mixture` as prediction for bass, drums, other and vocals."""
    def __init__(self, cuda=True, sources=[]):
        if(cuda and not torch.cuda.is_available()):
            print("Warning: You choose to use GPU but no CUDA device is found by pytorch.")
            time.sleep(2)
        self.use_gpu = cuda
        self.sources = sources
        if(self.use_gpu):
            print("Using GPU Accelerations")

    def prediction_setup(self):
        # print("Setting up")
        """Initialize predictor."""
        self.vocal_result_cache={}
        if ("bass" in self.sources or "drums" in self.sources):
            self.demucs = DemucsPredictor(use_gpu=self.use_gpu,sources=self.sources)
            self.demucs.prediction_setup()

        v_model_path = "models/resunet_conv8_vocals/checkpoints/vocals/epoch=49-val_loss=0.0902_trimed.ckpt"
        o_model_path = "models/resunet_joint_training_other/checkpoints_nov/other/epoch=33-val_loss=0.4293_trimed.ckpt"

        os.makedirs(os.path.dirname(v_model_path),exist_ok=True)
        os.makedirs(os.path.dirname(o_model_path),exist_ok=True)

        if (not os.path.exists(v_model_path) and "vocals" in self.sources):
            print("Downloading the weight of model for the vocal track")
            cmd = "wget https://zenodo.org/record/5175846/files/epoch%3D49-val_loss%3D0.0902_trimed.ckpt?download=1 -O "+ v_model_path
            print(cmd)
            os.system(cmd)
        if(not os.path.exists(o_model_path) and "other" in self.sources):
            print("Downloading the weight of model for the other track")
            cmd = "wget https://zenodo.org/record/5175846/files/epoch%3D33-val_loss%3D0.4293_trimed.ckpt?download=1 -O " + o_model_path
            print(cmd)
            os.system(cmd)

        if("vocals" in self.sources):
            print("Loading vocal model...")
            self.v_model = self.reload(v_model_path, Conv8Res(channels=2, target="vocals"), nsrc=2)
            if (self.use_gpu): self.v_model = self.v_model.cuda()
        if ("other" in self.sources):
            print("Loading other model...")
            self.o_model = self.reload(o_model_path, NO_V_multihead_Conv4(channels=2),stem="other", nsrc=2)
            if(self.use_gpu): self.o_model = self.o_model.cuda()

    def reload(self, pth:str, model: pl.LightningModule, nsrc: int, stem=None):
        model = model.eval()
        model = model.load_from_checkpoint(pth) if (len(pth) != 0) else model
        if(stem is not None):
            model.stem = stem
        if(self.sources == ['other']): # do not exclude vocal
            return LambdaOverlapAdd(
                nnet=model,
                n_src=nsrc,
                window_size=44100 * 10,
                in_margin=MARGIN,
                window="boxcar",
                reorder_chunks=False,
                enable_grad=False,
            ).eval()
        else:
            return Exclude_Vocal_LambdaOverlapAdd(
            nnet=model,
            n_src=nsrc,
            window_size=44100*10,
            in_margin=MARGIN,
            vocal_cache=self.vocal_result_cache,
            window="boxcar",
            reorder_chunks=False,
            enable_grad=False,
        ).eval()

    def sep(self, x, type: str):
        if(self.use_gpu):
            x = x.cuda()
        if("vocals" in type):
            return self.v_model(x, type=type), type
        elif("other" in type):
            return self.o_model(x, type=type), type

    def divide(self, x, threads):
        seg = x.shape[0] // threads
        mid_points = [0]
        segments = []
        for i in range(1, threads):
            mid_points.append(seg * i)
        for i in range(len(mid_points)):
            if(i == 0):
                segments.append(x[mid_points[0] : mid_points[0] + seg + MARGIN,...])
            elif(i == len(mid_points) - 1):
                segments.append(x[mid_points[-1]-MARGIN: ,...])
            else:
                segments.append(x[mid_points[i] - MARGIN: mid_points[i+1] + MARGIN, ...])
        for i in range(len(segments)):
            segments[i] = pre(segments[i])
        return segments, seg

    def trim_and_concatenate(self, res: dict, key: str, seg_length: int):
        members = []
        ret_val = []
        for _key in res.keys():
            if(key in _key): members.append(_key)
        members = sorted(members)
        for i in range(len(members)):
            if(i == 0):
                ret_val.append(res[members[0]][:seg_length,...])
            elif(i == len(members) - 1):
                ret_val.append(res[members[i]][MARGIN:, ...])
            else:
                ret_val.append(res[members[i]][MARGIN:-MARGIN, ...])
        return np.concatenate(ret_val)

    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        """Perform prediction."""
        # print("Mixture file is present at following location: %s" % mixture_file_path)
        x, rate = sf.read(mixture_file_path)  # (12002484, 2) mixture is stereo with sample rate of 44.1kHz
        if(len(x.shape) == 1):
            print("Warning: Processing audio with only one channel")
            x = np.concatenate([x[...,None],x[...,None]],axis=1)
        if (x.shape[1] == 1):
            x = np.concatenate([x[...], x[...]], axis=1)

        segments_v, seg_length_v = self.divide(x, threads=2)
        proc = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            if("bass" in self.sources or "drums" in self.sources):
                p = executor.submit(self.demucs.prediction,mixture_file_path,bass_file_path, drums_file_path, other_file_path, vocals_file_path)
                proc.append(p)
            for type in self.sources:
                if(type == "bass" or type == "drums"): continue # skip, use demucs for these two sources
                for i in range(len(segments_v)):
                    p = executor.submit(self.sep, segments_v[i], type+"_"+str(i))
                    proc.append(p)
        res = {}

        for i, f in enumerate(concurrent.futures.as_completed(proc)):
            result, t = f.result()
            result = post(result)
            res[t] = result

        if ("other" in self.sources):
            other = self.trim_and_concatenate(res,key="other",seg_length=seg_length_v)
            sf.write(other_file_path, other, rate)
            delete_band(other_file_path)
        if ("vocals" in self.sources):
            vocals = self.trim_and_concatenate(res,key="vocals",seg_length=seg_length_v)
            sf.write(vocals_file_path, vocals, rate)



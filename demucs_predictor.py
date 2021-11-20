

import time

import torch.hub
import torch
import torchaudio as ta

from demucs import pretrained
from demucs.apply import apply_model

class DemucsPredictor():

    def __init__(self, use_gpu=True, sources=[]):
        self.use_gpu = use_gpu
        self.sources = sources

    def prediction_setup(self):
        # Load your model here and put it into `evaluation` mode
        torch.hub.set_dir('./models/')

        # Use a pre-trained model
        self.separator = pretrained.get_model(name='mdx')
        self.separator.eval()
        if(self.use_gpu):
            self.separator = self.separator.cuda()

    def prediction(
        self,
        mixture_file_path,
        bass_file_path,
        drums_file_path,
        other_file_path,
        vocals_file_path,
    ):

        # Load mixture
        mix, sr = ta.load(str(mixture_file_path))
        assert sr == self.separator.samplerate
        assert mix.shape[0] == self.separator.audio_channels

        b = time.time()
        # Normalize track
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
        # Separate
        if(self.use_gpu): mix = mix.cuda()
        with torch.no_grad():
            estimates = apply_model(self.separator, mix[None], overlap=0.15)[0]
        estimates = estimates * std + mean

        # Store results
        target_file_map = {}
        if("drums" in self.sources): target_file_map["drums"] = drums_file_path
        if("bass" in self.sources): target_file_map["bass"] = bass_file_path
        for target, path in target_file_map.items():
            idx = self.separator.sources.index(target)
            source = estimates[idx]
            mx = source.abs().max()
            if mx >= 1:
                print('clipping', target, mx, std)
            source = source.clamp(-0.99, 0.99)
            if(source.is_cuda):
                source = source.detach().cpu()
            ta.save(str(path), source, sample_rate=sr)
        return None, "bass_and_drums"



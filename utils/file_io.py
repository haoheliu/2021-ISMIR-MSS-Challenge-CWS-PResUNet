import wave
import os
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import json
from pathlib import Path
import yaml
import pickle


def read_list(fname):
    result = []
    with open(fname, "r") as f:
        for each in f.readlines():
            each = each.strip('\n')
            result.append(each)
    return result


def write_list(list, fname):
    with open(fname, 'w') as f:
        for word in list:
            f.write(word)
            f.write('\n')


def write_json(my_dict, fname):
    # print("Save json file at "+fname)
    json_str = json.dumps(my_dict)
    with open(fname, 'w') as json_file:
        json_file.write(json_str)


def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
        return data


def save_pickle(obj, fname):
    # print("Save pickle at "+fname)
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    # print("Load pickle at "+fname)
    with open(fname, 'rb') as f:
        res = pickle.load(f)
    return res


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)


def tensor2numpy(tensor):
    if ("cuda" in str(tensor.device)):
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def save_wave(frames: np.ndarray, fname, sample_rate=44100):
    shape = list(frames.shape)
    if (len(shape) == 1):
        frames = frames[..., None]
    in_samples, in_channels = shape[-2], shape[-1]
    if (in_channels >= 3):
        if (len(shape) == 2):
            frames = np.transpose(frames, (1, 0))
        elif (len(shape) == 3):
            frames = np.transpose(frames, (0, 2, 1))
        msg = "Warning: Save audio with " + str(in_channels) + " channels, save permute audio with shape " + str(
            list(frames.shape)) + " please check if it's correct."
        # print(msg)
    if (np.max(frames) <= 1 and frames.dtype == np.float32 or frames.dtype == np.float16 or frames.dtype == np.float64):
        frames *= 2 ** 15
    frames = frames.astype(np.short)
    if (len(frames.shape) >= 3):
        frames = frames[0, ...]
    sf.write(fname, frames, samplerate=sample_rate)


def constrain_length(chunk, length):
    frames_length = chunk.shape[0]
    if (frames_length == length):
        return chunk
    elif (frames_length < length):
        return np.pad(chunk, ((0, int(length - frames_length)), (0, 0)), 'constant')
    else:
        return chunk[:length, ...]


def random_chunk_wav_file(fname, chunk_length):
    '''
    fname: path to wav file
    chunk_length: frame length in seconds
    '''
    with wave.open(fname) as f:
        params = f.getparams()
        duration = params[3] / params[2]
        sample_rate = params[2]
        sample_length = params[3]
        if (duration < chunk_length or abs(duration - chunk_length) < 1e-4):
            frames = read_wave(fname, sample_rate)
            return frames, duration, sample_rate  # [-1,1]
        else:
            # Random trunk
            random_starts = np.random.randint(0, sample_length - sample_rate * chunk_length)
            random_end = (random_starts + sample_rate * chunk_length)
            random_starts, random_end = random_starts / sample_rate, random_end / sample_rate
            random_starts, random_end = random_starts / duration, random_end / duration
            frames = read_wave(fname, sample_rate, portion_start=random_starts, portion_end=random_end)
            frames = constrain_length(frames, length=int(chunk_length * sample_rate))
            return frames, chunk_length, sample_rate


def random_chunk_wav_file_v2(fname, chunk_length, random_starts=None, random_end=None):
    '''
    fname: path to wav file
    chunk_length: frame length in seconds
    '''
    with wave.open(fname) as f:
        params = f.getparams()
        duration = params[3] / params[2]
        sample_rate = params[2]
        sample_length = params[3]
        if (duration < chunk_length or abs(duration - chunk_length) < 1e-4):
            frames = read_wave(fname, sample_rate)
            return frames, duration, sample_rate  # [-1,1]
        else:
            # Random trunk
            if (random_starts is None and random_end is None):
                random_starts = np.random.randint(0, sample_length - sample_rate * chunk_length)
                random_end = (random_starts + sample_rate * chunk_length)
                random_starts, random_end = random_starts / sample_rate, random_end / sample_rate
                random_starts, random_end = random_starts / duration, random_end / duration
            frames = read_wave(fname, sample_rate, portion_start=random_starts, portion_end=random_end)
            frames = constrain_length(frames, length=int(chunk_length * sample_rate))
            return frames, chunk_length, sample_rate, random_starts, random_end


def read_wave(fname,
              sample_rate,
              portion_start=0,
              portion_end=1,
              ):  # Whether you want raw bytes
    """
    :param fname: wav file path
    :param sample_rate:
    :param portion_start:
    :param portion_end:
    :return: [sample, channels]
    """
    # sr = get_sample_rate(fname)
    # if(sr != sample_rate):
    #     print("Warning: Sample rate not match, may lead to unexpected behavior.")
    if (portion_end > 1 and portion_end < 1.1):
        portion_end = 1
    if (portion_end != 1):
        duration = get_duration(fname)
        wav, _ = librosa.load(fname, sr=sample_rate,
                              offset=portion_start * duration,
                              duration=(portion_end - portion_start) * duration,
                              mono=False)
    else:
        wav, _ = librosa.load(fname, sr=sample_rate, mono=False)
    if (len(list(wav.shape)) == 1):
        wav = wav[..., None]
    else:
        wav = np.transpose(wav, (1, 0))
    return wav


def get_channels_sampwidth_and_sample_rate(fname):
    if (os.path.islink(fname)): fname = os.readlink(fname)
    with wave.open(fname) as f:
        params = f.getparams()
    return params[0], params[1], params[2]  # == (2,2,44100),(params[0],params[1],params[2])


def get_channels(fname):
    if (os.path.islink(fname)): fname = os.readlink(fname)
    with wave.open(fname) as f:
        params = f.getparams()
    return params[0]


def get_sample_rate(fname):
    if (os.path.islink(fname)): fname = os.readlink(fname)
    with wave.open(fname) as f:
        params = f.getparams()
    return params[2]


def get_duration(fname):
    if (os.path.islink(fname)): fname = os.readlink(fname)
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3] / params[2]


def get_framesLength(fname):
    if(os.path.islink(fname)): fname = os.readlink(fname)
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3]


def restore_wave(zxx):
    _, w = signal.istft(zxx)
    return w


def calculate_total_times(dir):
    total = 0
    for each in os.listdir(dir):
        fname = os.path.join(dir, each)
        try:
            duration = get_duration(fname)
        except:
            print(fname)
        total += duration
    return total


def filter(pth):
    global dic
    temp = []
    for each in os.listdir(pth):
        temp.append(os.path.join(pth, each))
    for each in temp:
        sr = get_sample_rate(each)
        if (sr not in dic.keys()):
            dic[sr] = []
        dic[sr].append(each)
    for each in dic[16000]:
        # print(each)
        pass
    print(dic.keys())
    for each in list(dic.keys()):
        print(each, len(dic[each]))

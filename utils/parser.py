import sys
from file_io import *
from progressbar import *

def find_and_build(root,path):
    path = os.path.join(root, path)
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path

def merge(li:list,target:str):
    res = None
    for i in li:
        if(res is None):
            res = read_wave(i, sample_rate=44100)
        else:
            res += read_wave(i, sample_rate=44100)
    save_wave(res,fname=target,sample_rate=44100)

r = os.getcwd() # todo
ROOT = os.path.join(r,"data/musdb18hq")
DATA = os.path.join(r,"data/meta")

find_and_build("",ROOT)
find_and_build("",DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "musdb18hq")

find_and_build(SOFTLINKSAVEDIR, "")

data = {
    "test":{
        "fname":[], "vocals":[], "drums":[], "bass":[], "other":[], "mixture":[], "acc":[], "no_bass":[], "no_other":[], "no_drums":[]
    },
    "train":{
        "fname":[], "vocals":[], "drums":[], "bass":[], "other":[], "mixture":[], "acc":[], "no_bass":[], "no_other":[], "no_drums":[]
    }
}

SubDir = os.path.join(ROOT,"test")
test = find_and_build(SOFTLINKSAVEDIR, "test")
widgets = [
    "Preparing MUSDB18HQ Test Set",
    ' [', Timer(), '] ',
    Bar(),
    ' (', ETA(), ') ',
]
pbar = ProgressBar(widgets=widgets).start()
files = os.listdir(SubDir)
for i, each in enumerate(files):
    if(each == ".DS_Store"): continue
    # datas['test']['fname'].append(os.path.join(SubDir,each))
    data['test']['vocals'].append(os.path.join(SubDir,each,"vocals.wav"))
    data['test']['drums'].append(os.path.join(SubDir,each,"drums.wav"))
    data['test']['bass'].append(os.path.join(SubDir,each,"bass.wav"))
    data['test']['other'].append(os.path.join(SubDir,each,"other.wav"))
    data['test']['mixture'].append(os.path.join(SubDir,each,"mixture.wav"))
    if(True or not os.path.exists(os.path.join(SubDir,each,"acc.wav"))):
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"acc.wav"))
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"vocals.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"no_bass.wav"))
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"vocals.wav")],target=os.path.join(SubDir,each,"no_other.wav"))
        merge([os.path.join(SubDir,each,"vocals.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"no_drums.wav"))
    data['test']['acc'].append(os.path.join(SubDir, each, "acc.wav"))
    data['test']['no_bass'].append(os.path.join(SubDir, each, "no_bass.wav"))
    data['test']['no_other'].append(os.path.join(SubDir, each, "no_other.wav"))
    data['test']['no_drums'].append(os.path.join(SubDir, each, "no_drums.wav"))
    pbar.update(int((i / (len(files) - 1)) * 100))

write_list(data['test']['vocals'],os.path.join(test,"vocals.lst"))
write_list(data['test']['drums'],os.path.join(test,"drums.lst"))
write_list(data['test']['bass'],os.path.join(test,"bass.lst"))
write_list(data['test']['other'],os.path.join(test,"other.lst"))
write_list(data['test']['mixture'],os.path.join(test,"mixture.lst"))
write_list(data['test']['acc'],os.path.join(test,"acc.lst"))
write_list(data['test']['no_bass'],os.path.join(test,"no_bass.lst"))
write_list(data['test']['no_other'],os.path.join(test,"no_other.lst"))
write_list(data['test']['no_drums'],os.path.join(test,"no_drums.lst"))

widgets = [
    "Preparing MUSDB18HQ Train Set",
    ' [', Timer(), '] ',
    Bar(),
    ' (', ETA(), ') ',
]
SubDir = os.path.join(ROOT,"train")
train = find_and_build(SOFTLINKSAVEDIR, "train")
pbar = ProgressBar(widgets=widgets).start()
files = os.listdir(SubDir)
for i, each in enumerate(files):
    if(each == ".DS_Store"): continue
    # datas['train']['fname'].append(os.path.join(SubDir,each))
    data['train']['vocals'].append(os.path.join(SubDir,each,"vocals.wav"))
    data['train']['drums'].append(os.path.join(SubDir,each,"drums.wav"))
    data['train']['bass'].append(os.path.join(SubDir,each,"bass.wav"))
    data['train']['other'].append(os.path.join(SubDir,each,"other.wav"))
    data['train']['mixture'].append(os.path.join(SubDir,each,"mixture.wav"))
    if(True or not os.path.exists(os.path.join(SubDir,each,"acc.wav"))):
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"acc.wav"))
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"vocals.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"no_bass.wav"))
        merge([os.path.join(SubDir,each,"drums.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"vocals.wav")],target=os.path.join(SubDir,each,"no_other.wav"))
        merge([os.path.join(SubDir,each,"vocals.wav"),
               os.path.join(SubDir,each,"bass.wav"),
               os.path.join(SubDir,each,"other.wav")],target=os.path.join(SubDir,each,"no_drums.wav"))
    data['train']['acc'].append(os.path.join(SubDir,each,"acc.wav"))
    data['train']['no_bass'].append(os.path.join(SubDir, each, "no_bass.wav"))
    data['train']['no_other'].append(os.path.join(SubDir, each, "no_other.wav"))
    data['train']['no_drums'].append(os.path.join(SubDir, each, "no_drums.wav"))
    pbar.update(int((i / (len(files) - 1)) * 100))

write_list(data['train']['vocals'],os.path.join(train,"vocals.lst"))
write_list(data['train']['drums'],os.path.join(train,"drums.lst"))
write_list(data['train']['bass'],os.path.join(train,"bass.lst"))
write_list(data['train']['other'],os.path.join(train,"other.lst"))
write_list(data['train']['mixture'],os.path.join(train,"mixture.lst"))
write_list(data['train']['acc'],os.path.join(train,"acc.lst"))
write_list(data['train']['no_bass'],os.path.join(train,"no_bass.lst"))
write_list(data['train']['no_other'],os.path.join(train,"no_other.lst"))
write_list(data['train']['no_drums'],os.path.join(train,"no_drums.lst"))

print("")



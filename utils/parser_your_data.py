# import git
# git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")

import os
from file_io import *
from progressbar import *

def find_and_build(root,path):
    path = os.path.join(root, path)
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path

def merge(li:list,target:str):
    if(os.path.exists(target)): return
    res = None
    for i in li:
        if(res is None):
            res = read_wave(i, sample_rate=44100)
        else:
            res += read_wave(i, sample_rate=44100)
    save_wave(res,fname=target,sample_rate=44100)

def ln(arg1, arg2):
    cmd = "ln -s "+ arg1+" "+ arg2
    # print(cmd)
    os.system(cmd)

def unify_format(files):
    pbar = ProgressBar(widgets=widgets).start()
    for i,f in enumerate(files):
        cmd = "sox  "+ f + " -c2 -r 44100 -b 16 "+os.path.join(os.path.dirname(f),os.path.splitext(os.path.basename(f))[0])+"_.wav "
        os.system(cmd)
        os.remove(f)
        # print(cmd)
        cmd = "mv "+ os.path.join(os.path.dirname(f),os.path.splitext(os.path.basename(f))[0])+"_.wav "+os.path.join(os.path.dirname(f),os.path.splitext(os.path.basename(f))[0])+".wav "
        # print(cmd)
        os.system(cmd)
        pbar.update(int((i / (len(files)+1)) * 100))

widgets = [
    "Preparing Data Set",
    ' [', Timer(), '] ',
    Bar(),
    ' (', ETA(), ') ',
]

def build_dataset(source_dir, target_dir):
    target_source_dir = os.path.join(source_dir,"the_source_you_want_to_get")
    other_sources_dir = os.path.join(source_dir, "the_source_you_want_to_remove")
    target_files = [os.path.join(target_source_dir,x) for x in os.listdir(target_source_dir)]
    other_files = [os.path.join(other_sources_dir, x) for x in os.listdir(other_sources_dir)]
    unify_format(target_files)
    unify_format(other_files)
    target_files = [os.path.join(target_source_dir, x) for x in os.listdir(target_source_dir)]
    other_files = [os.path.join(other_sources_dir, x) for x in os.listdir(other_sources_dir)]
    count = 0
    while(len(target_files) < len(other_files)):
        target_files += target_files[count % len(target_files): count % len(target_files)+1]; count += 1
    while(len(target_files) > len(other_files)):
        other_files += other_files[count % len(other_files): count % len(other_files) + 1]; count += 1
    for i in range(len(target_files)):
        link_dir = os.path.join(target_dir, str(i))
        os.makedirs(link_dir, exist_ok=True)
        ln(target_files[i],os.path.join(link_dir,"vocals.wav"))
        ln(other_files[i], os.path.join(link_dir, "acc.wav"))
        ln(other_files[i], os.path.join(link_dir, "drums.wav"))
        ln(other_files[i], os.path.join(link_dir, "bass.wav"))
        ln(other_files[i], os.path.join(link_dir, "other.wav"))
        # placeholders, no effects
        ln(other_files[i], os.path.join(link_dir, "no_bass.wav"))
        ln(other_files[i], os.path.join(link_dir, "no_other.wav"))
        ln(other_files[i], os.path.join(link_dir, "no_drums.wav"))
        ln(other_files[i], os.path.join(link_dir, "mixture.wav"))

r = os.getcwd() # todo
ROOT = os.path.join(r,"data/your_data_parsed")
os.makedirs(ROOT,exist_ok=True)
os.makedirs(os.path.join(ROOT,"test"),exist_ok=True)
os.makedirs(os.path.join(ROOT,"train"),exist_ok=True)
DATA = os.path.join(r,"data/meta")

build_dataset(os.path.join(r,"data/your_data/train"), os.path.join(r,"data/your_data_parsed/train"))
build_dataset(os.path.join(r,"data/your_data/test"), os.path.join(r,"data/your_data_parsed/test"))

find_and_build("",ROOT)
find_and_build("",DATA)

SOFTLINKSAVEDIR = os.path.join(DATA, "musdb18hq")

os.system("rm -r "+SOFTLINKSAVEDIR)

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



import sys
import os
sys.path.append(os.getcwd())
import threading
import time
import glob
import numpy as np
from utils.file_io import *
from argparse import ArgumentParser
import speechmetrics as sm

MAX_THREAD = 3

root = os.getcwd()
MUSDB_TEST = "data/musdb18hq/test"

def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

metrics_bsseval = sm.load(["bsseval"], window=1)  # [samples, channels]
metrics_sisdr = sm.load(["sisdr"], window=np.inf)  # [samples, channels]

parser = ArgumentParser()
parser.add_argument('--step', type=str, default="", help="A fold contain a validation step results")
parser.add_argument('--path', type=str, default="", help="A fold contain all validation step results (step's super dir)")
parser.add_argument('--type', type=str, default="", help="Evaluation data type")
args = parser.parse_args()

type = args.type

if(len(args.step) == 0 and len(args.path) == 0):
    raise RuntimeError("step argument and path argument at least should have one non-empty.")

# def unify_energy(est, target):
#     max_est, max_target = np.max(np.abs(est)), np.max(np.abs(target))
#     ratio = max_est/max_target
#     return est/ratio, target


def evaluate_file(target, est, step_dir, fname):
    """
    :param target: target .wav file absolute path
    :param est: est .wav file absolute path
    :param step_dir: path to validation step resuts, absolute path
    :param fname: file name to evaluate, not path
    :return:
    """
    if(not os.path.exists(os.path.join(step_dir, fname + type + ".pkl"))):
        target = read_wave(target, sample_rate=44100)
        est = read_wave(est, sample_rate=44100)
        est, target = est.astype(np.float32), target.astype(np.float32)
        eval_bsseval = metrics_bsseval(est, target,rate=44100)
        eval_sisdr = metrics_sisdr(est, target, rate=44100)
        eval_sdr = sdr(target[None,...],est[None,...])
        for k in eval_sisdr.keys():
            eval_bsseval[k] = eval_sisdr[k]
        eval_bsseval['sdr_ismir'] = eval_sdr
        save_pickle(eval_bsseval, os.path.join(step_dir, fname, type + ".pkl"))
    else:
        eval_bsseval = load_pickle(os.path.join(step_dir, fname, type + ".pkl"))
    print(os.getpid(),fname," - Score: ", [(each, np.nanmedian(eval_bsseval[each])) for each in eval_bsseval.keys()])
    return eval_bsseval

def evaluate_step(step_dir, files:list):
    """
    :param step_dir: path to validation step resuts, absolute path
    :param files: list of file names
    :return:
    """
    t = threading.current_thread()
    for file in files:
        try:
            evaluate_file(os.path.join(root,MUSDB_TEST,file,type+".wav"), os.path.join(step_dir, file, type+".wav"), step_dir, file)
        except Exception as e:
            print(os.path.join(step_dir, file, type+".wav"),"not found, skip this file.")


def evaluate_step_multiprocess(step_dir):
    """
    :param step_dir: path to validation step resuts, absolute path
    :return:
    """
    files = os.listdir(os.path.join(root, MUSDB_TEST))
    todos = divide_list(files, MAX_THREAD)
    threads = []
    for each in todos:
        threads.append(threading.Thread(target=evaluate_step, args=(step_dir, each)))
    for t in threads:
        print("Start: ", t.ident, t.getName())
        t.setDaemon(True)
        t.start()
    while True:
        time.sleep(0.2)
        status = []
        for each in threads:
            status.append(each.is_alive())
        if (True in status):
            continue
        else:
            print("Done!")
            break
    print("Start aggregating scores...")
    aggregate_thread_results(step_dir=step_dir)

def aggregate_thread_results(step_dir):
    eval = []
    for fname in glob.glob(os.path.join(step_dir,"*"+type+".pkl")):
        bsseval = load_pickle(os.path.join(step_dir,fname))
        eval.append(bsseval)
    aggregate = aggregate_score(eval)
    print(aggregate)
    write_json(aggregate, os.path.join(step_dir, "evaluation_result_"+type+".json"))
    print("Done")

def aggregate_score(outputs):
    """
    [{ # first test element (from test dataloader) result
            "<metrics>": <value>
    }]
    """
    # Calculate metrics
    res, metrics = {}, []
    for element in outputs:  # each validation sample
        if (len(metrics) == 0):  # collect all_mel_e2e metrics ( from the output of validation step )
            for k in element.keys(): metrics.append(k)
        for m in metrics:  # calculate median for a song
            element[m] = np.nanmedian(element[m])
    for m in metrics:
        res[m] = float(np.median([x[m] for x in outputs]))
    return res

def evaluate_path(path):
    files = os.listdir(os.path.join(root, MUSDB_TEST))
    for step in os.listdir(path):
        step_dir = os.path.join(path,step)
        print("EVALUATE PATH: ",step_dir)
        if(os.path.isdir(step_dir)):
            evaluate_step_multiprocess(step_dir)

def divide_list(li,thread_num):
    range_ = np.linspace(0,len(li),thread_num+1)
    res = []
    start,end = None,None
    for each in range(range_.shape[0]):
        if(each + 1 == range_.shape[0]):
            break
        start,end = int(range_[each]),int(range_[each+1])
        res.append(li[start:end])
    return res

if __name__ == "__main__":
    if(len(args.step) != 0):
        evaluate_step_multiprocess(args.step)

    if(len(args.path) != 0):
        evaluate_path(path=args.path)


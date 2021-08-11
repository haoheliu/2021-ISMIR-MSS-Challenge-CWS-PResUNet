import os
import os.path as op

class Config:
    ROOT = os.getcwd()
    TRAIL_NAME = os.environ['TRAIL_NAME']

    # DATA
    train_data = {
        "vocals": {
            "musdb18hq": "data/meta/musdb18hq/train/vocals.lst"
        },
        "bass": {
            "musdb18hq": "data/meta/musdb18hq/train/bass.lst"
        },
        "drums": {
            "musdb18hq": "data/meta/musdb18hq/train/drums.lst"
        },
        "other": {
            "musdb18hq": "data/meta/musdb18hq/train/other.lst"
        },
        "acc": {
            "musdb18hq": "data/meta/musdb18hq/train/acc.lst"
        },
        "no_bass": {
            "musdb18hq": "data/meta/musdb18hq/train/no_bass.lst"
        },
        "no_drums": {
            "musdb18hq": "data/meta/musdb18hq/train/no_drums.lst"
        },
        "no_other": {
            "musdb18hq": "data/meta/musdb18hq/train/no_other.lst"
        },
    }

    test_data = {
        "vocals": {
            "musdb18hq": "data/meta/musdb18hq/test/vocals.lst"
        },
        "bass": {
            "musdb18hq": "data/meta/musdb18hq/test/bass.lst"
        },
        "drums": {
            "musdb18hq": "data/meta/musdb18hq/test/drums.lst"
        },
        "other": {
            "musdb18hq": "data/meta/musdb18hq/test/other.lst"
        },
        "acc": {
            "musdb18hq": "data/meta/musdb18hq/test/acc.lst"
        },
        "no_bass": {
            "musdb18hq": "data/meta/musdb18hq/test/no_bass.lst"
        },
        "no_drums": {
            "musdb18hq": "data/meta/musdb18hq/test/no_drums.lst"
        },
        "no_other": {
            "musdb18hq": "data/meta/musdb18hq/test/no_other.lst"
        },
    }

    for k in train_data.keys():
        for k2 in train_data[k].keys():
            train_data[k][k2] = op.join(ROOT,train_data[k][k2])

    for k in test_data.keys():
        for k2 in test_data[k].keys():
            test_data[k][k2] = op.join(ROOT,test_data[k][k2])
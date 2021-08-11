# Music Source Separation with Channel-wise Subband ResUnet (CWS-ResUNet)

You can use this repo to separate 'bass', 'drums', 'vocals', and 'other' tracks from a music mixture. This repo contains the pretrained Music Source Separation models I submitted to the [2021 ISMIR MSS Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).
We only participate the Leaderboard A, so these models are only trained on MUSDB18HQ. 

As is shown in the following picture, in leaderboard A, we(ByteMSS) achieved the 2nd on Vocal score and 5th on average score.
We will open-source our training pipline soon.

![ranking](pics/ranks.png)

[comment]: <> (We use the following stratagy in this challenges)

[comment]: <> (1. Train models for the four stems &#40;vocals, bass, drums, and other&#41; separately.)

[comment]: <> (2. Use boxcar window to cut down the computation &#40;less overlap between windows&#41;.)

[comment]: <> (3. Separate vocal track first, substract it from the mixture, and use the remaining part to separate other stems.)

[comment]: <> (4. Since our final bass and drums score is still low. We directly use the open-source demucs model as the final submission for these two tracks.)

For bass and drums separation, we directly use [the open-sourced demucs model](https://github.com/facebookresearch/demucs). It's trained with only MUSDB18HQ data, thus is qualified for LeaderBoard A.

## Usage
### Prepare running environment
First you need to clone this repo:
```shell
git clone https://github.com/haoheliu/CWS-ResUNet-MSS-Challenge-ISMIR-2021.git
```
Install the required packages
```shell
cd CWS-ResUNet-MSS-Challenge-ISMIR-2021 
pip3 install --upgrade virtualenv==16.7.9 # this version virtualenv support the --no-site-packages option
virtualenv --no-site-packages env_mss # create new environment
source env_mss/bin/activate # activate environment
pip3 install -r requirements.txt # install requirements
```
You'd better have *wget* command installed so that to download pretrained models.

### Use pretrained model
You can run the following demo. If it's the first time you run this program, it will automatically download the pretrained models.

```shell
# <input-wav-file-path> is the .wav file to be separated
# <output-path-dir> is the folder to store the separation results 
# python3 main.py -i <input-wav-file-path> -o <output-path-dir>
python3 main.py -i example/test/zeno_sign_stereo.wav -o example/results
```

### Train new models from scratch
If you don't have 'musdb18q.zip' or 'musdb18hq' folder in the 'data' folder, we will automatically download the dataset for you.

```shell
# For track 'vocals'
source models/kqq_conv8_res/run.sh
# For track 'other'
source source models/no_v_kqq_multihead_v2_conv4/run.sh
```

## todo

- [ ] Open-source the ctraining pipline (before 2021-08-20)
- [ ] Write a report paper about my findings in this MSS Challenge (before 2021-08-31)

## Reference

If you find our code useful for your research, please consider citing:

>    @inproceedings{Liu2020,   
>      author={Haohe Liu and Lei Xie and Jian Wu and Geng Yang},   
>      title={{Channel-Wise Subband Input for Better Voice and Accompaniment Separation on High Resolution Music}},   
>      year=2020,   
>      booktitle={Proc. Interspeech 2020},   
>      pages={1241--1245},   
>      doi={10.21437/Interspeech.2020-2555},   
>      url={http://dx.doi.org/10.21437/Interspeech.2020-2555}   
>    }.





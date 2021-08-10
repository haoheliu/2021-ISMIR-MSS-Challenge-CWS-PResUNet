# Music Source Separation with Channel-wise Subband ResUnet (CWS-ResUNet)

You can use repo to separate bass, drums, vocals, and other track from a music mixture. This repo contains the pretrained Music Source Separation models I submitted to the [2021 ISMIR MSS Challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021).
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
First you need to clone this repo:
```shell
git clone https://github.com/haoheliu/CWS-ResUNet-MSS-Challenge-ISMIR-2021.git
```
Install the required packages
```shell
pip3 install -r requirement.txt
```
You'd better have *wget* command installed so that to download pretrained models.

Finally you can run the following demo. If it's the first time you run this program, it will automatically download the pretrained models.

```shell
# <input-wav-file-path> is the .wav file to be separated
# <output-path-dir> is the folder to store the separation results 
# python3 main.py -i <input-wav-file-path> -o <output-path-dir>
python3 main.py -i data/test/sign/mixture.wav -o data/results/sign
```

## todo

- [ ] Open-source the training pipline (before 2021-08-20)
- [ ] Write a report paper about my findings in this MSS Challenge (before 2021-08-31)







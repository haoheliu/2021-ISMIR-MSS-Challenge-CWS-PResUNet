python3 models/resunet_conv1_vocals/train.py --name 4_subband_resunet_vocals \
                                      --type vocals \
                                      --batchsize 8 \
                                      --gpuids 0 1 2 3 4 5 6 7

rm temp_path.json

python3 models/resunet_conv8_vocals/train.py --name 4_subband_resunet_vocals \
                                      --type vocals \
                                      --batchsize 16 \
                                      --gpuids 0 1

rm temp_path.json

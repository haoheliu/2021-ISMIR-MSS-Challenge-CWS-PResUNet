from predictor import SubbandResUNetPredictor
import time
import os
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-i", "--input_file", default="", help="The .wav file to be processed")
parser.add_argument("-o", "--output_path", default="", help="The output dirpath for the results")

args = parser.parse_args()

if __name__ == '__main__':
    scaledmixture_predictor = SubbandResUNetPredictor()

    submission = scaledmixture_predictor
    submission.prediction_setup()

    assert args.input_file[-3:] == "wav", "Error: invalid file "+ args.input_file+", we only accept .wav file."

    output_path = os.path.join(args.output_path,os.path.basename(args.input_file)[:-4])
    if(not os.path.exists(output_path)):
        os.makedirs(output_path, exist_ok=True)

    bass = os.path.join(output_path,"bass.wav")
    vocals = os.path.join(output_path,"vocals.wav")
    drums = os.path.join(output_path,"drums.wav")
    other = os.path.join(output_path,"other.wav")
    submission.prediction(
        mixture_file_path=args.input_file,
        vocals_file_path=vocals, bass_file_path=bass, drums_file_path=drums, other_file_path=other)


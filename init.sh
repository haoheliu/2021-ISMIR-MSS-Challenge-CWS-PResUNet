echo "Processing MUSDB18HQ dataset"

if [ ! -d "data/musdb18hq" ]; then
  if [ ! -f "data/musdb18hq.zip" ]; then
    echo "Dataset not found, downloading musdb18hq 44.1kHz dataset"
    wget https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1 -O data/musdb18hq.zip
  else
    unzip data/musdb18hq.zip -d data
  fi
fi

python3 utils/parser.py
echo "Data processing finished"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd ../Turtle
python setup.py develop --no_cuda_ext
cd ../scripts

# Download turtle pretrained models
python -m pip install gdown
mkdir -p ../Turtle/trained_models
gdown --folder "https://drive.google.com/drive/folders/1Mur4IboaNgEW5qyynTIHq8CSAGtyykrA" \
  --output ../Turtle/trained_models \
  --remaining-ok

# Download datasets
mkdir -p ../datasets/raw
gdown --folder "https://drive.google.com/drive/folders/1h21xh9JVhb_gmet8Vj8gxZy1_PpKJr8L" \
  --output ../datasets/raw \
  --remaining-ok

# Preprocess datasets
cd ../datasets/raw
unzip



sudo apt-get update
sudo apt-get install -y colmap imagemagick ffmpeg

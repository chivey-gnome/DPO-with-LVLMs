#!/bin/bash

#conda create -n new-hadpo python=3.10 -y
#conda activate new-hadpo
#pip install -r requirements.txt

#AMBER
python -m nltk.downloader all
python -m spacy download en_core_web_lg
mkdir data
mkdir data/AMBER
gdown https://drive.google.com/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl -O data/AMBER/images.zip
cd data/AMBER
curl -L -O https://github.com/junyangwang0410/AMBER/archive/refs/heads/master.zip
unzip images.zip
unzip master.zip
mv AMBER-master/data/* ./
mv AMBER-master/LICENSE ./
rm -f images.zip
rm -f master.zip
rm -rf AMBER-master
cd ../..
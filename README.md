# [Utagger]
This repository is a submission for IEEE Access 2020. The repository will be updated when getting new results which are not handled in the paper. 
We plan to report performances on all the languages in the CoNLL 2018 shared task.


# [Environment settings]

## 1. Install Anaconda
https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh

sh Anaconda3-5.2.0-Linux-x86_64.sh

conda search python

conda create -n Utagger python=3.6 anaconda

conda activate Utagger

## 2. Pytorch install
### Cuda version check!! :
nvcc --version

### An example of installing pytorch with CUDA 9.1
conda install pytorch=0.4.0 cuda91 -c soumith

### Pytorch without CUDA
conda install pytorch=0.4.0 -c soumith

## 3. Install python packages
conda install ftfy

### install ELMO
pip install allennlp


### install ELMO
pip install transformers

## 4. Clone the source
git clone https://github.com/jujbob/Utagger.git


## 5. Download corpora and external resources.
cd tagger

### Download training and testing corpora
cd corpus

wget --tries=150 https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2885/conll2018-test-runs.tgz

tar xvf conll2018-test-runs.tgz

wget --tries=150 https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz

tar xvf ud-treebanks-v2.2.tgz

### Download pre-trained word embeddings
cd ..

cd embeddings

wget --tries=150 https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1989/word-embeddings-conll17.tar

tar xvf word-embeddings-conll17.tar

mv ChineseT Chinese

xz --decompress ./\*/\*.xz

Note that we only use those embeddings for Japanese and Chinese, for other languages we use word embeddings trained by Facebook which is also permitted by the CoNLL 2018 shared task. Download here: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

e.g.) cd Finnish

wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fi.vec

mv fi.vectors fi.vectors_100

mv wiki.fi.vec fi.vectors

### Download ELMO
cd ..

wget -O ELMO.zip https://mycore.core-cloud.net/index.php/s/OKCV5HDllwdAAi6/download

unzip ELMO.zip

cd ELMO

unzip \\*.zip


### Make output forders (Models will be stored here)
cd ..

cd result

mkdir UD_Chinese-GSD
mkdir UD_Japanese-GSD
mkdir UD_Japanese-Modern
mkdir UD_English-EWT
mkdir UD_English-PUD
mkdir UD_French-GSD
mkdir UD_Korean-GSD
mkdir UD_Swedish-Talbanken
mkdir UD_Swedish-PUD
mkdir UD_Finnish-PUD
mkdir UD_Finnish-TDT
mkdir UD_Czech-PDT
mkdir UD_Czech-PUD


# [How to run?]

## Train

### 1. The Roberta model
python TTtagger_Roberta.py
* Note that the version of pytorch should higher than 1.4.0

### 2. The Joint model

In order to train a model, you need to select languages in "train_list.csv"

vi train_list.csv

Edit collum number 7 as "yes",  which means you will train that language. 
You can check several languages; it will be trained sequentially. For example:


language_ud_full,language_ud,language,language_code,train_size,dev_size,train,max_epoch,train_files,dev_file,train_language_codes,char,pos,train_type

Before: UD_Chinese-GSD,UD_Chinese,Chinese,zh_gsd,3997,0.7,no,280,zh_gsd-ud-train.conllu,zh_gsd-ud-dev.conllu,zh_gsd,yes,yes,

After : UD_Chinese-GSD,UD_Chinese,Chinese,zh_gsd,3997,0.7,yes,280,zh_gsd-ud-train.conllu,zh_gsd-ud-dev.conllu,zh_gsd,yes,yes,

CUDA_VISIBLE_DEVICES=0 python tagger.py train --cuda_device=0 --home_dir=$HOME_DIR/tagger/ --elmo_active=True --batch_size=32

cd $HOME_DIR/result/UD_Chinese-GSD/

Note that the output forder should be located in $HOME_DIR/result/UD_XXX-XXX.

### Run the tagger without GPU

python tagger.py train --home_dir=$HOME_DIR/tagger/ --batch_size=32

cd $HOME_DIR/result/UD_Chinese-GSD/




## 1. Prediction


### Run the tagger without GPU
python tagger.py predict --test_lang_code=ja_gsd --model $HOME_DIR/models/UD_Japanese-Modern-ELMO/Japanese139_98.43 --test_file=$HOME_DIR/corpus/official-submissions/Uppsala-18/ja_modern.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/ja_modern.conllu --elmo_weight=$HOME_DIR/ELMO/Japanese/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/Japanese/options.json --ud_name=ja_modern

### Run the tagger with GPU
cd $HOME_DIR/tagger

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=ja_gsd --model $HOME_DIR/models/UD_Japanese-Modern-ELMO/Japanese139_98.43 --test_file=$HOME_DIR/corpus/official-submissions/Uppsala-18/ja_modern.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/ja_modern.conllu --elmo_weight=$HOME_DIR/ELMO/Japanese/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/Japanese/options.json 

cat ja_modern.eval2

cat ja_modern.txt

[ja_modern]


[ja_gsd]
CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=ja_gsd --model $HOME_DIR/models/UD_Japanese-GSD-ELMO/Japanese139_98.43 --test_file=$HOME_DIR/corpus/official-submissions/HIT-SCIR-18/ja_gsd.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/ja_gsd.conllu --elmo_weight=$HOME_DIR/ELMO/Japanese/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/Japanese/options.json --ud_name=ja_gsd

[zh_gsd]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=zh_gsd --model $HOME_DIR/models/UD_Chinese-GSD-ELMO/Chinese151_95.81 --test_file=$HOME_DIR/corpus/official-submissions/HIT-SCIR-18/zh_gsd.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/zh_gsd.conllu --elmo_weight=$HOME_DIR/ELMO/Chinese/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/Chinese/options.json --ud_name=zh_gsd

[ko_gsd]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=ko_gsd --model $HOME_DIR/models/UD_Korean-GSD-ELMO/Korean92_96.63 --test_file=$HOME_DIR/corpus/official-submissions/HIT-SCIR-18/ko_gsd.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/ko_gsd.conllu --elmo_weight=$HOME_DIR/ELMO/Korean/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/Korean/options.json --ud_name=ko_gsd

[fr_gsd]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=fr_gsd --model $HOME_DIR/models/UD_French-GSD-ELMO/French100_97.97 --test_file=$HOME_DIR/corpus/official-submissions/Stanford-18/fr_gsd.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/fr_gsd.conllu --elmo_weight=$HOME_DIR/ELMO/French/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/French/options.json --ud_name=fr_gsd

[en_ewt]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=en_ewt --model $HOME_DIR/models/UD_English-EWT-ELMO/English --test_file=$HOME_DIR/corpus/official-submissions/LATTICE-18/en_ewt.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/en_ewt.conllu --elmo_weight=$HOME_DIR/ELMO/English/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/English/options.json --ud_name=en_ewt

[en_pud]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=en_ewt --model $HOME_DIR/models/UD_English-PUD-ELMO/English --test_file=$HOME_DIR/corpus/official-submissions/LATTICE-18/en_pud.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/en_pud.conllu --elmo_weight=$HOME_DIR/ELMO/English/weights.hdf5 --elmo_option=$HOME_DIR/ELMO/English/options.json --ud_name=en_pud

[fi_pud]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=fi_pud --model $HOME_DIR/models/UD_Finnish-PUD/Finnish200_97.45 --test_file=$HOME_DIR/corpus/official-submissions/LATTICE-18/fi_pud.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/fi_pud.conllu --ud_name=fi_pud

[cs_pud]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=cs_pud --model $HOME_DIR/models/UD_Swedish-PUD/Swedish75_97.62 --test_file=$HOME_DIR/corpus/official-submissions/LATTICE-18/sv_pud.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/sv_pud.conllu --ud_name=sv_pud

[sv_pud]

CUDA_VISIBLE_DEVICES=0 python tagger.py predict --cuda_device=0 --test_lang_code=sv_talbanken --model $HOME_DIR/models/UD_Swedish-PUD/Swedish75_97.62 --test_file=$HOME_DIR/corpus/official-submissions/LATTICE-18/sv_pud.conllu --gold_file=$HOME_DIR/corpus/official-submissions/00-gold-standard/sv_pud.conllu --ud_name=sv_pud




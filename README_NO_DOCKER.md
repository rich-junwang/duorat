# Instruction for running `DuoRAT` locally without using Docker

# Installation

Assume that you are now at `<your_duorat_folder>`.  

First, install required packages via `pipenv`. 
Note that you can have run it under a running virtual environment (w/ required `Python3.7`). 
```
pip install pipenv==2020.8.13
pipenv install --dev --system
sudo apt install libpython3.7-dev  # For jsonnet
```



Download `NLTK` data:
```
python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords
```

Download Stanford CoreNLP library:
```
mkdir ./third_party/corenlp 
cd ./third_party/corenlp
wget -nv http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
rm stanford-corenlp-full-2018-10-05.zip
cd ../..
```

Set up global environment variables:

```
./setup_env.sh printenv
```


Alternatively, 
```
export CACHE_DIR=./logdir
export TRANSFORMERS_CACHE=./logdir
export TOKENIZERS_PARALLELISM=true
export CORENLP_HOME=./third_party/corenlp/stanford-corenlp-full-2018-10-05
export CORENLP_SERVER_PORT=9000
```

Finally, install `DuoRAT` package:
```
pip install -e .
```

```
# Start virtual env
cd duorat
pipenv shell
```

# Usage

Download `Spider` dataset:
```
bash ./scripts/download_and_preprocess_spider.sh
```

Clone ElementAI's Spider repo (forked from original [Spider repo](https://github.com/taoyds/spider)):
```
git clone https://github.com/ElementAI/spider.git ./third_party/spider
```

## Train

```
# *** config w/ duorat-finetune-bert-large
# OOM: reduce batch_size to 4 and increse n_grad_accumulation_steps to 7
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ./logdir/duorat-finetune-bert-large &> logdir/train-duorat-finetune-bert-large.log &
```

```
# *** config w/ duorat-finetune-bert-large
# OOM: reduce batch_size to 4 and increse n_grad_accumulation_steps to 7
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content &> logdir/train-duorat-new-db-content.log &
```

## Inference & Evaluation

```
python scripts/infer.py --logdir ./logdir/duorat-new-db-content --section val --output ./logdir/duorat-new-db-content/val-duorat-new-db-content.output
```

```
python scripts/eval.py --config configs/duorat/duorat-new-db-content.jsonnet --section val --inferred ./logdir/duorat-new-db-content/val-duorat-new-db-content.output --output ./logdir/duorat-new-db-content/val-duorat-new-db-content.eval
```

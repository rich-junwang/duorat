# Instruction for running `DuoRAT` locally without using Docker

# Installation

Assume that you are now at `<your_duorat_folder>`.  

First, install required packages via `pipenv`. 
Note that you can have run it under a running virtual environment (w/ required `Python3.7`). 
```
pip install pipenv==2020.8.13
pipenv install --dev --system
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

## Interactive

```
python scripts/interactive.py --logdir ./logdir/duorat-new-db-content --config configs/duorat/duorat-new-db-content.jsonnet --db-path <your_path>/cinema.sqlite
```

### Results

Results on Spider dataset:

| Model   | Dev Accuracy  | 
|---|---|
| w/ GLOVE (100K steps)   | 62.6 (easy_exact = 0.8064516129032258, medium_exact = 0.6300448430493274, hard_exact = 0.5459770114942529, extra_exact = 0.42771084337349397)  | 
| w/ finetuned BERT (base, uncased) (150K steps)  | 64.3 (easy_exact = 0.8064516129032258, medium_exact = 0.6569506726457399, hard_exact = 0.5862068965517241, extra_exact = 0.42168674698795183)  | 
| w/ finetuned BERT (large, uncased) (150K steps) | 70.7 (easy_exact = 0.8911290322580645, medium_exact = 0.742152466367713, hard_exact = 0.5804597701149425, extra_exact = 0.46987951807228917)  |
| w/ finetuned BERT (large, uncased) + w/ database content (150K steps) | ~72 (easy_exact = 0.9153225806451613, medium_exact = 0.7533632286995515, hard_exact = 0.632183908045977, extra_exact = 0.42771084337349397) | 


# Contact
Vu Hoang (vu.hoang@oracle.com, duyvuleo@gmail.com)
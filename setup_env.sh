#!/bin/bash

export CACHE_DIR=./logdir
export TRANSFORMERS_CACHE=./logdir
export TOKENIZERS_PARALLELISM=true
export CORENLP_HOME=./third_party/corenlp/stanford-corenlp-full-2018-10-05
export CORENLP_SERVER_PORT=9000

exec "$@"

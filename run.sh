
set -eo pipefail
#mkdir -p ~/duorat/logdir
#export PYTHONPATH=~/duorat:$PYTHONPATH
#python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=1234 \
#scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir logdir/duorat-bert


SRC_DIR=$HOME/duorat
export PYTHONPATH=${SRC_DIR}:$PYTHONPATH
export CACHE_DIR=./logdir
export TRANSFORMERS_CACHE=./logdir
export TOKENIZERS_PARALLELISM=true
export CORENLP_SERVER_PORT=9000
export SM_NUM_GPUS=${SM_NUM_GPUS:-1} # default to 1 gpu, set to 8 if has 8 gpus
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CORENLP_HOME=$HOME/corenlp/stanford-corenlp-full-2018-10-05

OUTPUT_DIR=logdir/duorat-bert
for ((i = 1 ; i < $SM_NUM_GPUS; i++)); do
    STD_FILE=${OUTPUT_DIR}/"Job_"$i
#    python3 scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ${OUTPUT_DIR} --local_rank $i 1>$STD_FILE".stdout"  2>$STD_FILE".stderr" &
    python3 scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ${OUTPUT_DIR} --local_rank $i  &
    echo "Job_$i has been launched."
done

echo "Job_0 has been launched."
log_file=${OUTPUT_DIR}/"Job_0"
#python3 scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ${OUTPUT_DIR} --local_rank 0  2>&1 | tee -a ${log_file}
python3 scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ${OUTPUT_DIR} --local_rank 0

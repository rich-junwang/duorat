
#mkdir -p ~/duorat/logdir
export PYTHONPATH=~/duorat:$PYTHONPATH
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=1234 \
scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir logdir/duorat-bert

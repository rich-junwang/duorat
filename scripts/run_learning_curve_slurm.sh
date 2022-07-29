# coding=utf-8
#~/usr/bin/env bash

wdir=/mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat
logs_folder=${wdir}/logdir
running_script=${wdir}/scripts/train.py
running_config=${wdir}/configs/duorat/duorat-new-db-content.jsonnet
partition=batch
batch_job_folder=${logs_folder}/batch_jobs

g_constraint="shape=BM.GPU3.8|shape=VM.GPU3.4|shape=VM.GPU3.4|shape=VM.GPU3.2|shape=VM.GPU3.1,ad=3"
#c_constraint="shape=VM.Standard2.24|shape=VM.Standard.E3.8|shape=VM.Standard.E3.16|shape=VM.Standard.E3.32"

job_config="#SBATCH --partition=${partition} \n#SBATCH --mem=16GB \n#SBATCH --gres=gpu:1 \n#SBATCH -c 2 \n#SBATCH --constraint=${g_constraint}"

submit_job () {
    job_content="#!/bin/bash"
    job_content="${job_content}\n${job_config}"
    job_content="${job_content}\n#SBATCH --job-name=${job_name}"
    job_content="${job_content}\n#SBATCH --output=${job_log}_%A.out"
    job_content="${job_content}\n#SBATCH --error=${job_log}_%A.err"
    job_content="${job_content}\n\nexport CACHE_DIR=${wdir}/logdir\nexport TRANSFORMERS_CACHE=${wdir}/logdir\nexport CORENLP_HOME=${wdir}/third_party/corenlp/stanford-corenlp-full-2018-10-05\nexport CORENLP_SERVER_PORT=9002\n\nsource /mnt/shared/vchoang/tools/pyvenv37-oda-text2sql-duorat/bin/activate"
    job_content="${job_content}\n\ncd ${wdir}"
    job_content="${job_content}\n\n${job_command}"

    job_seed=$RANDOM
    job_file=${batch_job_folder}/job_${job_seed}.sh
    echo -e $job_content > $job_file
    run_command="sbatch ${job_file}"
    echo $run_command
    eval $run_command
    sleep 15
}


dataset=$1
ratio=$2

if [[ "${dataset}" = "sparc" ]] || [[ "${dataset}" = "cosql" ]]; then
    running_config=${wdir}/configs/duorat/duorat-${dataset}-new-db-content.jsonnet
fi

mkdir ${logs_folder}
mkdir ${batch_job_folder}

g_start=1
g_end=$((100/${ratio}))
for ((i=${g_start};i<${g_end};i++));
do
    percentage=$((${i}*${ratio}))
    echo "Start training job with ${percentage}% of training data..."
    job_name=duorat_new_db_content_${dataset}_${percentage}p
    job_log=${logs_folder}/train_duorat_${dataset}_${percentage}p
    job_command="python3 ${running_script} --config ${running_config} --logdir ${logs_folder}/duorat-${dataset}-new-db-content-${percentage}p --train-sample-ratio ${percentage} --force-preprocess --force-train"
    submit_job
done
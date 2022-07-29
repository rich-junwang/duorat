# *** Spider

# duorat-dev (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-dev.jsonnet --logdir ./logdir/duorat-dev &> logdir/train-duorat-dev.log &

# duorat-bert (base, freezed) --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-bert.jsonnet --logdir ./logdir/duorat-freeze-bert-base &> logdir/train-duorat-freeze-bert-base.log &

# duorat-finetune-bert-base --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-finetune-bert-base.jsonnet --logdir ./logdir/duorat-finetune-bert-base &> logdir/train-duorat-finetune-bert-base.log &

# duorat-finetune-bert-large --> ok
# OOM: reduce batch_size to 4
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ./logdir/duorat-finetune-bert-large &> logdir/train-duorat-finetune-bert-large.log &
python scripts/train.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --logdir ./logdir/duorat-finetune-bert-large --preprocess-only
python scripts/infer.py --logdir ./logdir/duorat-finetune-bert-large --section val --output ./logdir/duorat-finetune-bert-large/val-duorat-finetune-bert-large.output  --force --nproc 5 --beam-size 1
python scripts/eval.py --config configs/duorat/duorat-finetune-bert-large.jsonnet --section val --do-execute --inferred ./logdir/duorat-finetune-bert-large/val-duorat-finetune-bert-large.output --output ./logdir/duorat-finetune-bert-large/val-duorat-finetune-bert-large.eval
python scripts/interactive.py --logdir ./logdir/duorat-finetune-bert-large --config configs/duorat/duorat-finetune-bert-large.jsonnet --db-path ./tests/data/test_new_db.sqlite

# duorat-finetune-bert-large-attention-maps --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-finetune-bert-large-attention-maps.jsonnet --logdir ./logdir/duorat-finetune-bert-large-attention-maps &> logdir/train-duorat-finetune-bert-large-attention-maps.log &

# duorat-new-db-content --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content &> logdir/train-duorat-new-db-content.log &
python scripts/infer.py --logdir ./logdir/duorat-new-db-content --section val --output ./logdir/duorat-new-db-content/val-duorat-new-db-content.output  --force --nproc 5 --beam-size 1
python scripts/eval.py --config configs/duorat/duorat-new-db-content.jsonnet --section val --do-execute --inferred ./logdir/duorat-new-db-content/val-duorat-new-db-content.output --output ./logdir/duorat-new-db-content/val-duorat-new-db-content.eval
python scripts/interactive.py --logdir ./logdir/duorat-new-db-content --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./tests/data/test_new_db.sqlite

# w/ better model duorat-new-db-content-bs4-ac7
python scripts/infer.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --section val --output ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.output --force --nproc 5 --beam-size 1
python scripts/eval.py --config configs/duorat/duorat-new-db-content.jsonnet --section val --do-execute --inferred ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.output --output ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval
python scripts/interactive.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./tests/data/test_new_db.sqlite

# on train
python scripts/infer.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --section train --output ./logdir/duorat-new-db-content-bs4-ac7/train-duorat-new-db-content-bs4-ac7.output --force --nproc 5 --beam-size 1
python scripts/eval.py --config configs/duorat/duorat-new-db-content.jsonnet --section train --do-execute --inferred ./logdir/duorat-new-db-content-bs4-ac7/train-duorat-new-db-content-bs4-ac7.output --output ./logdir/duorat-new-db-content-bs4-ac7/train-duorat-new-db-content-bs4-ac7.eval

# w/ custom seed
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-seed1 --seed 1 &> logdir/train-duorat-new-db-content-seed1.log &

# serve
TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=1 python scripts/serve.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./data/database --server-port 8000 --do-logging --log-append --do-sql-post-processing --db-passwords-file ./data/db_passwords.sec &>./logdir/duorat-new-db-content-bs4-ac7/server_conn.log &

# for debugging only
TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=3 python scripts/serve.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./data/database --server-port 8900 --do-logging --log-append --do-sql-post-processing --log-file-name serve_security.log --db-passwords-file ./data/db_passwords.sec &>./logdir/duorat-new-db-content-bs4-ac7/server_conn_security.log &

# *** duorat-new-db-content-no-whole --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-no-whole.jsonnet --logdir ./logdir/duorat-new-db-content-no-whole &> logdir/train-duorat-new-db-content-no-whole.log &

# *** duorat-good-no-bert --> ok
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-good-no-bert.jsonnet --logdir ./logdir/duorat-good-no-bert &> logdir/train-duorat-good-no-bert.log &

# *** duorat-good-no-bert-no-from --> failed
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-good-no-bert-no-from.jsonnet --logdir ./logdir/duorat-good-no-bert-no-from &> logdir/train-duorat-good-no-bert-no-from.log &

# *** DATA AUGMENTATION

# Paraphrases by Back Translation
# EN-DE-EN (Google Translate)
python3 scripts/split_spider_by_db.py --aug-data train_spider_bt_aug_paraphrases.json,train_others_bt_aug_paraphrases.json --aug-suffix bt_para_aug

# run1
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug &> logdir/train-duorat-new-db-content-bt-para-aug.log &
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug-150k-steps.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug &> logdir/train-duorat-new-db-content-bt-para-aug-150k-steps.log &

# run2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug-run2 &> logdir/train-duorat-new-db-content-bt-para-aug-run2.log &
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug-150k-steps.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug-run2 &> logdir/train-duorat-new-db-content-bt-para-aug-run2-150k-steps.log &

# run3
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug-run3 &> logdir/train-duorat-new-db-content-bt-para-aug-run3.log &
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-bt-para-aug-150k-steps.jsonnet --logdir ./logdir/duorat-new-db-content-bt-para-aug-run3 &> logdir/train-duorat-new-db-content-bt-para-aug-run3-150k-steps.log &

python scripts/collect_training_data_from_oda_para_dm.py /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider/paraphrasing_results.csv /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider/train_oda_dm_para.json

# Paraphrases by Manual Paraphrase Collection
python3 scripts/split_spider_by_db.py --aug-data train_oda_dm_para.json --aug-suffix oda_dm_para_aug

# *** Evaluate on other semantic parsing datasets

# Geo
cd ./data
bash scripts/download_michigan_no_docker.sh geo
python scripts/infer_questions.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --data-config data/michigan.libsonnet --questions data/database/geo_test/examples.json --output-google ./logdir/duorat-new-db-content-bs4-ac7/inferred_geo.json
python scripts/evaluation_google.py --predictions_filepath ./logdir/duorat-new-db-content-bs4-ac7/inferred_geo.json --output_filepath ./logdir/duorat-new-db-content-bs4-ac7/output_geo.json --cache_filepath data/database/geo_test/geo_cache.json  --timeout 180
[NOT_EXIST] python scripts/filter_results.py ./logdir/duorat-new-db-content-bs4-ac7/output_geo.json

# Atis (failed)
bash scripts/download_michigan_no_docker.sh atis

# Academic (failed)
bash scripts/download_michigan_no_docker.sh academic

# Restaurants
bash scripts/download_michigan_no_docker.sh restaurants
python scripts/infer_questions.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --data-config data/michigan_restaurants.libsonnet --questions data/database/restaurants_test/examples.json --output-google ./logdir/duorat-new-db-content-bs4-ac7/inferred_restaurants.json

# Yelp
bash scripts/download_michigan_no_docker.sh yelp
python scripts/infer_questions.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --data-config data/michigan_yelp.libsonnet --questions data/database/yelp_test/examples.json --output-google ./logdir/duorat-new-db-content-bs4-ac7/inferred_yelp.json

# IMDB
bash scripts/download_michigan_no_docker.sh imdb
python scripts/infer_questions.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --data-config data/michigan_imdb.libsonnet --questions data/database/imdb_test/examples.json --output-google ./logdir/duorat-new-db-content-bs4-ac7/inferred_imdb.json

# Scholar
bash scripts/download_michigan_no_docker.sh scholar

# Advising
bash scripts/download_michigan_no_docker.sh advising

# *** Sparc

# duorat-sparc-dev
python scripts/train.py --config configs/duorat/duorat-sparc-dev.jsonnet --logdir ./logdir/duorat-sparc-dev --force-preprocess --force-train

# duorat-sparc-new-db-content (baseline)
# no interaction history in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-baseline.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-baseline --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-baseline.log &

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-baseline.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-baseline &> logdir/train-duorat-sparc-new-db-content-baseline.log1 &

# duorat-sparc-new-db-content

# train
# interaction history (source, 1) in the inputs
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --logdir ./logdir/duorat-sparc-new-db-content --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content.log &
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-150k-steps.jsonnet --logdir ./logdir/duorat-sparc-new-db-content &> logdir/train-duorat-sparc-new-db-content.log1 &

CUDA_VISIBLE_DEVICES=3 python scripts/infer.py --logdir ./logdir/duorat-sparc-new-db-content --section val --output ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content.output  --force

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content-eval-testsuite.output

[FAILED] CUDA_VISIBLE_DEVICES=3 python scripts/eval.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --section val --do-execute --inferred ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content.output --output ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content.eval

# serve
# sparc
CUDA_VISIBLE_DEVICES=2 python scripts/serve.py --logdir ./logdir/duorat-sparc-new-db-content --config configs/duorat/duorat-sparc-new-db-content.jsonnet --db-path ./data/sparc/database --server-port 8200 --do-logging --log-append --do-sql-post-processing --db-passwords-file ./data/db_passwords.sec --log-file-name serve_followup.log &>./logdir/duorat-sparc-new-db-content/server_followup_conn.log &
cd text2sql-poc-ui-demo-jet9
node ./node_modules/@oracle/ojet-cli/ojet.js serve --server-port=8300 --livereload-port 36729

# testsuite eval
# quick test
python3 evaluation.py --gold ./evaluation_examples/gold.txt --pred ./evaluation_examples/predict.txt --db ./database  --etype all  --progress_bar_for_each_datapoint

# run1
# testsuite execution accuracy without values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value  --progress_bar_for_each_datapoint

# testsuite execution accuracy with values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content-eval-testsuite.output --db ./database --etype exec --progress_bar_for_each_datapoint

# interaction history (source, 1) in the inputs (run2)
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-run2 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-run2.log &

# interaction history (source, 1) in the inputs (run3)
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-run3 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-run3.log &

# interaction history (source, 2) in the inputs
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-int2.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-int2 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-int2.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-int2.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-int2 &> logdir/train-duorat-sparc-new-db-content-int2.log1 &

# interaction history (target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-dev-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-dev-target-interaction --force-preprocess --force-train

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-target-interaction --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-target-interaction.log &

CUDA_VISIBLE_DEVICES=3 python scripts/infer.py --logdir ./logdir/duorat-sparc-new-db-content-target-interaction --section val --output ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth.output  --force

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-eval-testsuite.output --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-eval-testsuite.output

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value --progress_bar_for_each_datapoint

# run2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-target-interaction-run2 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-target-interaction-run2.log &

# run3
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-target-interaction-run3 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-target-interaction-run3.log &

# interaction history (source&target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-source-target-interaction --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-source-target-interaction.log &

# run2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-source-target-interaction-run2 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-source-target-interaction-run2.log &

# run3
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-source-target-interaction-run3 --force-preprocess --force-train &> logdir/train-duorat-sparc-new-db-content-source-target-interaction-run3.log &

# *** CoSQL

# interaction history (target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-target-interaction --force-preprocess --force-train &> logdir/train-duorat-cosql-new-db-content-target-interaction.log &

CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --logdir ./logdir/duorat-cosql-new-db-content-target-interaction --section val --output ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth.output  --force

python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth-eval-testsuite.output "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

# interaction history (source&target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-source-target-interaction --force-preprocess --force-train &> logdir/train-duorat-cosql-new-db-content-source-target-interaction.log &

# duorat-cosql-dev
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-cosql-dev.jsonnet --logdir ./logdir/duorat-cosql-dev --force-preprocess --force-train &> logdir/train-duorat-cosql-dev.log &

# duorat-cosql-new-db-content
# interaction history (1) in the inputs

# run1
# train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-cosql-new-db-content --force-preprocess --force-train &> logdir/train-cosql-new-db-content.log &

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content-150k-steps.jsonnet --logdir ./logdir/duorat-cosql-new-db-content &> logdir/train-cosql-new-db-content.log1 &

# infer
CUDA_VISIBLE_DEVICES=3 python scripts/infer.py --logdir ./logdir/duorat-cosql-new-db-content --section val --output ./logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content.output  --force

# eval
python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content-eval-testsuite.output "I have left the chat"

# testsuite execution accuracy without values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value  --progress_bar_for_each_datapoint

# testsuite execution accuracy with values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content-eval-testsuite.output --db ./database --etype exec --progress_bar_for_each_datapoint

# run2
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-run2 --force-preprocess --force-train &> logdir/train-cosql-new-db-content-run2.log &

CUDA_VISIBLE_DEVICES=2 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content-150k-steps.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-run2 &> logdir/train-cosql-new-db-content-run2.log1 &

# infer
CUDA_VISIBLE_DEVICES=3 python scripts/infer.py --logdir ./logdir/duorat-cosql-new-db-content-run2 --section val --output ./logdir/duorat-cosql-new-db-content-run2/val-duorat-cosql-new-db-content.output  --force

# eval
python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-run2/val-duorat-cosql-new-db-content.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-run2/val-duorat-cosql-new-db-content-eval-testsuite.output "I have left the chat"

# testsuite execution accuracy without values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-run2/val-duorat-cosql-new-db-content-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value  --progress_bar_for_each_datapoint

# testsuite execution accuracy with values
python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-run2/val-duorat-cosql-new-db-content-eval-testsuite.output --db ./database --etype exec --progress_bar_for_each_datapoint

# run3
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-run3 --force-preprocess --force-train &> logdir/train-cosql-new-db-content-run3.log &

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content-150k-steps.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-run3 &> logdir/train-cosql-new-db-content-run3.log1 &

# *** Learning curve for Spider/Sparc/CoSQL

# Spider
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-5p --train-sample-ratio 5
sh run_learning_curve_slurm.sh spider 5

# Sparc
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-5p --train-sample-ratio 5
sh run_learning_curve_slurm.sh sparc 5

# CoSQL
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-5p --train-sample-ratio 5
sh run_learning_curve_slurm.sh cosql 5

# *** Joint Spider+Sparc+CoSQL
# dev (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-sparc-cosql-dev.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-dev --force-preprocess --force-train &> logdir/train-duorat-spider-sparc-cosql-dev.log &

# train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content --force-preprocess --force-train &> logdir/train-duorat-spider-sparc-cosql-new-db-content.log &

# infer
CUDA_VISIBLE_DEVICES=2 python scripts/infer.py --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content --output ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content.output  --force
# outputs: val-duorat-spider-sparc-cosql-new-db-content.output{.Spider,.Sparc,.CoSQL}

# run4
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-run4 --output ./logdir/duorat-spider-sparc-cosql-new-db-content-run4/val-duorat-spider-sparc-cosql-new-db-content.output  --force
# outputs: val-duorat-spider-sparc-cosql-new-db-content.output{.Spider,.Sparc,.CoSQL}

# interaction history (target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction --force-preprocess --force-train &> logdir/train-duorat-spider-sparc-cosql-new-db-content-target-interaction.log &

# interaction history (source&target, 1) in the inputs
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction --force-preprocess --force-train &> logdir/train-duorat-spider-sparc-cosql-new-db-content-source-target-interaction.log &

CUDA_VISIBLE_DEVICES=2 python scripts/serve.py --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-source-target-interaction.jsonnet --db-path ./data/database --server-port 8200 --do-logging --log-append --do-sql-post-processing --log-file-name serve_followup.log  --db-passwords-file ./data/db_passwords.sec &>./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/server_followup_conn.log &

# eval w/ TestSuite
# Spider
python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content.output.Spider ./data/spider/dev.json ./data/spider/dev_gold.sql /tmp/dump_file.txt ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.Spider

# Sparc

python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content.output.Sparc ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.Sparc

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.Sparc --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value  --progress_bar_for_each_datapoint

# run4
python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content-run4/val-duorat-spider-sparc-cosql-new-db-content.output.Sparc ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-spider-sparc-cosql-new-db-content-run4/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.Sparc

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-cosql-new-db-content-run4/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.Sparc --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value  --progress_bar_for_each_datapoint

# CoSQL
python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content.output.CoSQL ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.CoSQL "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-cosql-new-db-content/val-duorat-spider-sparc-cosql-new-db-content-eval-testsuite.output.CoSQL --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value  --progress_bar_for_each_datapoint

# 200K steps
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-200k-steps.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content &>logdir/train-duorat-spider-sparc-cosql-new-db-content-200k-steps.log &

# *** User intent prediction
python convert_to_fasttext_format.py cosql_train.json cosql_train_intent.fasttext
python convert_to_fasttext_format.py cosql_dev.json cosql_dev_intent.fasttext
python scripts/build_text_classifier.py /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/user_intent_prediction/cosql_train_intent.fasttext /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/user_intent_prediction/cosql_dev_intent.fasttext /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/user_intent_prediction/cosql_dev_intent.fasttext exp/models/cosql_intent_model.bin
# (1503, 0.8569527611443779, 0.8293625241468127)
# Accuracy on test split: 0.8380889183808892

# *** WikiSQL
# create dataset
bash scripts/create_wikisql_dataset.sh

# dev (for debugging)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-wikisql-dev.jsonnet --logdir ./logdir/duorat-wikisql-dev --force-preprocess --force-train &> logdir/train-duorat-wikisql-dev.log &

# train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-wikisql-new-db-content.jsonnet --logdir ./logdir/duorat-wikisql-new-db-content --force-preprocess --force-train &> logdir/train-duorat-wikisql-new-db-content.log &

# infer

# *** Extracting NL2SQL templates
# Spider

# w/o OP & SC denotations
CUDA_VISIBLE_DEVICES=0 python scripts/data_aug/extract_templates.py --sql-keyword-list-file ./scripts/data_aug/sql_keywords.txt --duorat-prediction-file ./logdir/duorat-new-db-content-bs4-ac7/train-duorat-new-db-content-bs4-ac7.eval --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --template-output-file ./logdir/duorat-new-db-content-bs4-ac7/train_spider.nl2sql_templates_no_sc_op_fixed --output-in-csv --with-stemming --top-k-t 100 --top-k-e 10

# w/ OP & SC denotations
CUDA_VISIBLE_DEVICES=0 python scripts/data_aug/extract_templates.py --sql-keyword-list-file ./scripts/data_aug/sql_keywords.txt --duorat-prediction-file ./logdir/duorat-new-db-content-bs4-ac7/train-duorat-new-db-content-bs4-ac7.eval --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --template-output-file ./logdir/duorat-new-db-content-bs4-ac7/train_spider.nl2sql_templates_sc_op_fixed --output-in-csv --with-stemming --with-op-denotation --with-sc-denotation --top-k-t 100 --top-k-e 10

# *** Spider w/ extra schema descriptions
python3 scripts/split_spider_by_db.py --tables-path tables_descriptions.json

# train & test w/ additional schema descriptions
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-extra-schema-descriptions.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-extra-schema-descriptions --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-extra-schema-descriptions.log &

# run2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-extra-schema-descriptions.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-extra-schema-descriptions-run2 --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-extra-schema-descriptions-run2.log &

# run3
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-extra-schema-descriptions.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-extra-schema-descriptions-run3 --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-extra-schema-descriptions-run3.log &

# *** Spider w/ inferred column types
python3 scripts/split_spider_by_db.py --tables-path tables_inferred_col_types.json

# train & test w/ inferred column types
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-inferred-col-types.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-inferred-col-types --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-inferred-col-types.log &

# run2
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-inferred-col-types.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-inferred-col-types-run2 --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-inferred-col-types-run2.log &

# run3
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-inferred-col-types.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-inferred-col-types-run3 --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-inferred-col-types-run3.log &

# test only w/ additional schema descriptions
CUDA_VISIBLE_DEVICES=3 python scripts/infer.py --config configs/duorat/duorat-spider-new-db-content-with-extra-schema-descriptions.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7-with-extra-schema-descriptions/ --section val --output ./logdir/duorat-new-db-content-bs4-ac7-with-extra-schema-descriptions/val-duorat-new-db-content-bs4-ac7-with-extra-schema-descriptions.output --force

# *** Inference latency

# Spider
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content.output

# Sparc
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-sparc-new-db-content.jsonnet --logdir ./logdir/duorat-sparc-new-db-content --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-sparc-new-db-content/val-duorat-sparc-new-db-content.output &> ./logdir/duorat-sparc-new-db-content/gpu_latency.log

# Sparc w/ target interaction (prediction)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-sparc-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction.output &> ./logdir/duorat-sparc-new-db-content-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-eval-testsuite.output

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-eval-testsuite.output --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# Sparc w/ target interaction (groundtruth)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-sparc-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-io.output --use-groundtruths

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-io.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-io-eval-testsuite.output

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-io-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-sparc-new-db-content-target-interaction/val-duorat-sparc-new-db-content-target-interaction-groundtruth-io-eval-testsuite.output --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# Sparc w/ source&target interaction (groundtruth)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-sparc-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-source-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-groundtruth.output --use-groundtruths

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-groundtruth.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-groundtruth-eval-testsuite.output

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-groundtruth-eval-testsuite.output --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# Sparc w/ source&target interaction (prediction)
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-sparc-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-sparc-new-db-content-source-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction.output

python scripts/get_testsuite_preds.py ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction.output ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-eval-testsuite.output

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-sparc-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-eval-testsuite.output --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# CoSQL
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-cosql-new-db-content.jsonnet --logdir ./logdir/duorat-cosql-new-db-content --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content.output --ignored-patterns "I have left the chat" &> ./logdir/duorat-cosql-new-db-content/gpu_latency.log

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-cosql-new-db-content/val-duorat-cosql-new-db-content-eval-testsuite.output --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# CoSQL w/ target interaction (groundtruth)
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth.output --ignored-patterns "I have left the chat" --use-groundtruths &> ./logdir/duorat-cosql-new-db-content-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth-eval-testsuite.output "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-groundtruth-eval-testsuite.output --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# CoSQL w/ target interaction (prediction)
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction.output --ignored-patterns "I have left the chat" &> ./logdir/duorat-cosql-new-db-content-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-eval-testsuite.output "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-cosql-new-db-content-target-interaction/val-duorat-cosql-new-db-content-target-interaction-eval-testsuite.output --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# CoSQL w/ source&target interaction (groundtruth)
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-source-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-groundtruth.output --ignored-patterns "I have left the chat" --use-groundtruths &> ./logdir/duorat-cosql-new-db-content-source-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-groundtruth.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-groundtruth-eval-testsuite.output "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-groundtruth-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-groundtruth-eval-testsuite.output --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# CoSQL w/ source&target interaction (prediction)
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-cosql-new-db-content-source-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction.output --ignored-patterns "I have left the chat" &> ./logdir/duorat-cosql-new-db-content-source-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction.output ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-eval-testsuite.output "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-eval-testsuite.output --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-cosql-new-db-content-source-target-interaction/val-duorat-cosql-new-db-content-source-target-interaction-eval-testsuite.output --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# Spider + Sparc + CoSQL w/ target interaction (prediction)

# Sparc
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction.output.sparc &> ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction.output.sparc ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.sparc

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.sparc --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.sparc --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# CoSQL
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction.output.cosql --ignored-patterns "I have left the chat" &> ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction.output.cosql ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.cosql "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.cosql --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-spider-sparc-cosql-new-db-content-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-target-interaction-eval-testsuite.output.cosql --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# Spider + Sparc + CoSQL w/ source&target interaction (prediction)

# Sparc
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction --data-type Sparc --db-folder-path ./data/sparc/database/ --eval-file ./data/sparc/dev.json --output-eval-file ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction.output.sparc &> ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction.output.sparc ./data/sparc/dev.json ./data/sparc/dev_gold.txt /tmp/dump_file.txt ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-eval-testsuite.output.sparc

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/dev_gold_fixed.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction-eval-testsuite.output.sparc --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/sparc/tables.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/sparc/dev_gold.txt --pred ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-sparc-new-db-content-source-target-interaction-eval-testsuite.output.sparc --etype match --db ./data/sparc/database/ --table ./data/sparc/tables.json

# CoSQL
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-spider-sparc-cosql-new-db-content-source-target-interaction.jsonnet --logdir ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction --data-type CoSQL --db-folder-path ./data/cosql/database/ --eval-file ./data/cosql/sql_state_tracking/cosql_dev.json --output-eval-file ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction.output.cosql --ignored-patterns "I have left the chat" &> ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/gpu_latency.log

python scripts/get_testsuite_preds.py ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction.output.cosql ./data/cosql/sql_state_tracking/cosql_dev_fixed.json ./data/cosql/sql_state_tracking/dev_gold_fixed.txt ./data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt  ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction-eval-testsuite.output.cosql "I have left the chat"

python3 evaluation.py --gold /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/sql_state_tracking/dev_gold_fixed_filtered.txt --pred /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/duorat-spider-sparc-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction-eval-testsuite.output.cosql --db ./database --etype all --table /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/cosql/tables_fixed.json --plug_value --progress_bar_for_each_datapoint

python2 ./third_party/sparc/evaluation.py --gold ./data/cosql/sql_state_tracking/dev_gold_fixed.txt --pred ./logdir/duorat-spider-sparc-cosql-new-db-content-source-target-interaction/val-duorat-spider-sparc-cosql-new-db-content-source-target-interaction-eval-testsuite.output.cosql --etype match --db ./data/cosql/database/ --table ./data/cosql/tables.json

# *** Experiments for focusing idea

# ** Use Spider synthetic data from tensor2struct
#python collect_spider_synthetic_data_tensor2struct.py --tensor2struct-synthetic-data-file
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data spider_synthetic_data_tensor2struct.json --aug-suffix spider_synthetic_data_tensor2struct

# original training data + synthetic data

# train w/ synthetic data --> finetune w/ original data

python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-synthetic-data-tensor2struct.jsonnet --logdir ./logdir/duorat-spider-new-db-content-synthetic-data-tensor2struct --force-preprocess --force-train

# ** Use synthetic data generated by template-based SCFG

# get synthetic data from @Philip Arthur
# v1
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type tsv --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v1/spider --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_100s.json --samples-by-db 100
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_100s.json --aug-suffix spider_synthetic_data_template_scfg_100s\

# v3-fixed
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed.json --samples-by-db -1
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed

# v1
# train w/ synthetic data --> finetune w/ original data
# train w/ synthetic data
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-synthetic-data-template-scfg-100s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-synthetic-data-template-scfg-100s --force-preprocess --force-train
# finetune w/ original data
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-initialized-from-synthetic-data-template-scfg-100s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-initialized-from-synthetic-data-template-scfg-100s --force-preprocess --force-train

# train w/ mix of original and synthetic data
# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing.log &

# w/o batch balancing
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data --force-preprocess --force-train &>././logdir/train-duorat-spider-new-db-content-with-original-plus-synthetic-data.log &

# v3_fixed
# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing-v3-fixed.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing-v3-fixed --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-with-original-plus-synthetic-data-batch-balancing-v3-fixed.log &

# *** v5 (w/ gold templates)

# 50 samples per db
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5 --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_50s.json --samples-by-db 50
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_50s.json' --default-example-file-name examples_with_synthetic_data_v5_by_gold_template_scfg_50s.json

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-50s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-50s --force-preprocess --force-train

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-50s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-50s-bb --force-preprocess --force-train

# * continuous training with synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-50s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-50s --force-preprocess --force-train

# 100 samples per db
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5 --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_100s.json --samples-by-db 100
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_100s.json' --default-example-file-name examples_with_synthetic_data_v5_by_gold_template_scfg_100s.json

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-100s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-100s --force-preprocess --force-train

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-100s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-100s-bb --force-preprocess --force-train

# * continuous training with synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-100s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-100s --force-preprocess --force-train

# 200 samples per db
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5 --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_200s.json --samples-by-db 200
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_by_gold_template_scfg_200s.json' --default-example-file-name examples_with_synthetic_data_v5_by_gold_template_scfg_200s.json

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-200s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-200s --force-preprocess --force-train

# * initialized from pretrained model then continuously training with original training data + synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-200s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-by-gold-template-scfg-200s-bb --force-preprocess --force-train

# * continuous training with synthetic data by gold template-based SCFG for validation dbs (50 examples per db)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-200s.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-synthetic-data-by-gold-template-scfg-200s --force-preprocess --force-train

# ** all synthetic data (v5)
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5 --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_by_gold_template_scfg.json --samples-by-db -1
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_by_gold_template_scfg.json' --default-example-file-name examples_with_synthetic_data_v5_by_gold_template_scfg.json

# ** v5_mono (NL parts are generated by T5 translator)

# 50 samples
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5_mono/spider/database --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_50s.json --samples-by-db 50
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_50s.json' --default-example-file-name examples_with_synthetic_data_v5_mono_nl_by_t5_gen_50s.json

# w/ batch balancing
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-50s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-50s-bb --force-preprocess --force-train

# 100 samples
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5_mono/spider/database --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_100s.json --samples-by-db 100
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_100s.json' --default-example-file-name examples_with_synthetic_data_v5_mono_nl_by_t5_gen_100s.json

# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-100s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-100s-bb --force-preprocess --force-train

# 200 samples
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5_mono/spider/database --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_200s.json --samples-by-db 200
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_200s.json' --default-example-file-name examples_with_synthetic_data_v5_mono_nl_by_t5_gen_200s.json

# w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-200s-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-200s-bb --force-preprocess --force-train

# all samples
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/databases/v5_mono/spider/database --output-data-file ./data/spider/spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_full.json --samples-by-db -1
python3 scripts/split_spider_by_db.py --examples-paths 'spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_full.json' --default-example-file-name examples_with_synthetic_data_v5_mono_nl_by_t5_gen_full.json

# full training from scratch w/ concatenated data w/ batch balancing
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-ori-train-plus-synthetic-data-v5-mono-nl-by-gen-full-train-only-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-ori-train-plus-synthetic-data-v5-mono-nl-by-gen-full-train-only-bb --force-preprocess --force-train

# training on synthetic data only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-synthetic-data-v5-mono-nl-by-gen-full-train-only-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-synthetic-data-v5-mono-nl-by-gen-full-train-only-bb --force-preprocess --force-train

# w/ world_1 only
# w/ batch balancing
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-world-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-world-1-bb --force-preprocess --force-train

# w/ car_1 only
# w/ batch balancing
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-car-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-initialized-train-plus-dev-synthetic-data-v5-mono-nl-by-gen-car-1-bb --force-preprocess --force-train

# 1-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_1shot.json --samples-by-db 1
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_1shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_1shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-1shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-1shot --force-preprocess --force-train

# 5-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_5shot.json --samples-by-db 5
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_5shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_5shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-5shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-5shot --force-preprocess --force-train

# 10-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_10shot.json --samples-by-db 10
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_10shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_10shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-10shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-10shot --force-preprocess --force-train

# 20-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_20shot.json --samples-by-db 20
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_20shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_20shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-20shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-20shot --force-preprocess --force-train

# 50-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_50shot.json --samples-by-db 50
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_50shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_50shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-50shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-50shot --force-preprocess --force-train

# 100-shot
python scripts/data_aug/collect_synthetic_data_template_scfg.py --file-type json --files-folder-path /mnt/shared/parthur/experiments/nl2sql/output/data/v3_fixed/database --output-data-file ./data/spider/train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_100shot.json --samples-by-db 100
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data train_synthetic_data_by_template_scfg_v3_fixed_val_db_only_100shot.json --aug-suffix spider_synthetic_data_template_scfg_v3_fixed_val_db_only_100shot

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-100shot.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-original-plus-synthetic-data-v3-fixed-val-db-only-100shot --force-preprocess --force-train

# *** Experiments for adding dev data into training data

# train on original data but evaluate on splitted val data
# 5-5
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_5_5_half2.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev55-duorat-new-db-content.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev55-duorat-new-db-content.output --gold-txt-file ./data/spider/dev_split_5_5_gold.txt --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev55-duorat-new-db-content.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_split_5_5_gold.txt --pred ./logdir/duorat-new-db-content-bs4-ac7/val-dev55-duorat-new-db-content.output.txt --etype match --db ./data/database --table ./data/spider/tables.json
# 4-6
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_4_6_half2.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev46-duorat-new-db-content.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev46-duorat-new-db-content.output --gold-txt-file ./data/spider/dev_split_4_6_gold.txt --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev46-duorat-new-db-content.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_split_4_6_gold.txt --pred ./logdir/duorat-new-db-content-bs4-ac7/val-dev46-duorat-new-db-content.output.txt --etype match --db ./data/database --table ./data/spider/tables.json
# 3-7
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_3_7_half2.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev37-duorat-new-db-content.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev37-duorat-new-db-content.output --gold-txt-file ./data/spider/dev_split_3_7_gold.txt --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev37-duorat-new-db-content.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_split_3_7_gold.txt --pred ./logdir/duorat-new-db-content-bs4-ac7/val-dev37-duorat-new-db-content.output.txt --etype match --db ./data/database --table ./data/spider/tables.json
# 2-8
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_2_8_half2.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev28-duorat-new-db-content.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev28-duorat-new-db-content.output --gold-txt-file ./data/spider/dev_split_2_8_gold.txt --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-dev28-duorat-new-db-content.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_split_2_8_gold.txt --pred ./logdir/duorat-new-db-content-bs4-ac7/val-dev28-duorat-new-db-content.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# split dev randomly
# Spider

# 5-5
python scripts/split_dev.py --dev-json-file ./data/spider/dev.json --split-json-file-prefix ./data/spider/dev_split_5_5 --split-rate 0.5
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1.json' --default-example-file-name examples_plus_dev55.json
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data dev_split_5_5_half2.json --aug-suffix dev_split_5_5

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-train-plus-dev55.jsonnet --logdir ./logdir/duorat-spider-new-db-content-train-plus-dev55 --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-train-plus-dev55.log &

# ** split validation data into two halves:
# half 1: will be used at different rates, e.g., 50%, 40%, 30%, 20%, 10%, 5%
# half 2: keep unchanged for evaluation only

# evaluate on half 2 only without retraining
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_5_5_half2.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/val-split55-half2-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/val-split55-half2-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_split_5_5_gold.txt --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/val-split55-half2-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_split_5_5_gold.txt --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/val-split55-half2-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# 50% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1.json' --default-example-file-name examples_plus_dev55.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1.json' --default-example-file-name examples_dev55.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev55.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev55 --force-preprocess --force-train

# continuous training on dev55 part only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev55.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev55 --force-preprocess --force-train

# 40% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1_split04.json' --default-example-file-name examples_plus_dev45.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1_split04.json' --default-example-file-name examples_dev45.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev45.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev45 --force-preprocess --force-train

# continuous training on dev45 part only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev45.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev45 --force-preprocess --force-train

# 30% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1_split03.json' --default-example-file-name examples_plus_dev35.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1_split03.json' --default-example-file-name examples_dev35.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev35.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev35 --force-preprocess --force-train

# continuous training on dev35 part only
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev35.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev35 --force-preprocess --force-train

# 20% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1_split02.json' --default-example-file-name examples_plus_dev25.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1_split02.json' --default-example-file-name examples_dev25.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev25.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev25 --force-preprocess --force-train

# continuous training on dev25 part only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev25.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev25 --force-preprocess --force-train

# 10% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1_split01.json' --default-example-file-name examples_plus_dev15.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1_split01.json' --default-example-file-name examples_dev15.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev15.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev15 --force-preprocess --force-train

# continuous training on dev15 part only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev15.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev15 --force-preprocess --force-train

# 5% validation data added to the original training data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_5_5_half1_split005.json' --default-example-file-name examples_plus_dev055.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_split_5_5_half1_split005.json' --default-example-file-name examples_dev055.json

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev055.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-train-plus-dev055 --force-preprocess --force-train

# continuous training on dev055 part only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev055.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev055 --force-preprocess --force-train

# 2-8
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-train-plus-dev55.jsonnet --logdir ./logdir/duorat-spider-new-db-content-train-plus-dev55 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_split_5_5_half2.json --output-eval-file ./logdir/duorat-spider-new-db-content-train-plus-dev55/val-dev55-duorat-new-db-content.output

# 4-6
python scripts/split_dev.py --dev-json-file ./data/spider/dev.json --split-json-file-prefix ./data/spider/dev_split_4_6 --split-rate 0.4
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_4_6_half1.json' --default-example-file-name examples_plus_dev46.json
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data dev_split_4_6_half2.json --aug-suffix dev_split_4_6

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-train-plus-dev46.jsonnet --logdir ./logdir/duorat-spider-new-db-content-train-plus-dev46 --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-train-plus-dev46.log &

# 3-7
python scripts/split_dev.py --dev-json-file ./data/spider/dev.json --split-json-file-prefix ./data/spider/dev_split_3_7 --split-rate 0.3
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_3_7_half1.json' --default-example-file-name examples_plus_dev37.json
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data dev_split_3_7_half2.json --aug-suffix dev_split_3_7

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-train-plus-dev37.jsonnet --logdir ./logdir/duorat-spider-new-db-content-train-plus-dev37 --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-train-plus-dev37.log &

# 2-8
python scripts/split_dev.py --dev-json-file ./data/spider/dev.json --split-json-file-prefix ./data/spider/dev_split_2_8 --split-rate 0.2
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider.json,train_others.json,dev_split_2_8_half1.json' --default-example-file-name examples_plus_dev28.json
python3 scripts/split_spider_by_db.py --examples-paths '' --aug-data dev_split_2_8_half2.json --aug-suffix dev_split_2_8

CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-train-plus-dev28.jsonnet --logdir ./logdir/duorat-spider-new-db-content-train-plus-dev28 --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-content-train-plus-dev28.log &

# * Experiments for catastrophic forgetting

# split dev data given a specific database

# world_1
# 100 examples for continuous training, 20 examples for testing
python scripts/split_dev_by_dbs.py --dev-json-file ./data/spider/dev.json --dev-json-output-file-prefix ./data/spider/dev --dbs world_1 --split-rate 100
python3 scripts/split_spider_by_db.py --examples-paths 'dev_with_world_1_train.json' --default-example-file-name examples_dev_world_1_train.json

# infer previously trained system on two sets

# ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_world_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_world_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_world_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_world_1.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_world_1_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_wo_world_1.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_wo_world_1.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_wo_world_1_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_wo_world_1_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# continuous train on ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1 --force-preprocess --force-train

# infer on ./data/spider/dev_with_world_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.output --gold-txt-file ./data/spider/dev_with_world_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# continuous train on original training (as data regularization) + ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1 --force-preprocess --force-train

# infer on ./data/spider/dev_with_world_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output --gold-txt-file ./data/spider/dev_with_world_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# infer on ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output --gold-txt-file ./data/spider/dev_with_world_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# continuous train on original training (as data regularization) + ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb --force-preprocess --force-train

# infer on ./data/spider/dev_with_world_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output --gold-txt-file ./data/spider/dev_with_world_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# infer on ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output --gold-txt-file ./data/spider/dev_with_world_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-world-1-bb.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# car_1
# 72 examples for continuous training, 20 examples for testing
python scripts/split_dev_by_dbs.py --dev-json-file ./data/spider/dev.json --dev-json-output-file-prefix ./data/spider/dev --dbs car_1 --split-rate 72
python3 scripts/split_spider_by_db.py --examples-paths 'dev_with_car_1_train.json' --default-example-file-name examples_dev_car_1_train.json
python scripts/split_dev_by_dbs.py --dev-json-file ./data/spider/dev.json --dev-json-output-file-prefix ./data/spider/dev --dbs world_1 car_1 --split-rate 0.9

# infer previously trained system on two sets

# ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_car_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_car_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_car_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_car_1.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_with_car_1_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-with-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_wo_car_1.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_wo_car_1.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_wo_car_1_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_wo_car_1_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# continuous train on ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1 --force-preprocess --force-train

# infer on ./data/spider/dev_with_car_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1.output --gold-txt-file ./data/spider/dev_with_car_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-car-1.output.txt

# continuous train on original training (as data regularization) + ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1 --force-preprocess --force-train

# infer on ./data/spider/dev_with_car_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output --gold-txt-file ./data/spider/dev_with_car_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# infer on ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output --gold-txt-file ./data/spider/dev_with_car_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# continuous train on original training (as data regularization) + ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb --force-preprocess --force-train

# infer on ./data/spider/dev_with_car_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output --gold-txt-file ./data/spider/dev_with_car_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# infer on ./data/spider/dev_with_car_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output --gold-txt-file ./data/spider/dev_with_car_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb/dev-with-car-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-original-plus-dev-with-car-1-bb.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_wo_world_1_car_1.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps --db-folder-path ./data/database/ --eval-file ./data/spider/dev_wo_world_1_car_1.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output --gold-txt-file ./data/spider/dev_wo_world_1_car_1_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_wo_world_1_car_1_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps/dev-wo-world-1-car-1-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_world_1_train.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_train.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output --gold-txt-file ./data/spider/dev_with_world_1_train_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_train_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_world_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_world_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output --gold-txt-file ./data/spider/dev_with_world_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_world_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-world-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# ./data/spider/dev_with_car_1_test.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_car_1_test.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output --gold-txt-file ./data/spider/dev_with_car_1_test_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_with_car_1_test_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1/dev-with-car-1-test-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-continuous-train-dev-with-world-1-then-car-1.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# *** Get prediction errors

# db_id=concert_singer
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.concert_singer concert_singer

# db_id=pets_1
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.pets_1 pets_1

# db_id=car_1
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.car_1 car_1

# db_id=department_management (perfect)
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.department_management department_management

# db_id=course_teach
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.course_teach course_teach

# db_id=student_transcripts_tracking
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.student_transcripts_tracking student_transcripts_tracking

# db_id=flight_2
python scripts/get_output_errors.py ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.eval ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.errors.flight_2 flight_2

# *** CustomNER for NL2SQL

# run interactive mode for error analysis
CUDA_VISIBLE_DEVICES=0 python scripts/interactive.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./data/database/concert_singer/concert_singer.sqlite --schema-path ./data/database/concert_singer/tables.json

CUDA_VISIBLE_DEVICES=0 python scripts/interactive.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./data/database/pets_1/pets_1.sqlite --schema-path ./data/database/pets_1/tables.json

CUDA_VISIBLE_DEVICES=0 python scripts/interactive.py --logdir ./logdir/duorat-new-db-content-bs4-ac7 --config configs/duorat/duorat-new-db-content.jsonnet --db-path ./data/database/car_1/car_1.sqlite --schema-path ./data/database/car_1/tables.json

# get SLML outputs

# dev
python scripts/get_slml_outputs.py --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --input-files ./data/spider/dev.json  --output-file ./data/spider/dev_with_unsup_slml.json

CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_unsup_slml.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-slml.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-slml.output --gold-txt-file ./data/spider/dev_gold.sql --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-slml.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_gold.sql --pred ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-slml.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# baseline
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.output --gold-txt-file ./data/spider/dev_gold.sql --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_gold.sql --pred ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-bs4-ac7.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# w/ human-corrected SLML
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-new-db-content.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev_with_human_slml.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-human-slml.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-human-slml.output --gold-txt-file ./data/spider/dev_gold.sql --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-human-slml.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_gold.sql --pred ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-human-slml.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# train
python scripts/get_slml_outputs.py --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --input-files ./data/spider/train_spider.json ./data/spider/train_others.json  --output-file ./data/spider/train_spider_and_others_with_unsup_slml.json

## Flatten NER

# for MeNER
# train
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/train_spider_and_others_with_schema_custom_ner.json --output-file ./data/spider/train_spider_plus_others_flatten_schema_ner.txt --schema-json-file ./data/spider/tables.json --ner-type ner_hf --data-type train
# dev
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-file ./data/spider/dev_flatten_schema_ner.txt --schema-json-file ./data/spider/tables.json --ner-type ner_hf  --data-type dev

# for HF's NER
# train
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/train_spider_and_others_with_schema_custom_ner.json --output-file ./data/spider/train_spider_plus_others_flatten_schema_ner.json --schema-json-file ./data/spider/tables.json --ner-type ner_hf --data-type train --output-file-ext json
# dev
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-file ./data/spider/dev_flatten_schema_ner.json --schema-json-file ./data/spider/tables.json --ner-type ner_hf  --data-type dev  --output-file-ext json

# HF's NER
python3 run_ner.py \
  --model_name_or_path bert-base-uncased \
  --train_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider/train_spider_plus_others_flatten_schema_ner.json \
  --validation_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider/dev_flatten_schema_ner.json \
  --test_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider/dev_flatten_schema_ner.json \
  --output_dir ./exp/ner/spider/flatten_schema_ner_model_1 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --return_entity_level_metrics \
  --metric_for_best_model eval_loss \
  --label_smoothing_factor 0.0 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --do_train \
  --do_eval --do_predict

# MeNER
# long_text_processing.mode = truncating; max_seq_len 256
CUDA_VISIBLE_DEVICES=3 python3 -m mener \
--config ner_config_multilingual.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider \
--data.train_file train_spider_plus_others_flatten_schema_ner.txt \
--data.eval_file dev_flatten_schema_ner.txt \
--data.augmented_data_file.file_list [] \
--feature_extractor.long_text_processing.max_seq_len 256 \
--feature_extractor.long_text_processing.mode truncating \
--mener_model.batch_size 8 \
--mener_model.subword_label_strategy 2 \
--mener_model.gazetteer_layer.enable False \
--mener_model.birnn_model.rnn_units [512] \
--mener_model.model_dir ./exp/models/baseline_1mlp_bert_base_uncased_spider_flatten_schema_ner_ltp_tr_mlen256_sls2 \
--evaluation.evaluation_csv_test_files [] --evaluation.evaluation_output_files []

# long_text_processing.mode = truncating; max_seq_len 512
CUDA_VISIBLE_DEVICES=0 python3 -m mener \
--config ner_config_multilingual.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider \
--data.train_file train_spider_plus_others_flatten_schema_ner.txt \
--data.eval_file dev_flatten_schema_ner.txt \
--data.augmented_data_file.file_list [] \
--feature_extractor.long_text_processing.max_seq_len 512 \
--feature_extractor.long_text_processing.mode truncating \
--mener_model.batch_size 4 \
--mener_model.subword_label_strategy 2 \
--mener_model.gazetteer_layer.enable False \
--mener_model.birnn_model.rnn_units [512] \
--mener_model.model_dir ./exp/models/baseline_1mlp_bert_base_uncased_spider_flatten_schema_ner_ltp_tr_mlen512_sls2 \
--evaluation.evaluation_csv_test_files [] --evaluation.evaluation_output_files []

# long_text_processing.mode = overlapping; max_seq_len 64
CUDA_VISIBLE_DEVICES=0 python3 -m mener \
--config ner_config_multilingual.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/spider \
--data.train_file train_spider_plus_others_flatten_schema_ner.txt \
--data.eval_file dev_flatten_schema_ner.txt \
--data.augmented_data_file.file_list [] \
--feature_extractor.long_text_processing.max_seq_len 64 \
--feature_extractor.long_text_processing.mode overlapping \
--mener_model.batch_size 32 \
--mener_model.subword_label_strategy 2 \
--mener_model.gazetteer_layer.enable False \
--mener_model.birnn_model.rnn_units [512] \
--mener_model.model_dir ./exp/models/baseline_1mlp_bert_base_uncased_spider_flatten_schema_ner_ltp_op_mlen64_sls2 \
--evaluation.evaluation_csv_test_files [] --evaluation.evaluation_output_files []

# * Filtering bad tokens for matching (simple heuristic)

# infer only
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-schema-linker-bad-match-filtering.jsonnet --logdir ./logdir/duorat-new-db-content-bs4-ac7 --db-folder-path ./data/database/ --eval-file ./data/spider/dev.json --output-eval-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-schema-linker-bad-match-filtering.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-schema-linker-bad-match-filtering.output --gold-txt-file ./data/spider/dev_gold.sql --output-preds-txt-file ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-schema-linker-bad-match-filtering.output.txt
python -m third_party.spider.evaluation --gold ./data/spider/dev_gold.sql --pred ./logdir/duorat-new-db-content-bs4-ac7/val-duorat-new-db-content-with-schema-linker-bad-match-filtering.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# re-train
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-schema-linker-bad-match-filtering.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-schema-linker-bad-match-filtering --force-preprocess --force-train &> logdir/train-duorat-spider-new-db-content-with-schema-linker-bad-match-filtering.log &

# create silver training data for custom NER model
# train
TOKENIZERS_PARALLELISM=true python scripts/custom_ner/create_silver_training_data.py --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --input-files ./data/spider/train_spider.json ./data/spider/train_others.json --output-file ./data/spider/train_spider_and_others_with_schema_custom_ner.json --schema-json-path ./data/spider/tables.json

rm -rf ./data/custom_ner/spider/train
mkdir -p ./data/custom_ner/spider/train
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/train_spider_and_others_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/train --data-type spider_train

# --split-k 0.5
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/train_spider_and_others_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/train --data-type spider_train --split-k 0.5
python ./scripts/custom_ner/get_spider_test_data_with_slml.py ./data/custom_ner/spider/train/spider_train_set_train_split0.5_wo_slmls.json ./logdir/cner_models/spider/train/ 0.5 spider_dev ./data/custom_ner/spider/train/spider_train_set_train_split0.5_w_predicted_slmls.json


# * train customNER model (MeNER) on silver training data
source /mnt/shared/vchoang/tools/pyvenv368-oda-custom-ner-master/bin/activate

# cinema
CUDA_VISIBLE_DEVICES=0 python3 -m mener \
--config ../cner/cner_config.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/train \
--data.train_file cinema_db_spider_train_set_train_split.txt \
--data.test_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/train/cinema_db_spider_train_set_test_split.txt \
--data.format row \
--mener_model.model_dir /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/cner_models/spider/train/cinema \
--evaluation.ner_tag_seq_format bare

# movie_1
CUDA_VISIBLE_DEVICES=0 python3 -m mener \
--config ../cner/cner_config.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/train \
--data.train_file movie_1_db_spider_train_set_train_split.txt \
--data.test_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/train/movie_1_db_spider_train_set_test_split.txt \
--data.format row \
--mener_model.model_dir /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/cner_models/spider/train/movie_1 \
--evaluation.ner_tag_seq_format bare

# employee_hire_evaluation
CUDA_VISIBLE_DEVICES=0 python3 -m mener \
--config ../cner/cner_config.yaml \
--logging_level 0 \
--data.data_path /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/dev \
--data.train_file employee_hire_evaluation_db_spider_dev_set_train_split.txt \
--data.test_file /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/data/custom_ner/spider/dev/employee_hire_evaluation_db_spider_dev_set_test_split.txt \
--data.format row \
--mener_model.model_dir /mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/logdir/cner_models/spider/dev/employee_hire_evaluation \
--evaluation.ner_tag_seq_format bare

# dev
TOKENIZERS_PARALLELISM=true python scripts/custom_ner/create_silver_training_data.py --duorat-config-file ./configs/duorat/duorat-new-db-content.jsonnet --input-files ./data/spider/dev.json --output-file ./data/spider/dev_with_schema_custom_ner.json --schema-json-path ./data/spider/tables.json

rm -rf ./data/custom_ner/spider/dev
mkdir -p ./data/custom_ner/spider/dev
# --split-k 0.8
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/dev --data-type spider_dev
# --split-k 0.5
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/dev --data-type spider_dev --split-k 0.5
# --split-k 0.4
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/dev --data-type spider_dev --split-k 0.4
# --split-k 0.3
python scripts/custom_ner/extract_custom_ner_data.py --input-file ./data/spider/dev_with_schema_custom_ner.json --output-folder ./data/custom_ner/spider/dev --data-type spider_dev --split-k 0.3

# * train end-to-end w/ custom NER silver data
python3 scripts/split_spider_by_db.py --examples-paths 'train_spider_and_others_with_schema_custom_ner.json' --default-example-file-name examples_with_schema_custom_ner_silver_data.json
python3 scripts/split_spider_by_db.py --examples-paths 'dev_with_schema_custom_ner.json' --default-example-file-name examples_with_schema_custom_ner_silver_data.json

# * evaluate

# w/o unsupervised schema linking
# --split-k 0.5
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base --db-folder-path ./data/database/ --eval-file  ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_wo_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt --etype match --db ./data/database --table ./data/spider/tables.json
# --split-k 0.4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_wo_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt --etype match --db ./data/database --table ./data/spider/tables.json
# --split-k 0.3
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_wo_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# w/ supervised schema linking

# gold schema entities
# --split-k 0.5
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_w_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# --split-k 0.4
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_w_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# --split-k 0.3
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_w_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# predicted schema entities
# --split-k 0.5
python ./scripts/custom_ner/get_spider_test_data_with_slml.py ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_wo_slmls.json ./logdir/cner_models/spider/dev/ 0.5 spider_dev ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_w_predicted_slmls.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_w_predicted_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.5_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split05-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# --split-k 0.4
python ./scripts/custom_ner/get_spider_test_data_with_slml.py ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_wo_slmls.json ./logdir/cner_models/spider/dev/ 0.4 spider_dev ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_w_predicted_slmls.json
CUDA_VISIBLE_DEVICES=3 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_w_predicted_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.4_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split04-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# --split-k 0.3
python ./scripts/custom_ner/get_spider_test_data_with_slml.py ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_wo_slmls.json ./logdir/cner_models/spider/dev/ 0.3 spider_dev ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_w_predicted_slmls.json
CUDA_VISIBLE_DEVICES=0 python scripts/infer_one.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4 --db-folder-path ./data/database/ --eval-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_w_predicted_slmls.json --output-eval-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output
python scripts/get_preds_from_json_file.py --preds-json-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output --gold-txt-file ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --output-preds-txt-file ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt
python -m third_party.spider.evaluation --gold ./data/custom_ner/spider/dev/spider_dev_set_train_split0.3_gold.sql --pred ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs-run4/val-split03-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-predicted-slml-inputs-run4.output.txt --etype match --db ./data/database --table ./data/spider/tables.json

# Idea: replace default unsupervised schema linking by custom NER schema silver data
# BERT large
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-new-db-content-with-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-new-db-content-with-silver-slml-inputs --force-preprocess --force-train

# ELECTRA base
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-silver-slml-inputs --force-preprocess --force-train

# * Evaluating unsupervised schema linking by alternating match types

# table only
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-only.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-only --force-preprocess --force-train

# column only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-column-only.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-column-only --force-preprocess --force-train

# value only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-value-only.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-value-only --force-preprocess --force-train

# table+column
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-column.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-column --force-preprocess --force-train

# table+value
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-value.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-table-value --force-preprocess --force-train

# column+value
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-column-value.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-unsup-schema-linker-match-type-column-value --force-preprocess --force-train

# * Evaluate pretrained embeddings

# ** DistilBERT

# distilbert-base-uncased
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased --force-preprocess --force-train

# distilbert-base-uncased-distilled-squad
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased-distilled-squad.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased-distilled-squad --force-preprocess --force-train

# distilbert-base-uncased-finetuned-sst-2-english
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased-finetuned-sst-2-english.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-distilbert-base-uncased-finetuned-sst-2-english --force-preprocess --force-train

# ** RoBERTa

# base
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-roberta-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-roberta-base --force-preprocess --force-train

# large
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-roberta-large.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-roberta-large --force-preprocess --force-train

# ** ELECTRA

# small
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-small.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-small --force-preprocess --force-train

# base
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base --force-preprocess --force-train

# infer
python scripts/infer.py --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base --section train --output ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base/train-duorat-spider-new-db-content-with-pretrained-embeddings-electra-base.output --force

# base w/ unsup schema linking for table + column only
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-unsup-schema-linker-match-type-table-column.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-unsup-schema-linker-match-type-table-column --force-preprocess --force-train

# large
CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-large.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-large --force-preprocess --force-train

# GraPPa
TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-grappa.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-grappa --force-preprocess --force-train

# ** T5

# small
TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-t5-small.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-t5-small --force-preprocess --force-train &>./logdir/train-duorat-spider-new-db-ƒcontent-with-pretrained-embeddings-t5-small.log &

# base
TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=3 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-t5-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-t5-base --force-preprocess --force-train &> ./logdir/train-duorat-spider-new-db-content-with-pretrained-embeddings-t5-base.log &

# ** BART

# base
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-bart-base.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-bart-base --force-preprocess --force-train

# base (150K steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-bart-base-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-bart-base-150k-steps --force-preprocess --force-train

# *** Examine schema ordering

# base
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-schema-order-tctc.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-electra-base-150k-steps-schema-order-tctc --force-preprocess --force-train

# *** Focusing model

# focusing BERT (base, uncased)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-base-uncased-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-base-uncased-150k-steps --force-preprocess --force-train

# focusing BERT (large, uncased)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-large-uncased-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-large-uncased-150k-steps --force-preprocess --force-train

CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duorat/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-base-uncased-5e-hf-150k-steps.jsonnet --logdir ./logdir/duorat-spider-new-db-content-with-pretrained-embeddings-focusing-bert-base-uncased-5e-hf-150k-steps-hf --force-preprocess --force-train
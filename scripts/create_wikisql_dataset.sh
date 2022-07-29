# coding=utf-8
#~/usr/bin/env bash

# This script must be executed in the duorat's root dir.

MICHIGAN_GITHUB=https://raw.githubusercontent.com/jkkummerfeld/text2sql-data/master
CODE=$PWD
DATA_DIR=${CODE}/data

# WikiSQL
function go_wikisql {
    mkdir ${DATA_DIR}/wikisql && cd ${DATA_DIR}/wikisql

    # download from $MICHIGAN_GITHUB/data/wikisql.json.bz2
    wget $MICHIGAN_GITHUB/data/wikisql.json.bz2 || exit
    bzip2 -d wikisql.json.bz2

    # download from WikiSQL github
    git clone https://github.com/salesforce/WikiSQL.git || exit
    cd WikiSQL
    bzip2 -d data.tar.bz2
    tar -xvf data.tar
    cd ../..
    cd -

    python $CODE/scripts/get_tables_wikisql.py\
        ${DATA_DIR}/wikisql/WikiSQL/data/train.db\
        ${DATA_DIR}/wikisql/WikiSQL/data/train.tables.jsonl\
        ${DATA_DIR}/wikisql/tables_train.json
    python $CODE/scripts/get_tables_wikisql.py\
        ${DATA_DIR}/wikisql/WikiSQL/data/dev.db\
        ${DATA_DIR}/wikisql/WikiSQL/data/dev.tables.jsonl\
        ${DATA_DIR}/wikisql/tables_dev.json
    python $CODE/scripts/get_tables_wikisql.py\
        ${DATA_DIR}/wikisql/WikiSQL/data/test.db\
        ${DATA_DIR}/wikisql/WikiSQL/data/test.tables.jsonl\
        ${DATA_DIR}/wikisql/tables_test.json

    python $CODE/scripts/convert_from_ori_wikisql.py --input ${DATA_DIR}/wikisql/WikiSQL/data/train.jsonl \
        --db-schema ${DATA_DIR}/wikisql/tables_train.json --output ${DATA_DIR}/wikisql/examples_train.json
    python $CODE/scripts/convert_from_ori_wikisql.py --input ${DATA_DIR}/wikisql/WikiSQL/data/dev.jsonl\
        --db-schema ${DATA_DIR}/wikisql/tables_dev.json --output ${DATA_DIR}/wikisql/examples_dev.json
    python $CODE/scripts/convert_from_ori_wikisql.py --input ${DATA_DIR}/wikisql/WikiSQL/data/test.jsonl \
        --db-schema ${DATA_DIR}/wikisql/tables_test.json --output ${DATA_DIR}/wikisql/examples_test.json
}

rm -rf $DATA_DIR/wikisql
go_wikisql
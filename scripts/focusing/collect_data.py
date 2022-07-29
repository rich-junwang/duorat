import json
from typing import List
import tqdm
import os
import csv


def collect_spider_wikisql(json_data_files: List[str],
                           json_schema_files: List[str],
                           output_file: str,
                           data_type: str = 'spider',
                           use_col_type: bool = False) -> None:
    # load data
    spider_data = []
    for data_file in json_data_files:
        with open(data_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
            spider_data.extend(data)

    # load schema
    schema_data = {}
    for json_schema_file in json_schema_files:
        with open(json_schema_file) as f:
            data = json.load(f)
            for entry in tqdm.tqdm(data):
                schema_data[entry['db_id']] = []

                # column type and names
                if use_col_type:
                    for col_type, col_name in zip(entry['column_types'][1:], entry['column_names'][1:]):
                        schema_data[entry['db_id']].append(f"{str(col_type)} {str(col_name[1])}")
                else:
                    for col_name in entry['column_names'][1:]:
                        schema_data[entry['db_id']].append(f"{str(col_name[1])}")

                if data_type == 'spider':
                    # table names
                    schema_data[entry['db_id']].extend([table_name for table_name in entry['table_names']])

    output_data = []
    for example in tqdm.tqdm(spider_data):
        question = example['question'].strip()
        db_id = example['db_id']
        db_schema = schema_data[db_id]

        concat_output = [question]
        for db_info in db_schema:
            concat_output.append('</s>')
            concat_output.append(db_info)

        output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in output_data:
            outf.write(f"{entry}\n")

    return


def collect_wikitablequestions(dataset_path: str, tsv_data_file_name: str, output_file: str) -> None:
    output_data = []
    with open(os.path.join(dataset_path, tsv_data_file_name)) as f:
        f.readline()  # ignore the TSV header
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            question = parts[1]
            table_csv_file = parts[2]
            csv_table_file = os.path.join(dataset_path, table_csv_file)
            with open(csv_table_file, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    col_list = [l.strip().replace('\n', ' ').replace('  ', '') for l in line]
                    break

            concat_output = [question]
            for col in col_list:
                col = col.strip()
                if col == '':
                    continue
                concat_output.append('</s>')
                concat_output.append(col)

            output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


def collect_tabfact_tsv(tsv_data_file: str, data_path: str, output_file: str) -> None:
    tsv_data_file_path = os.path.join(data_path, tsv_data_file)
    csv_tables_folder_path = os.path.join(data_path, 'data/all_csv')
    output_data = []
    with open(tsv_data_file_path) as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            text = parts[-2]
            csv_tables_file = os.path.join(csv_tables_folder_path, parts[0])
            with open(csv_tables_file) as tf:
                header_line = tf.readline().strip()
                col_list = header_line.split('#')

            concat_output = [text]
            for col in col_list:
                col = col.strip()
                if col == '':
                    continue
                concat_output.append('</s>')
                concat_output.append(col)

            output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


def collect_tabfact_json(json_data_file: str, data_path: str, output_file: str) -> None:
    json_data_file_path = os.path.join(data_path, json_data_file)
    csv_tables_folder_path = os.path.join(data_path, 'data/all_csv')
    output_data = []
    with open(json_data_file_path) as f:
        json_data = json.load(f)
        for csv_file_id, data in tqdm.tqdm(json_data.items()):
            csv_tables_file = os.path.join(csv_tables_folder_path, csv_file_id)
            with open(csv_tables_file) as tf:
                header_line = tf.readline().strip()
                col_list = header_line.split('#')

            text_list = data[0]
            for text in text_list:
                concat_output = [text]
                for col in col_list:
                    col = col.strip()
                    if col == '':
                        continue
                    concat_output.append('</s>')
                    concat_output.append(col)
                output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


def collect_hybridqa(json_data_files: List[str], dataset_path: str, output_file: str) -> None:
    json_tables_folder_path = os.path.join(dataset_path, 'WikiTables-WithLinks/tables_tok')
    output_data = []
    for json_data_file in json_data_files:
        json_data_file_path = os.path.join(dataset_path, json_data_file)
        with open(json_data_file_path) as fdata:
            data = json.load(fdata)  # a list
            for entry in tqdm.tqdm(data):
                question = entry["question"]
                table_id = entry["table_id"]
                json_table_file = os.path.join(json_tables_folder_path, f"{table_id}.json")
                with open(json_table_file) as ftab:
                    tab_data = json.load(ftab)  # a dict
                    header = tab_data["header"]
                    col_list = [col_info[0] for col_info in header]

                concat_output = [question]
                for col in col_list:
                    col = col.strip()
                    if col == '':
                        continue
                    concat_output.append('</s>')
                    concat_output.append(col)

                output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


def collect_totto(jsonl_data_files: List[str], dataset_path: str, output_file: str) -> None:
    output_data = []
    for jsonl_data_file in jsonl_data_files:
        jsonl_data_file_path = os.path.join(dataset_path, jsonl_data_file)
        with open(jsonl_data_file_path) as f:
            for line in f:
                line = line.strip()
                json_data = json.loads(line)
                final_sentence_annotation = json_data["sentence_annotations"][0]["final_sentence"]
                table = json_data["table"]
                header = table[0]
                col_list = [col_info["value"] for col_info in header if col_info["is_header"]]
                if col_list:
                    concat_output = [final_sentence_annotation]
                    for col in col_list:
                        col = col.strip()
                        if col == '':
                            continue
                        concat_output.append('</s>')
                        concat_output.append(col)

                    output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


def collect_logicnlg(json_data_files: List[str], dataset_path: str, output_file: str) -> None:
    output_data = []
    for json_data_file in json_data_files:
        json_data_file_path = os.path.join(dataset_path, json_data_file)
        with open(json_data_file_path) as fdat:
            json_data = json.load(fdat)
            for csv_tab_id, data in tqdm.tqdm(json_data.items()):
                text_list = [entry[0] for entry in data]
                csv_tab_file_path = os.path.join(dataset_path, f"data/all_csv/{csv_tab_id}")
                if os.path.exists(csv_tab_file_path):
                    with open(csv_tab_file_path) as tf:
                        header_line = tf.readline().strip()
                        col_list = header_line.split('#')

                    for text in text_list:
                        concat_output = [text]
                        for col in col_list:
                            col = col.strip()
                            if col == '':
                                continue
                            concat_output.append('</s>')
                            concat_output.append(col)
                        output_data.append(' '.join(concat_output))

    with open(output_file, 'w') as outf:
        for entry in tqdm.tqdm(output_data):
            outf.write(f"{entry}\n")

    return


if __name__ == "__main__":
    print("Collecting Spider...")
    collect_spider_wikisql(json_data_files=['./data/spider/train_spider.json', './data/spider/train_others.json'],
                           json_schema_files=['./data/spider/tables.json'],
                           output_file='./data/focusing/spider_train_nl.txt')
    collect_spider_wikisql(json_data_files=['./data/spider/dev.json'],
                           json_schema_files=['./data/spider/tables.json'],
                           output_file='./data/focusing/spider_dev_nl.txt')
    print("Collecting Spider (synthetic)...")
    collect_spider_wikisql(json_data_files=['./data/spider/spider_all_dbs_synthetic_data_v5_mono_nl_by_t5_gen_full.json'],
                           json_schema_files=['./data/spider/tables.json'],
                           output_file='./data/focusing/spider_synthetic_nl.txt')

    print("Collecting WikiSQL...")
    collect_spider_wikisql(json_data_files=['./data/wikisql/examples_train.json',
                                            './data/wikisql/examples_dev.json',
                                            './data/wikisql/examples_test.json'],
                           json_schema_files=['./data/wikisql/tables_train.json',
                                              './data/wikisql/tables_dev.json',
                                              './data/wikisql/tables_test.json'],
                           output_file='./data/focusing/wikisql_full_nl.txt',
                           data_type='wikisql')

    print("Collecting WikiTableQuestions...")
    collect_wikitablequestions(dataset_path='../../data/focusing/non_sql_tabular_datasets/WikiTableQuestions',
                               tsv_data_file_name='data/training.tsv',
                               output_file='./data/focusing/wikitablequestions.txt')

    print("Collecting TabFact...")
    # collect_tabfact_tsv(tsv_data_file='processed_datasets/tsv_data_horizontal/tabfact_all_data.tsv',
    #                     data_path='../../data/focusing/non_sql_tabular_datasets/Table-Fact-Checking',
    #                     output_file='./data/focusing/tabfact.txt')
    collect_tabfact_json(json_data_file='tokenized_data/total_examples.json',
                         data_path='../../data/focusing/non_sql_tabular_datasets/Table-Fact-Checking',
                         output_file='./data/focusing/tabfact.txt')

    print("Collecting HybridQA...")
    collect_hybridqa(json_data_files=['released_data/train.json', 'released_data/dev.json', 'released_data/test.json'],
                     dataset_path='../../data/focusing/non_sql_tabular_datasets/HybridQA',
                     output_file='./data/focusing/hybridqa.txt')

    print("Collecting ToTTo...")
    collect_totto(jsonl_data_files=['totto_data/totto_train_data.jsonl', 'totto_data/totto_dev_data.jsonl'],
                  dataset_path="../../data/focusing/non_sql_tabular_datasets/ToTTo",
                  output_file="./data/focusing/totto.txt")

    print("Collecting LogicNLG...")
    collect_logicnlg(json_data_files=['data/train_lm.json', 'data/val_lm.json', 'data/test_lm.json'],
                     dataset_path='../../data/focusing/non_sql_tabular_datasets/LogicNLG',
                     output_file='./data/focusing/logicnlg.txt')

# Vu Hoang (vu.hoang@oracle.com)
# Oracle Corp.
# Extract custom NER data


import argparse
import json
import os
import copy

from typing import Dict, List, Set

import tqdm

from sklearn import model_selection


def extract_custom_ner_data(input_file: str,
                            output_folder: str,
                            data_type: str,
                            split_k: float = 0.8):
    data = json.load(open(input_file))
    data_by_db = {}
    for entry in tqdm.tqdm(data):
        db_id = entry["db_id"]
        # question = entry['schema_custom_ner']['toked_question']
        # tags = entry['schema_custom_ner']['tags']

        if db_id not in data_by_db:
            data_by_db[db_id] = [entry]  # [{'toked_question': question, 'tags': tags}]
        else:
            data_by_db[db_id].append(entry)  # {'toked_question': question, 'tags': tags})

    val_data_wo_slmls = []
    val_data_w_slmls = []
    val_gold_sql_data = []
    for db_id, entries in data_by_db.items():
        # split into train/test sections
        n_examples_in_val = int((1.0 - split_k) * len(entries))
        if n_examples_in_val < 1:
            print(f"WARNING: #examples in test has 1 example only for db '{db_id}'.")
            train_entries, test_entries = model_selection.train_test_split(entries,
                                                                           random_state=42,
                                                                           test_size=1)
        else:
            train_entries, test_entries = model_selection.train_test_split(entries,
                                                                           random_state=42,
                                                                           train_size=split_k)

        # Validate the number of labels in test that must exist in train
        def _get_tag_set(entries: List[str]) -> Dict[str, Set[int]]:
            tsets = {}
            for index, entry in enumerate(entries):
                for tag in entry['schema_custom_ner']['tags'].split():
                    if tag in tsets:
                        tsets[tag].add(index)
                    else:
                        tsets[tag] = set([index])
            return tsets

        train_tag_set = set(_get_tag_set(entries=train_entries).keys())
        test_sets = _get_tag_set(entries=test_entries)
        index_mask = set()
        for tag, indices in test_sets.items():
            if tag not in train_tag_set:
                # Here, we accept some entries in test which also appears in train.
                for index in indices:
                    if index not in index_mask:
                        train_entries.append(test_entries[index])
                        index_mask.add(index)

        # train
        fout = open(os.path.join(output_folder, f"{db_id}_db_{data_type}_set_train_split{split_k}.txt"), "w")
        for entry in train_entries:
            fout.write(f"{entry['schema_custom_ner']['toked_question']}\n")
            fout.write(f"{entry['schema_custom_ner']['tags']}\n")
        fout.flush()
        fout.close()

        # test
        fout = open(os.path.join(output_folder, f"{db_id}_db_{data_type}_set_test_split{split_k}.txt"), "w")
        for entry in test_entries:
            fout.write(f"{entry['schema_custom_ner']['toked_question']}\n")
            fout.write(f"{entry['schema_custom_ner']['tags']}\n")
        fout.flush()
        fout.close()

        val_data_w_slmls.extend([copy.deepcopy(entry) for entry in test_entries])
        val_gold_sql_data.extend([f'{entry["query"]}\t{entry["db_id"]}' for entry in test_entries])

        for entry in test_entries:
            del entry["schema_custom_ner"]
            del entry["slml_question"]
            del entry["unsup_slml_question"]
        val_data_wo_slmls.extend([entry for entry in test_entries])

    with open(os.path.join(output_folder, f"{data_type}_set_train_split{split_k}_wo_slmls.json"), "w") as outf:
        json.dump(val_data_wo_slmls, outf, indent=4, sort_keys=False)

    with open(os.path.join(output_folder, f"{data_type}_set_train_split{split_k}_w_slmls.json"), "w") as outf:
        json.dump(val_data_w_slmls, outf, indent=4, sort_keys=False)

    with open(os.path.join(output_folder, f"{data_type}_set_train_split{split_k}_gold.sql"), "w") as outf:
        for sql in val_gold_sql_data:
            outf.write(f"{sql}\n")

    return


def extract_system_ner_data_v1(input_file: str,
                               schema_json_file: str,
                               output_file: str,
                               data_type: str = 'train',
                               output_file_ext: str = 'txt') -> None:
    # load data
    spider_data = []
    with open(input_file) as f:
        data = json.load(f)
        assert isinstance(data, list)
        spider_data.extend(data)

    # load schema
    schema_data = {}
    ignored_dbs = set()
    with open(schema_json_file) as f:
        data = json.load(f)
        for entry in tqdm.tqdm(data):
            schema_data[entry['db_id']] = {'column_map': {}, 'table_map': {}}

            index = 0
            ind2tab = {}
            for table_name, ori_table_name in zip(entry["table_names"], entry["table_names_original"]):
                schema_data[entry['db_id']]['table_map'][f"@{ori_table_name.lower()}"] = (index, str(table_name))
                ind2tab[index] = ori_table_name.lower()
                index += 1

            index = 0
            tab2col = {}
            for col_name, ori_col_name in zip(entry['column_names'][1:], entry['column_names_original'][1:]):
                tab_index = col_name[0]
                schema_data[entry['db_id']]['column_map'][f"@{ind2tab[tab_index]}.{ori_col_name[1].lower()}"] = (
                index, tab_index, str(col_name[1]))
                if f"@{ind2tab[tab_index]}" in tab2col:
                    tab2col[f"@{ind2tab[tab_index]}"].append(f"{ori_col_name[1].lower()}")
                else:
                    tab2col[f"@{ind2tab[tab_index]}"] = [f"{ori_col_name[1].lower()}"]
                index += 1

            for k, v in tab2col.items():
                if len(v) > 20:
                    ignored_dbs.add(entry['db_id'])

    print(f"Ignored dbs: {ignored_dbs}")

    output_data = []
    ner_tag_set = set()
    for data_instance in tqdm.tqdm(spider_data):
        db_id = data_instance['db_id']
        if data_type == 'train' and db_id in ignored_dbs:
            continue

        schema_custom_ner = data_instance['schema_custom_ner']
        ner_tags = schema_custom_ner['tags']
        toked_question = schema_custom_ner['toked_question']

        tag_schema_concat_list = []
        schema_concat_list = []
        for _, col_info in schema_data[db_id]['column_map'].items():
            schema_concat_list.append('[CLS]')
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"C{col_info[0]}")
            tag_schema_concat_list.append(f"O")
            schema_concat_list.append(f"{col_info[2]}")
            tag_schema_concat_list.extend([f"C{col_info[0]}"] * len(col_info[2].split()))

        for _, tab_info in schema_data[db_id]['table_map'].items():
            schema_concat_list.append('[CLS]')
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"T{tab_info[0]}")
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"{tab_info[1]}")
            tag_schema_concat_list.extend([f"T{tab_info[0]}"] * len(tab_info[1].split()))

        ner_tag_list = ner_tags.split()
        indexed_ner_tags = []
        for ner_tag in ner_tag_list:
            if ner_tag == 'O':
                indexed_ner_tags.append(ner_tag)
            else:
                ner_tag = str(ner_tag).lower()
                if '.' in ner_tag:  # column or value
                    if ner_tag.endswith('.value'):  # value
                        indexed_ner_tags.append(f"V{schema_data[db_id]['column_map'][ner_tag[:-6]][0]}")
                    else:  # column
                        indexed_ner_tags.append(f"C{schema_data[db_id]['column_map'][ner_tag][0]}")
                else:  # table
                    indexed_ner_tags.append(f"T{schema_data[db_id]['table_map'][ner_tag][0]}")

        final_text_sequence = f"{toked_question} {' '.join(schema_concat_list)}"
        final_tag_sequence = f"{' '.join(indexed_ner_tags)} {' '.join(tag_schema_concat_list)}"

        ner_tag_set = ner_tag_set.union(set(final_tag_sequence.split()))

        output_data.append((final_text_sequence, final_tag_sequence))

    print(f"There are {len(ner_tag_set)} NER labels: {sorted(list(ner_tag_set))}")

    if output_file_ext == 'txt':
        with open(output_file, 'w') as outf:
            for data_entry in tqdm.tqdm(output_data):
                outf.write(f"{data_entry[0]}\n{data_entry[1]}\n")
    elif output_file_ext == 'json':
        with open(output_file, 'w') as outf:
            for id, data_entry in enumerate(tqdm.tqdm(output_data)):
                jsonl_inst = {
                    'id': id,
                    'ner_tags': str(data_entry[1]).split(),
                    'tokens': str(data_entry[0]).split()
                }
                outf.write(f"{json.dumps(jsonl_inst)}\n")

    return


def extract_system_ner_data_v2(input_file: str,
                               schema_json_file: str,
                               output_file: str,
                               data_type: str = 'train',
                               output_file_ext: str = 'txt') -> None:
    # load data
    spider_data = []
    with open(input_file) as f:
        data = json.load(f)
        assert isinstance(data, list)
        spider_data.extend(data)

    # load schema
    schema_data = {}
    ignored_dbs = set()
    with open(schema_json_file) as f:
        data = json.load(f)
        for entry in tqdm.tqdm(data):
            schema_data[entry['db_id']] = {'column_map': {}, 'table_map': {}}

            index = 0
            ind2tab = {}
            for table_name, ori_table_name in zip(entry["table_names"], entry["table_names_original"]):
                schema_data[entry['db_id']]['table_map'][f"@{ori_table_name.lower()}"] = (index, str(table_name))
                ind2tab[index] = ori_table_name.lower()
                index += 1

            index = 0
            tab2col = {}
            for col_name, ori_col_name in zip(entry['column_names'][1:], entry['column_names_original'][1:]):
                tab_index = col_name[0]
                schema_data[entry['db_id']]['column_map'][f"@{ind2tab[tab_index]}.{ori_col_name[1].lower()}"] = (
                index, tab_index, str(col_name[1]))
                if f"@{ind2tab[tab_index]}" in tab2col:
                    tab2col[f"@{ind2tab[tab_index]}"].append(f"{ori_col_name[1].lower()}")
                else:
                    tab2col[f"@{ind2tab[tab_index]}"] = [f"{ori_col_name[1].lower()}"]
                index += 1

            for k, v in tab2col.items():
                if len(v) > 20:
                    ignored_dbs.add(entry['db_id'])

    print(f"Ignored dbs: {ignored_dbs}")

    output_data = []
    ner_tag_set = set()
    for data_instance in tqdm.tqdm(spider_data):
        db_id = data_instance['db_id']
        if data_type == 'train' and db_id in ignored_dbs:
            continue

        query = data_instance['query']

        schema_custom_ner = data_instance['schema_custom_ner']
        ner_tags = schema_custom_ner['tags']
        toked_question = schema_custom_ner['toked_question']

        tag_schema_concat_list = []
        schema_concat_list = []
        for _, col_info in schema_data[db_id]['column_map'].items():
            schema_concat_list.append('[CLS]')
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"C{col_info[0]}")
            tag_schema_concat_list.append(f"O")
            schema_concat_list.append(f"{col_info[2]}")
            tag_schema_concat_list.extend([f"C{col_info[0]}"] * len(col_info[2].split()))

        for _, tab_info in schema_data[db_id]['table_map'].items():
            schema_concat_list.append('[CLS]')
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"T{tab_info[0]}")
            tag_schema_concat_list.append('O')
            schema_concat_list.append(f"{tab_info[1]}")
            tag_schema_concat_list.extend([f"T{tab_info[0]}"] * len(tab_info[1].split()))

        ner_tag_list = ner_tags.split()
        indexed_ner_tags = []
        for ner_tag in ner_tag_list:
            if ner_tag == 'O':
                indexed_ner_tags.append(ner_tag)
            else:
                ner_tag = str(ner_tag).lower()
                if '.' in ner_tag:  # column or value
                    if ner_tag.endswith('.value'):  # value
                        indexed_ner_tags.append(f"V{schema_data[db_id]['column_map'][ner_tag[:-6]][0]}")
                    else:  # column
                        indexed_ner_tags.append(f"C{schema_data[db_id]['column_map'][ner_tag][0]}")
                else:  # table
                    indexed_ner_tags.append(f"T{schema_data[db_id]['table_map'][ner_tag][0]}")

        final_text_sequence = f"{toked_question} {' '.join(schema_concat_list)}"
        final_tag_sequence = f"{' '.join(indexed_ner_tags)} {' '.join(tag_schema_concat_list)}"

        ner_tag_set = ner_tag_set.union(set(final_tag_sequence.split()))

        output_data.append((final_text_sequence, final_tag_sequence))

    print(f"There are {len(ner_tag_set)} NER labels: {sorted(list(ner_tag_set))}")

    if output_file_ext == 'txt':
        with open(output_file, 'w') as outf:
            for data_entry in tqdm.tqdm(output_data):
                outf.write(f"{data_entry[0]}\n{data_entry[1]}\n")
    elif output_file_ext == 'json':
        with open(output_file, 'w') as outf:
            for id, data_entry in enumerate(tqdm.tqdm(output_data)):
                jsonl_inst = {
                    'id': id,
                    'ner_tags': str(data_entry[1]).split(),
                    'tokens': str(data_entry[0]).split()
                }
                outf.write(f"{json.dumps(jsonl_inst)}\n")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract/split custom NER data')
    parser.add_argument("--input-file",
                        help="The input file", required=True)
    parser.add_argument("--output-folder",
                        help="The output folder", required=False)
    parser.add_argument("--output-file",
                        help="The output file", required=False)
    parser.add_argument("--output-file-ext",
                        help="The output file extension", default='txt', required=False)
    parser.add_argument("--data-type",
                        help="The data type", required=False)
    parser.add_argument("--split-k",
                        help="The split-k ratio", default=0.8, type=float, required=False)
    parser.add_argument("--schema-json-file",
                        help="The Spider schema file in JSON", required=False)
    parser.add_argument("--ner-type",
                        help="NER data type",
                        default="custom_ner_spider", type=str, required=False)
    args, _ = parser.parse_known_args()

    if args.ner_type == 'custom_ner_spider':
        extract_custom_ner_data(input_file=args.input_file,
                                output_folder=args.output_folder,
                                data_type=args.data_type,
                                split_k=args.split_k)
    elif args.ner_type == 'ner_hf':
        extract_system_ner_data_v1(input_file=args.input_file,
                                schema_json_file=args.schema_json_file,
                                output_file=args.output_file,
                                data_type=args.data_type,
                                output_file_ext=args.output_file_ext)

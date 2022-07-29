import json
import argparse
from typing import List, Optional, Union
from sklearn import model_selection


def split_dev_by_dbs(dev_json_file: str,
                     dev_json_output_file_prefix: str,
                     dbs: List[str],
                     split_rate: Optional[Union[float, int]] = 0.9):
    dev_data = json.load(open(dev_json_file))
    examples_by_db = {}
    for entry in dev_data:
        db_id = entry["db_id"]
        if db_id not in examples_by_db:
            examples_by_db[db_id] = [entry]
        else:
            examples_by_db[db_id].append(entry)

    examples_including_dbs = []
    for db in dbs:
        if db in examples_by_db:
            examples_including_dbs.extend(examples_by_db[db])
    examples_including_dbs_train, examples_including_dbs_test = model_selection.train_test_split(examples_including_dbs,
                                                                                                 random_state=42,
                                                                                                 train_size=split_rate)
    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}.json", "w") as outf_json:
        json.dump(examples_including_dbs, outf_json, indent=4, sort_keys=False)
    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}_train.json", "w") as outf_json:
        json.dump(examples_including_dbs_train, outf_json, indent=4, sort_keys=False)
    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}_test.json", "w") as outf_json:
        json.dump(examples_including_dbs_test, outf_json, indent=4, sort_keys=False)

    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}_gold.sql", "w") as outf_gold:
        for example in examples_including_dbs:
            outf_gold.write(f"{example['query']}\t{example['db_id']}\n")
    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}_train_gold.sql", "w") as outf_gold:
        for example in examples_including_dbs_train:
            outf_gold.write(f"{example['query']}\t{example['db_id']}\n")
    with open(f"{dev_json_output_file_prefix}_with_{'_'.join(dbs)}_test_gold.sql", "w") as outf_gold:
        for example in examples_including_dbs_test:
            outf_gold.write(f"{example['query']}\t{example['db_id']}\n")

    examples_excluding_dbs = []
    for db, examples in examples_by_db.items():
        if db not in dbs:
            examples_excluding_dbs.extend(examples)

    with open(f"{dev_json_output_file_prefix}_wo_{'_'.join(dbs)}_gold.sql", "w") as outf_gold:
        for example in examples_excluding_dbs:
            outf_gold.write(f"{example['query']}\t{example['db_id']}\n")
    with open(f"{dev_json_output_file_prefix}_wo_{'_'.join(dbs)}.json", "w") as outf_json:
        json.dump(examples_excluding_dbs, outf_json, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-json-file", required=True)
    parser.add_argument("--dev-json-output-file-prefix", required=True)
    parser.add_argument("--dbs", nargs='+', help='<Required> DBs to leave out', required=True)
    parser.add_argument("--split-rate", type=float, default=0.9, required=False)
    args = parser.parse_args()

    split_dev_by_dbs(dev_json_file=args.dev_json_file,
                     dev_json_output_file_prefix=args.dev_json_output_file_prefix,
                     dbs=args.dbs,
                     split_rate=int(args.split_rate) if int(args.split_rate) == args.split_rate else args.split_rate)

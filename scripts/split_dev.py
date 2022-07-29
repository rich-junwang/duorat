import json
import argparse
from sklearn import model_selection


def split_dev(dev_json_file: str,
              split_json_file_prefix: str,
              split_rate: float = 0.5):
    dev_data = json.load(open(dev_json_file))
    examples_by_db = {}
    for entry in dev_data:
        db_id = entry["db_id"]
        if db_id not in examples_by_db:
            examples_by_db[db_id] = [entry]
        else:
            examples_by_db[db_id].append(entry)

    half1_examples_by_db = {}
    half2_examples_by_db = {}
    for db_id, examples in examples_by_db.items():
        if int(split_rate * len(examples)) < 1:
            half1_examples_by_db[db_id], half2_examples_by_db[db_id] = model_selection.train_test_split(examples,
                                                                                                        random_state=42,
                                                                                                        train_size=1)
        else:
            half1_examples_by_db[db_id], half2_examples_by_db[db_id] = model_selection.train_test_split(examples,
                                                                                                        random_state=42,
                                                                                                        train_size=split_rate)

    half1_examples = []
    for _, examples in half1_examples_by_db.items():
        half1_examples.extend(examples)
    # split into different rates
    half1_examples_by_db_with_rates = {0.8: {}, 0.6: {}, 0.4: {}, 0.2: {}, 0.1: {}}
    half1_example_with_rates = {}
    for rate, example_dict in half1_examples_by_db_with_rates.items():
        for db_id, examples in half1_examples_by_db.items():
            if int(rate * len(examples)) < 1.0:
                example_dict[db_id], _ = model_selection.train_test_split(examples,
                                                                          random_state=42,
                                                                          train_size=1)
            else:
                example_dict[db_id], _ = model_selection.train_test_split(examples,
                                                                          random_state=42,
                                                                          train_size=rate)

        for _, examples in example_dict.items():
            if rate in half1_example_with_rates:
                half1_example_with_rates[rate].extend(examples)
            else:
                half1_example_with_rates[rate] = examples

        with open(f"{split_json_file_prefix}_half1_split{str(rate * 0.5).replace('.', '')}.json", "w") as fout1:
            json.dump(half1_example_with_rates[rate], fout1, sort_keys=True, indent=4)

    # this will be used for evaluation (unchanged)
    half2_examples = []
    for _, examples in half2_examples_by_db.items():
        half2_examples.extend(examples)

    with open(f"{split_json_file_prefix}_half1.json", "w") as fout1, open(f"{split_json_file_prefix}_half2.json",
                                                                          "w") as fout2:
        json.dump(half1_examples, fout1, sort_keys=True, indent=4)
        json.dump(half2_examples, fout2, sort_keys=True, indent=4)

    with open(f"{split_json_file_prefix}_gold.txt", "w") as outf_gold:
        for example in half2_examples:
            outf_gold.write(f"{example['query']}\t{example['db_id']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-json-file", required=True)
    parser.add_argument("--split-json-file-prefix", required=True)
    parser.add_argument("--split-rate", type=float, default=0.5, required=False)
    args = parser.parse_args()

    split_dev(dev_json_file=args.dev_json_file,
              split_json_file_prefix=args.split_json_file_prefix,
              split_rate=args.split_rate)

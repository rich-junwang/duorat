import argparse
import glob
import json
from sklearn import model_selection


def collect_synthetic_data_tsv_files(tsv_files_folder_path: str,
                                     output_data_file: str,
                                     samples_by_db: int = 100):
    output_data = []
    for file_name in glob.iglob(f'{tsv_files_folder_path}/**/*.tsv', recursive=True):
        print(f"Processing file path: {file_name}...")
        with open(file_name) as f:
            examples = []
            for line in f:
                line = line.strip()
                _, question, query = line.split('\t')
                db_id = file_name.split('/')[-2]
                example_dict = {
                    'db_id': db_id,
                    'query': query,
                    'question': question,
                    'query_toks': '',
                    'query_toks_no_value': '',
                    'question_toks': '',
                    'sql': ''
                }
                examples.append(example_dict)

            if samples_by_db != -1 and samples_by_db < len(examples):
                sampled_examples, _ = model_selection.train_test_split(examples,
                                                                       random_state=42,
                                                                       train_size=samples_by_db)
            else:  # otherwise, get all.
                sampled_examples = examples
            output_data.extend(sampled_examples)

    with open(output_data_file, 'w') as outf:
        json.dump(output_data, outf, indent=4, sort_keys=True)


def collect_synthetic_data_json_files(json_files_folder_path: str,
                                      output_data_file: str,
                                      samples_by_db: int = 100):
    output_data = []
    for json_file in glob.iglob(f'{json_files_folder_path}/**/*.json', recursive=True):
        print(f"Processing file path: {json_file}...")
        with open(json_file) as f:
            examples = []
            try:
                json_data = json.load(f)
                for entry_data in json_data:
                    question = entry_data['nl_lexicalized']
                    query = entry_data['sql_lexicalized']
                    db_id = entry_data['db'].split('/')[-1].replace('.sqlite', '')
                    example_dict = {
                        'db_id': db_id,
                        'query': query,
                        'question': question,
                        'query_toks': '',
                        'query_toks_no_value': '',
                        'question_toks': '',
                        'sql': ''
                    }

                    examples.append(example_dict)
            except ValueError:
                # print(f'{json_file} is not proper JSON file. Reverting to JSONL...')
                # json_file = f"{json_file}l"
                # with open(json_file) as f1:
                #     for line in f1:
                #         line = line.strip()
                #         json_data = json.loads(line)
                #         question = json_data['nl_lexicalized']
                #         query = json_data['sql_lexicalized']
                #         db_id = json_data['db'].split('/')[-1].replace('.sqlite', '')
                #         example_dict = {
                #             'db_id': db_id,
                #             'query': query,
                #             'question': question,
                #             'query_toks': '',
                #             'query_toks_no_value': '',
                #             'question_toks': '',
                #             'sql': ''
                #         }
                #         examples.append(example_dict)
                print("Something wrong!")

            if samples_by_db != -1 and samples_by_db < len(examples):
                sampled_examples, _ = model_selection.train_test_split(examples,
                                                                       random_state=42,
                                                                       train_size=samples_by_db)
            else:  # otherwise, get all.
                sampled_examples = examples

            output_data.extend(sampled_examples)

    with open(output_data_file, 'w') as outf:
        json.dump(output_data, outf, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-folder-path", required=True)
    parser.add_argument("--file-type", required=False, default="json")
    parser.add_argument("--output-data-file", required=True)
    parser.add_argument("--samples-by-db", type=int, default=100, required=False)
    args = parser.parse_args()

    if args.file_type == 'tsv':
        collect_synthetic_data_tsv_files(tsv_files_folder_path=args.files_folder_path,
                                         output_data_file=args.output_data_file,
                                         samples_by_db=args.samples_by_db)
    elif args.file_type == 'json':
        collect_synthetic_data_json_files(json_files_folder_path=args.files_folder_path,
                                          output_data_file=args.output_data_file,
                                          samples_by_db=args.samples_by_db)
    else:
        raise ValueError("{args.file_type} is unsupported.")

import sys
import json
import os

spider_data_file = sys.argv[1]
output_folder = sys.argv[2]
split_k = sys.argv[3]
data_type = sys.argv[4]  # spider_train, spider_dev
spider_data_output_file = sys.argv[5]

spider_data = None
with open(spider_data_file) as f:
    spider_data = json.load(f)

    data_by_db = {}
    for entry in spider_data:
        db_id = entry["db_id"]

        if db_id not in data_by_db:
            data_by_db[db_id] = [entry]
        else:
            data_by_db[db_id].append(entry)

slml_outputs = []
for db_id, entry in data_by_db.items():
    cner_preds_file = os.path.join(output_folder,
                                   f"{db_id}_split{split_k.replace('.', '')}",
                                   f"{db_id}_db_{data_type}_set_test_split{split_k}_preds.txt")
    with open(cner_preds_file) as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if line == '':
                # form new SLML question
                new_slml_tokens = []
                qindex = 0
                while qindex < len(tokens):
                    qtoken = tokens[qindex]
                    stag = tags[qindex]

                    if stag.startswith('@') and len(stag) > 1:
                        cur_qindex = qindex
                        while qindex + 1 < len(tokens) and tags[qindex + 1] == stag:
                            qindex += 1

                        merged_tokens = tokens[cur_qindex:qindex + 1]
                        if '.value' in stag:  # value matching
                            stokens = stag.split('.')
                            table_mention = stokens[0].replace('@', '')
                            column_mention = stokens[1]
                            value = ' '.join(merged_tokens)
                            slml_text = f"<vm table=\"{table_mention}\" column=\"{column_mention}\" value=\"{value}\" confidence=\"high\">{value}</vm>"
                            new_slml_tokens.append(slml_text)
                        elif '.' in stag:
                            stokens = stag.split('.')
                            table_mention = stokens[0].replace('@', '')
                            column_mention = stokens[1]
                            slml_text = f"<cm table=\"{table_mention}\" column=\"{column_mention}\" confidence=\"high\">{' '.join(merged_tokens)}</cm>"
                            new_slml_tokens.append(slml_text)
                        else:
                            slml_text = f"<tm table=\"{stag.replace('@', '')}\" confidence=\"high\">{' '.join(merged_tokens)}</tm>"
                            new_slml_tokens.append(slml_text)
                    else:
                        new_slml_tokens.append(qtoken)

                    qindex += 1

                slml_outputs.append(' '.join(new_slml_tokens))
                tokens = []
                tags = []
                continue

            splits = line.split()
            assert len(splits) == 3
            tokens.append(splits[0])
            tags.append(splits[2])  # word gold_tag pred_tag

assert len(slml_outputs) == len(spider_data)

new_spider_data = []
for slml_output, entry in zip(slml_outputs, spider_data):
    entry["slml_question"] = slml_output
    new_spider_data.append(entry)

with open(spider_data_output_file, "w") as outf:
    json.dump(new_spider_data, outf, sort_keys=False, indent=4)

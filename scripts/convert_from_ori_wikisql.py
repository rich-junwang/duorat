# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json


def remove_semicolon_in_sql_query(query):
    query = str(query)
    if query.strip().endswith(";"):
        query = query[:-1].strip()
    return query


def remove_unicode_chars(text):
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--db-schema', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(args)

    # read input
    data = []  # expected: array of dict
    with open(args.input) as finput:
        for line in finput:
            line = line.strip()
            data.append(json.loads(line))

    # read db schema
    schema_dict = {}
    with open(args.db_schema) as fschema:
        schema = json.load(fschema)
        for entry in schema:
            schema_dict[entry["db_id"].replace("table_", "").replace("_", "-")] = [e[1] for e in entry["column_names_original"] if e[0] != -1]

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<']

    items = []
    for entry in data:
        db_id = entry["table_id"]

        question = remove_unicode_chars(entry["question"])
        question = question.replace("\"", "")
        question = question.replace('\'', '᾿')

        ori_sql_dict = entry["sql"]
        print(db_id, question, ori_sql_dict)

        # SELECT
        spider_sql = []
        spider_sql.append('SELECT')
        sel_col_name = f"{schema_dict[db_id][int(ori_sql_dict['sel'])]}"
        agg = agg_ops[int(ori_sql_dict['agg'])]
        if agg is not '':
            spider_sql.append(f"{agg}({sel_col_name})")
        else:
            spider_sql.append(f"{sel_col_name}")

        # FROM
        spider_sql.append("FROM")
        spider_sql.append(f"table_{db_id.replace('-', '_')}")

        # WHERE
        spider_sql.append("WHERE")
        cond_list = []
        for cond in ori_sql_dict["conds"]:
            col_ind, cond_ind, val = cond

            col_name = f"{schema_dict[db_id][int(col_ind)]}"
            cond_txt = cond_ops[int(cond_ind)]
            try:
                val = int(val)
            except:
                try:
                    val = float(val)
                except:
                    val = str(val)
                    val = val.replace("\"", '')  # remove quote in val since it causes some issues with Spider SQL parser
                    val = val.replace('\'', '᾿')  # dirty hack. Use a special single quote instead

            cur_cond_list = []
            cur_cond_list.append(col_name)
            cur_cond_list.append(cond_txt)
            if isinstance(val, float) or isinstance(val, int):
                cur_cond_list.append(f"{val}")
            elif isinstance(val, str):
                if val.startswith("\"") and val.endswith("\""):
                    cur_cond_list.append(f"{val}")
                else:
                    cur_cond_list.append(f"\"{val}\"")
            cond_list.append(' '.join(cur_cond_list))
        spider_sql.append(' AND '.join(cond_list))
        final_sql = ' '.join(spider_sql)

        items.append({
            'db_id': f"table_{db_id.replace('-', '_')}",
            'query': remove_semicolon_in_sql_query(final_sql),
            'sql': "",
            'question': question
        })
        print(items[-1])

    with open(args.output, 'w') as foutput:
        json.dump(items, foutput, indent=2, separators=(",", ": "))

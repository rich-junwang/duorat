# Vu Hoang (vu.hoang@oracle.com)
# Oracle Corp.
# Create a silver training data for training custom NER model
""" Motivation
1) List the name, born state and age of the heads of departments ordered by age.

List the <cm table=\"head\" column=\"name\" confidence=\"high\">name</cm
> , <cm table=\"head\" column=\"born_state\" confidence=\"high\">born state</cm> and <cm table=\"head\" column
=\"age\" confidence=\"high\">age</cm> of the <tm table=\"head\" confidence=\"low\">heads</tm> of departments ordered by <cm table=\"head\" column=\"age\" confidence=\"high\">age</cm> .

List the name , born state and age of the heads of departments ordered by age.
O O @head.name O @head.born_state @head.born_state O @head.age O O @head O departments ordered by @head.age O


SELECT name ,  born_state ,  age FROM head ORDER BY age
--> SELECT head.name ,  head.born_state ,  head.age FROM head ORDER BY head.age

----
2) What is the average number of employees of the departments whose rank is between 10 and 15?

What is the average number of <cm table=\"department\" column=\"Num_Employees\" confidence=\"low\">employees</cm> of the <tm table=\"department\" confidence=\"low\">departments</tm> whose <cm table=\"department\" column=\"Ranking\" confidence=\"low\">rank</cm> is between <vm table=\"department\" column=\"Ranking\" value=\"10\" confidence=\"high\">10</vm> and <vm table=\"department\" column=\"Ranking\" value=\"15\" confidence=\"high\">15</vm> ?

What is the average number of employees of the departments whose rank is between 10 and 15?
O O O O O O @department.Num_Employees O O @department O @department.Ranking O O @department.Ranking.value O @department.Ranking.value O

SELECT avg(num_employees) FROM department WHERE ranking BETWEEN 10 AND 15
--> SELECT avg(department.num_employees) FROM department WHERE department.ranking BETWEEN 10 AND 15

---
3) What are the names of the heads who are born outside the California state?

What are the <cm table=\"head\" column=\"name\" confidence=\"low\">names<
/cm> of the <tm table=\"head\" confidence=\"low\">heads</tm> who are <cm table=\"head\" column=\"born_state\" confidence=\"low\">born</cm> outside the <vm
 table=\"head\" column=\"born_state\" value=\"California\" confidence=\"high\">California</vm> <cm table=\"head\" column=\"born_state\" confidence=\"low\">state</cm> ?

 What are the names of the heads who are born outside the California state?
 O O O @head.name O O @head O O @head.born_state O O @head.born_state.value @head.born_state O

SELECT name FROM head WHERE born_state != 'California'
--> SELECT head.name FROM head WHERE head.born_state != 'California'

---
4) What are names of stations that have average bike availability above 10 and are not located in San Jose city?

What are <cm table=\"station\" column=\"name\" confidence=\"low\">names</cm> of <tm table=\"station\" confidence=\"low\">stations</tm> that have average <cm table=\"status\" column=\"bikes_available\" confidence=\"low\">bike</cm> availability above <vm table=\"status\" column=\"bikes_available\" value=\"10\" confidence=\"high\">10</vm> and are not located in <vm table=\"station\" column=\"city\" value=\"San Jose\" confidence=\"high\">San Jose</vm> <cm table=\"station\" column=\"city\" confidence=\"high\">city</cm> ?

What are the names of stations that have average bike availability above 10 and are not located in San Jose city ?
O O @station.name O @station O O O @status.bikes_available @status.bikes_available O @status.bikes_available.value O O O O O @station.city.value @station.city.value @station.city O

SELECT T1.name FROM station AS T1 JOIN status AS T2 ON T1.id  =  T2.station_id GROUP BY T2.station_id HAVING avg(bikes_available)  >  10 EXCEPT SELECT name FROM station WHERE city  =  \"San Jose\"
--> SELECT station.name FROM station JOIN status ON station.id  =  status.station_id GROUP BY status.station_id HAVING avg(status.bikes_available)  >  10 EXCEPT SELECT station.name FROM station WHERE station.city  =  \"San Jose\"

"select": [
    false,                    # isDistinct
    [                         # val_units (a list)
        [                     # tuple
            0,                # agg_id
            [                 # val_unit (tuple)
                0,            # unit_op
                [             # col_unit1 (tuple)
                    0,        # agg_id
                    2,        # col_id
                    false     # isDistinct
                ],
                null          # col_unit2 (tuple)
            ]
        ]
    ]
]

"from": {
    "conds": [
        [
            false,                # is 'not' op
            2,                    # = (op id)
            [                     # val_unit
                0,                # none (unit_op)
                [                 # col_unit1
                    0,            # none (agg_id)
                    1,            # col_id
                    false         # isDistinct
                ],
                null              # col_unit2
            ],
            [                     # val1
                0,                # none (agg_id)
                8,                # col_id
                false             # isDistinct
            ],
            null                  # val2
        ]
    ],
    "table_units": [              # a list
        [                         # tuple
            "table_unit",         # unused
            0                     # table_id
        ],
        [                         # tuple
            "table_unit",         # unused
            1                     # table_id
        ]
    ]
}

"where": [                 # a list
    [                      # tuple
        false,             # not_op
        2,                 # op_id
        [                  # val_unit (tuple)
            0,             # unit_op
            [              # col_unit1 (tuple)
                0,         # agg_id
                6,         # col_id
                false      # isDistinct
            ],
            null           # col_unit2 (tuple)
        ],
        "\"San Jose\"",    # val1
        null               # val2
    ]
]

"having": [                # a list
    [                      # tuple
        false,             # not_op
        3,                 # op_id (e.g., >)
        [                  # val_unit (tuple)
            0,             # unit_op
            [              # col_unit1 (tuple)
                5,         # agg_id (e.g., avg)
                9,         # col_id
                false      # isDistinct
            ],
            null           # col_unit1 (tuple)
        ],
        10.0,              # val1
        null               # val2
    ]
]
"""

import argparse
import json
import os

import _jsonnet
import tqdm

from typing import List, Dict, Tuple, Set, Any

from duorat.preproc import offline  # *** Compulsory for registering duorat.preproc classes
from duorat.preproc.abstract_preproc import AbstractPreproc
from duorat.utils import registry
from duorat.datasets.spider import load_original_schemas
from scripts.data_aug.extract_templates import get_processed_db_cached
from third_party.spider.process_sql import get_sql
from third_party.spider.preprocess.schema import Schema
from duorat.preproc.slml import SLMLParser
from duorat.types import (
    MatchTag,
    TableMatchTag,
    ColumnMatchTag,
    ValueMatchTag,
    LowConfidenceMatch,
    SQLSchema
)

IUE_OPS = ['intersect', 'union', 'except']
ALWAYS_O_TAG_LIST = set(['a', 'an', 'the', 'as', 'in', 'is', 'of', 'or', 'at', 'average', 'maximum', 'minimum'])


def create_silver_training_data(duorat_preprocessor: AbstractPreproc,
                                input_files: List[str],
                                output_file: str,
                                db_folder_path: str,
                                schema_json_path: str):
    # read data
    data = None
    for input_file in input_files:
        with open(input_file) as f:
            if data is None:
                data = json.load(f)
            else:
                data.extend(json.load(f))

    # read JSON schema
    all_spider_schemas: List[Schema] = load_original_schemas([schema_json_path])

    schema_cache = {}
    dstats: Dict[str, Tuple[int, Set]] = {}
    for ind, item in enumerate(tqdm.tqdm(data)):
        db_id = item["db_id"]
        db_path = os.path.join(db_folder_path, db_id, f"{db_id}.sqlite")
        question = item["question"]
        # print("---------------------------")
        # print(question)

        # Get unsupervised SLML output
        if db_id in schema_cache:
            sql_schema = schema_cache[db_id]
        else:
            sql_schema = get_processed_db_cached(db_path=db_path, duorat_preprocessor=duorat_preprocessor)
            schema_cache[db_id] = sql_schema

        slml_question: str = duorat_preprocessor.schema_linker.question_to_slml(
            question=question, sql_schema=sql_schema,
        )
        item['unsup_slml_question'] = slml_question

        def _get_idMap_rev(idMap: Dict[str, id]) -> Tuple[Dict[id, str], Dict[id, str]]:
            table_rmap = {}
            column_rmap = {}
            for k, v in idMap.items():
                if '.' in k or '*' in k:
                    column_rmap[v] = k
                else:
                    table_rmap[v] = k
            return table_rmap, column_rmap

        # Get SQL
        # print(item["query"])
        spider_schema: Schema = all_spider_schemas[db_id]
        table_rmap, column_rmap = _get_idMap_rev(idMap=spider_schema.idMap)
        # spider_sql: dict = get_sql(schema=spider_schema, query=item["query"])  # a dictionary
        # or just use existing item["sql"]
        spider_sql = item["sql"]

        # print(spider_sql)

        def _extract_allowed_schema_entities(spider_sql: Dict, allowed_schema_entities: Dict):
            # *** Get allowed schema entities (heuristics)

            # for select clause
            select_clause = spider_sql['select']
            # print(select_clause)
            for cond in select_clause[1]:  # cond is a tuple/list
                if not isinstance(cond, tuple) and not isinstance(cond, list):
                    continue

                # col_unit1
                col_id = cond[1][1][1]
                allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[col_id])
                # col_unit2
                if cond[1][2] is not None:
                    allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[cond[1][2][1]])

            def _process_cond_val(cond_val: Any, cid: str) -> None:
                if cond_val is not None and any([isinstance(cond_val, float),
                                                 isinstance(cond_val, int),
                                                 isinstance(cond_val, str)]):
                    # potential value match
                    if not isinstance(cond_val, str) and str(cond_val).endswith('.0'):
                        cond_val = int(cond_val)  # convert to integer if required
                    if isinstance(cond_val, str):
                        cond_val = str(cond_val).replace('\"', '')
                    else:
                        cond_val = str(cond_val)
                    for tok in duorat_preprocessor.tokenizer.tokenize(s=cond_val):
                        allowed_schema_entities['value_entity_mapping'][tok] = column_rmap[cid]

            # for from clause
            from_clause = spider_sql['from']
            # print(from_clause)
            for table_unit in from_clause['table_units']:
                table_id_or_nested_sql = table_unit[1]
                if isinstance(table_id_or_nested_sql, int):
                    allowed_schema_entities['table_or_column_entity_set'].add(table_rmap[table_id_or_nested_sql])
                elif isinstance(table_id_or_nested_sql, dict):
                    # process nested SQL here, e.g., SELECT COUNT(*) FROM (SELECT T1.Name FROM country AS T1 JOIN
                    # countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE T2.Language  =  "English" INTERSECT
                    # SELECT T1.Name FROM country AS T1 JOIN countrylanguage AS T2 ON T1.Code  =  T2.CountryCode WHERE
                    # T2.Language  =  "Dutch")
                    _extract_allowed_schema_entities(spider_sql=table_id_or_nested_sql,
                                                     allowed_schema_entities=allowed_schema_entities)
            if from_clause['conds']:
                for cond in from_clause['conds']:
                    if not isinstance(cond, tuple) and not isinstance(cond, list):
                        continue

                    _process_cond_val(cond_val=cond[3], cid=cond[2][1][1])
                    _process_cond_val(cond_val=cond[4], cid=cond[2][1][1])

            # for where clause
            where_clause = spider_sql['where']
            # print(where_clause)
            for cond in where_clause:  # cond is a tuple
                if not isinstance(cond, tuple) and not isinstance(cond, list):
                    continue

                # col_unit1
                col_id = cond[2][1][1]
                allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[col_id])
                # col_unit2
                if cond[2][2] is not None:
                    allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[cond[2][2][1]])

                if cond[3] is not None:
                    if isinstance(cond[3], list):
                        allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[cond[3][1]])
                    elif isinstance(cond[3], dict):
                        # process nested SQL here, e.g., 'SELECT song_name FROM singer WHERE age  >  (SELECT avg(age)
                        # FROM singer)'
                        _extract_allowed_schema_entities(spider_sql=cond[3],
                                                         allowed_schema_entities=allowed_schema_entities)

                _process_cond_val(cond_val=cond[3], cid=col_id)
                _process_cond_val(cond_val=cond[4], cid=col_id)

            # for having clause
            having_clause = spider_sql['having']
            # print(having_clause)
            for cond in having_clause:  # cond is a tuple
                if not isinstance(cond, tuple) and not isinstance(cond, list):
                    continue

                # col_unit1
                col_id = cond[2][1][1]
                allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[col_id])
                # col_unit2
                if cond[2][2] is not None:
                    allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[cond[2][2][1]])

            # for order by clause
            order_by_clause = spider_sql['orderBy']
            if order_by_clause:
                for val_unit in order_by_clause[1]:  # val_unit = tuple(unit_op, col_unit1, col_unit2)
                    # col_unit1 = tuple(agg_id, col_id, isDistinct)
                    if isinstance(val_unit, tuple) or isinstance(val_unit, list):
                        if (isinstance(val_unit[1], tuple) or isinstance(val_unit[1], list)) \
                                and val_unit[1] is not None:
                            allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[val_unit[1][1]])
                        if (isinstance(val_unit[2], tuple) or isinstance(val_unit[2], list)) \
                                and val_unit[2] is not None:
                            allowed_schema_entities['table_or_column_entity_set'].add(column_rmap[val_unit[2][1]])

            # for intersect/union/except
            for op in IUE_OPS:
                op_clause = spider_sql[op]
                if op_clause is not None:
                    _extract_allowed_schema_entities(spider_sql=op_clause,
                                                     allowed_schema_entities=allowed_schema_entities)

            return

        def _get_sql_schema_mapping(sql_schema: SQLSchema) -> Tuple[Dict[str, str], Dict[str, str]]:
            tabMap = {}
            colMap = {}
            for tab_id, tab_name in sql_schema.original_table_names.items():
                tabMap[tab_id] = tab_name
            for col_id, col_name in sql_schema.original_column_names.items():
                colMap[col_id] = col_name
            return tabMap, colMap

        allowed_schema_entities = {"table_or_column_entity_set": set(),
                                   "value_entity_mapping": {}}
        _extract_allowed_schema_entities(spider_sql=spider_sql,
                                         allowed_schema_entities=allowed_schema_entities)
        table_map, col_map = _get_sql_schema_mapping(sql_schema=sql_schema)
        slml_parser = SLMLParser(sql_schema=sql_schema,
                                 tokenizer=duorat_preprocessor.tokenizer,
                                 do_filter_bad_matches=duorat_preprocessor.schema_linker.do_filter_bad_matches)
        slml_parser.feed(data=slml_question)

        # Here, we apply the following heuristics:
        # - only columns after SELECT/FROM/WHERE in SQL are considered to have matches in NL question.
        # - filter bad matches
        # - final match should be with high confidence. If there are multiple matches, select one randomly.
        #   Make sure no nested matches.
        question_tokens = []
        schema_tags = []
        for question_token in slml_parser.question_tokens:
            def _create_schema_tag(match_tag: MatchTag) -> str:
                if isinstance(match_tag, ValueMatchTag):
                    return f"@{table_map[match_tag.table_id]}.{col_map[match_tag.column_id]}.value"
                elif isinstance(match_tag, ColumnMatchTag):
                    return f"@{table_map[match_tag.table_id]}.{col_map[match_tag.column_id]}"
                elif isinstance(match_tag, TableMatchTag):
                    return f"@{table_map[match_tag.table_id]}"

                return None

            raw_value = question_token.raw_value
            # print(raw_value)
            question_tokens.append(raw_value)
            if raw_value in ALWAYS_O_TAG_LIST:
                schema_tags.append('O')
                continue

            match_tags = question_token.match_tags
            # print(match_tags)
            if len(match_tags) > 0:
                new_match_tags = []
                for match_tag in match_tags:
                    if isinstance(match_tag, ValueMatchTag) or isinstance(match_tag, ColumnMatchTag):
                        tab_col_ref = f"{table_map[match_tag.table_id]}.{col_map[match_tag.column_id]}".lower()
                        if tab_col_ref in allowed_schema_entities['table_or_column_entity_set']:
                            new_match_tags.append(match_tag)
                    elif isinstance(match_tag, TableMatchTag):
                        tab_ref = f"{table_map[match_tag.table_id]}".lower()
                        if tab_ref in allowed_schema_entities['table_or_column_entity_set']:
                            new_match_tags.append(match_tag)

                if len(new_match_tags) > 0:
                    # Here, we don't support nested schema entities since training custom nested NER model is very
                    # tricky. It may be a main weakness of this idea.
                    new_match_tags_sorted = sorted(new_match_tags, key=lambda t: t.confidence, reverse=True)
                    # If there is no high confidence, pick first one with low confidence randomly.
                    # If there are mixed of high and low confidence, pick high confidence only.
                    # If there are multiple high confidence, pick one randomly.
                    if isinstance(new_match_tags_sorted[0].confidence, LowConfidenceMatch) \
                            and raw_value in allowed_schema_entities['value_entity_mapping']:
                        schema_tags.append(f"@{allowed_schema_entities['value_entity_mapping'][raw_value]}.value")
                    else:
                        schema_tags.append(_create_schema_tag(match_tag=new_match_tags_sorted[0]))
                else:
                    if raw_value in allowed_schema_entities['value_entity_mapping']:
                        schema_tags.append(f"@{allowed_schema_entities['value_entity_mapping'][raw_value]}.value")
                    else:
                        schema_tags.append('O')
            else:
                if raw_value in allowed_schema_entities['value_entity_mapping']:
                    schema_tags.append(f"@{allowed_schema_entities['value_entity_mapping'][raw_value]}.value")
                else:
                    schema_tags.append('O')

        def _detokenize(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str]]:
            assert len(tokens) == len(tags)

            new_tokens = []
            new_tags = []
            index = 0
            while index < len(tokens):
                token = tokens[index]
                tag = tags[index]
                if '##' in token:
                    if len(new_tokens) > 0:
                        new_tokens[-1] = f"{new_tokens[-1]}{token.replace('##', '')}"
                        # Here, tag of first subword token is reserved.
                    else:
                        print(f"WARNING: invalid subword tokenization.")
                elif '.' == token and index - 1 >= 0 and index + 1 < len(tokens):
                    prev_token = tokens[index - 1]
                    next_token = tokens[index + 1].replace('##', '')
                    prev_tag = tags[index - 1]
                    next_tag = tags[index + 1]
                    if (prev_tag == tag == next_tag) and ((prev_token.isnumeric() and next_token.isnumeric()) \
                                                          or (prev_token.isalpha() and next_token.isalpha())):
                        # 37 . 5 or example . org
                        new_tokens[-1] = f"{new_tokens[-1]}{token}{next_token}"
                        # Here, tag of first subword token is reserved.

                        index += 1
                    elif prev_tag == tag and prev_token.isalpha():  # Dr .
                        new_tokens[-1] = f"{new_tokens[-1]}{token}"
                        # Here, tag of first subword token is reserved.
                    else:
                        new_tokens.append(token)
                        new_tags.append(tag)
                else:
                    new_tokens.append(token)
                    new_tags.append(tag)

                index += 1

            # post-process for email addresses, e.g., stanley @ example.com --> stanley@example.com
            tokens = new_tokens
            tags = new_tags
            new_tokens = []
            new_tags = []
            index = 0
            while index < len(tokens):
                token = tokens[index]
                tag = tags[index]
                if '@' == token and index - 1 >= 0 and index + 1 < len(tokens):
                    next_token = tokens[index + 1]
                    prev_tag = tags[index - 1]
                    next_tag = tags[index + 1]
                    if prev_tag == tag == next_tag:
                        new_tokens[-1] = f"{new_tokens[-1]}{token}{next_token}"
                        # Here, tag of first subword token is reserved.
                        index += 2
                        continue

                new_tokens.append(token)
                new_tags.append(tag)

                index += 1

            return new_tokens, new_tags

        def _get_ne_stats(tags: List[str], db_id: str, dstats: Dict[str, Tuple[int, Set]]) -> None:
            if db_id not in dstats:
                dstats[db_id] = {'#examples': 0, "tags": {}}

            if db_id in dstats:
                dstats[db_id]['#examples'] += 1

            for ind, tag in enumerate(tags):
                if ind > 0 and tags[ind - 1] == tag:
                    continue

                if tag not in dstats[db_id]['tags']:
                    dstats[db_id]['tags'][tag] = 1
                else:
                    dstats[db_id]['tags'][tag] += 1

            return

        question_tokens, schema_tags = _detokenize(tokens=question_tokens, tags=schema_tags)
        _get_ne_stats(tags=schema_tags, db_id=db_id, dstats=dstats)
        item["schema_custom_ner"] = {"toked_question": ' '.join(question_tokens),
                                     "tags": ' '.join(schema_tags)}

        # form new SLML question
        new_slml_tokens = []
        qindex = 0
        while qindex < len(question_tokens):
            qtoken = question_tokens[qindex]
            stag = schema_tags[qindex]

            if stag.startswith('@') and len(stag) > 1:
                cur_qindex = qindex
                while qindex + 1 < len(question_tokens) and schema_tags[qindex + 1] == stag:
                    qindex += 1

                merged_tokens = question_tokens[cur_qindex:qindex + 1]
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

        item["slml_question"] = ' '.join(new_slml_tokens)

    # write output data
    json.dump(data, open(output_file, "w"), indent=4, sort_keys=True)
    json.dump(dstats, open(f"{output_file.replace('.json', '_stats.json')}", "w"), indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create silver training data for custom NER model')
    parser.add_argument("--duorat-config-file",
                        help="The DuoRAT config", required=True)
    parser.add_argument("--input-files", nargs='+',
                        help="The input file", required=True)
    parser.add_argument("--output-file",
                        help="The JSON output file", required=True)
    parser.add_argument("--db-folder-path",
                        help="The folder path to DB folder", default='./data/database')
    parser.add_argument("--schema-json-path",
                        help="The file path to schema JSON", default='./data/spider/tables.json')
    parser.add_argument(
        "--with-stemming",
        default=False,
        action="store_true",
        help="If True, do stemming with schema linker",
    )
    args, _ = parser.parse_known_args()

    # Initialize
    duorat_config_file = args.duorat_config_file

    # DuoRAT config
    print("Initializing DuoRAT config...")
    config = json.loads(_jsonnet.evaluate_file(duorat_config_file))
    config['model']['preproc']['schema_linker']['with_stemming'] = args.with_stemming

    # DuoRAT preprocessor
    print("Initializing DuoRAT preprocessor...")
    duorat_preprocessor: AbstractPreproc = registry.construct("preproc", config["model"]["preproc"])

    # Create silver training data for custom NER model
    print(f"Processing input files from {str(args.input_files)}...")
    create_silver_training_data(duorat_preprocessor=duorat_preprocessor,
                                input_files=args.input_files,
                                output_file=args.output_file,
                                db_folder_path=args.db_folder_path,
                                schema_json_path=args.schema_json_path)

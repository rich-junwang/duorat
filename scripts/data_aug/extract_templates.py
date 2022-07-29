# Vu Hoang (vu.hoang@oracle.com)
# Oracle Corp.
# Implementation of NL2SQL Template Extraction Algorithm

import os
import json
from typing import Dict, Set, List
import csv
from bidict import bidict
import re

import argparse

import _jsonnet

import tqdm

from duorat.types import SQLSchema
from duorat.utils import registry
from duorat.preproc.slml import SLMLParser
from duorat.preproc import offline  # *** Compulsory for registering duorat.preproc classes
from duorat.preproc.utils import preprocess_schema_uncached, refine_schema_names
from duorat.datasets.spider import (
    schema_dict_to_spider_schema,
)
from duorat.preproc.abstract_preproc import AbstractPreproc
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from duorat.types import ColumnMatchTag, TableMatchTag, ValueMatchTag, HighConfidenceMatch
from functools import lru_cache


@lru_cache(maxsize=None)
def get_processed_db_cached(db_path: str, duorat_preprocessor: AbstractPreproc) -> SQLSchema:
    sql_schema: SQLSchema = preprocess_schema_uncached(
        schema=schema_dict_to_spider_schema(refine_schema_names(dump_db_json_schema(db_path, ""))),
        db_path=db_path,
        tokenize=duorat_preprocessor._schema_tokenize,
    )

    return sql_schema


def postprocess_question(question: str) -> str:
    question = question.strip()
    if question.startswith("\"") and question.endswith("\""):
        question = question[1:-1]

    question = question.replace("\"\"", "\"")
    question = question.replace("  ", " ")

    return question.strip()


def postprocess_sql(sql: str) -> str:
    # " AirCon " -> "AirCon"
    sql = sql.strip()
    if sql.startswith("\"") and sql.endswith("\""):
        sql = sql[1:-1]

    if sql.endswith(';'):
        sql = sql[:-1]

    sql = sql.replace("\"\"", "\"")
    sql = sql.replace("\"", " \" ")
    sql = sql.replace("  ", " ")

    # 7 . 5 -> 7.5
    # sql = sql.replace(" . ", ".")
    sql = re.sub(r'([0-9]+) \. ([0-9]+)', r'\1.\2', sql)

    #  stanley.monahan @ example.org -->  stanley.monahan@example.org
    sql = sql.replace(" @ ", "@")

    # standardize ()
    sql = sql.replace('(', ' ( ').replace(')', ' ) ')

    sql = ' '.join(sql.split())

    return sql.strip()


def remove_quotes(text: str) -> str:
    return text.replace('\'', '').replace('"', '')


def is_sql_keyword(text: str, sql_keyword_set: Set) -> bool:
    s_text = text
    if s_text[0] == '(':
        s_text = s_text[1:]
    if s_text in sql_keyword_set:
        return True
    return False


def extract_nl_template(duorat_preprocessor: AbstractPreproc,
                        tab_mask_dict: Dict[str, str],
                        col_mask_dict: Dict[str, Dict],
                        val_mask_dict: bidict,
                        question: str,
                        db_path: str,
                        max_ngrams: int = 5) -> str:
    def _maybe_naive_postprocess_slml_output(slml: str) -> str:
        return slml.replace("-LRB-", "(").replace("-RRB-", ")")

    sql_schema = get_processed_db_cached(db_path=db_path, duorat_preprocessor=duorat_preprocessor)
    # print(sql_schema)

    slml_question: str = duorat_preprocessor.schema_linker.question_to_slml(
        question=question, sql_schema=sql_schema,
    )
    slml_question = _maybe_naive_postprocess_slml_output(slml=slml_question)
    # print(slml_question)

    parser = SLMLParser(sql_schema=sql_schema, tokenizer=duorat_preprocessor.tokenizer)
    parser.feed(data=slml_question)

    nl_token_list = []
    for question_token in parser.question_tokens:
        # print(question_token)
        match_tags = question_token.match_tags
        if len(match_tags) > 0:  # matching happens!
            best_match = {}
            for match_tag in match_tags:
                if isinstance(match_tag, TableMatchTag):
                    table_name = sql_schema.original_table_names[match_tag.table_id]
                    if table_name in tab_mask_dict:
                        if 'table' not in best_match:
                            best_match['table'] = (tab_mask_dict[table_name], match_tag.confidence)
                        else:
                            if match_tag.confidence > best_match['table'][1]:
                                best_match['table'] = (tab_mask_dict[table_name], match_tag.confidence)
                    else:
                        if isinstance(match_tag.confidence, HighConfidenceMatch):
                            best_match['table'] = (get_table_mask_sid(mask_dict=tab_mask_dict,
                                                                      tab_name=table_name),
                                                   match_tag.confidence)
                elif isinstance(match_tag, ColumnMatchTag):
                    table_name = sql_schema.original_table_names[match_tag.table_id]
                    column_name = sql_schema.original_column_names[match_tag.column_id]
                    if table_name in col_mask_dict:
                        if column_name in col_mask_dict[table_name]:
                            if 'column' not in best_match:
                                best_match['column'] = (
                                    f"{tab_mask_dict[table_name]}.{col_mask_dict[table_name][column_name]}",
                                    match_tag.confidence)
                            else:
                                if match_tag.confidence > best_match['column'][1]:
                                    best_match['column'] = (
                                        f"{tab_mask_dict[table_name]}.{col_mask_dict[table_name][column_name]}",
                                        match_tag.confidence)
                        else:
                            if isinstance(match_tag.confidence, HighConfidenceMatch):
                                best_match['column'] = (
                                    f"{get_table_mask_sid(mask_dict=tab_mask_dict, tab_name=table_name)}.{get_column_mask_sid(mask_dict=col_mask_dict, tab_name=table_name, col_name=column_name)}",
                                    match_tag.confidence)
                elif isinstance(match_tag, ValueMatchTag) and isinstance(match_tag.confidence, HighConfidenceMatch):
                    match_val = match_tag.value
                    if match_val in val_mask_dict:
                        if len(nl_token_list) > 0 and nl_token_list[-1].startswith("VALUE#"):
                            nl_token_list.pop()
                        best_match['value'] = val_mask_dict[match_val]
                    else:
                        best_match['value'] = f"{question_token.raw_value}"

            if len(best_match) == 0:
                if question_token.raw_value in val_mask_dict:
                    nl_token_list.append(val_mask_dict[question_token.raw_value])
                else:
                    nl_token_list.append(question_token.raw_value)
            else:
                # Suppose, column > table > value
                if 'value' in best_match:
                    best_match_str = best_match['value']
                    if best_match_str in val_mask_dict:
                        best_match_str = val_mask_dict[best_match_str]
                if 'table' in best_match:
                    if 'value' in best_match and not isinstance(best_match['table'][1], HighConfidenceMatch):
                        best_match_str = best_match['value']
                    else:
                        best_match_str = best_match['table'][0]
                if 'column' in best_match:
                    best_match_str = best_match['column'][0]

                if len(nl_token_list) > 0 and nl_token_list[-1] == best_match_str:
                    nl_token_list.pop()
                nl_token_list.append(best_match_str)
        else:
            # 5-gram -> 4-gram -> 3-gram -> 2-gram -> 1-gram. Here, we only allow max 5-grams. Inefficient implementation ^_%.
            def _get_ngram_list(cur_token: str, max_n: int = 5) -> Dict[int, str]:
                ngrams = {2: None, 3: None, 4: None, 5: None}
                for i in range(max_n, 1, -1):
                    if len(nl_token_list) >= i - 1:
                        ngrams[i] = duorat_preprocessor.tokenizer.detokenize(
                            xs=f"{' '.join(nl_token_list[-(i - 1):])} {cur_token}".split())
                return ngrams

            ngrams = _get_ngram_list(cur_token=question_token.raw_value, max_n=max_ngrams)
            ngrams[1] = question_token.raw_value
            matched = False
            for n in range(max_ngrams, 0, -1):
                if ngrams[n] and ngrams[n] in val_mask_dict:
                    for j in range(n - 1):
                        nl_token_list.pop()
                    nl_token_list.append(val_mask_dict[ngrams[n]])
                    matched = True
                    break

            if not matched:  # otherwise, normal token
                nl_token_list.append(question_token.raw_value)

    return duorat_preprocessor.tokenizer.detokenize(xs=nl_token_list)


def get_table_mask_sid(mask_dict: Dict[str, str], tab_name: str) -> str:
    if tab_name in mask_dict:
        return mask_dict[tab_name]
    mask_dict[tab_name] = f"TABLE#{len(mask_dict)}"
    return mask_dict[tab_name]


def get_column_mask_sid(mask_dict: Dict[str, Dict], tab_name: str, col_name: str) -> str:
    if tab_name in mask_dict:
        if col_name in mask_dict[tab_name]:
            return mask_dict[tab_name][col_name]
        else:
            mask_dict[tab_name][col_name] = f"COLUMN#{len(mask_dict[tab_name])}"
    else:
        mask_dict[tab_name] = {}
        mask_dict[tab_name][col_name] = f"COLUMN#0"
    return mask_dict[tab_name][col_name]


SQL_OP_LIST = ['<', '<=', '>', '>=', '=', '!=', 'LIKE', 'BETWEEN', 'AND']


def extract_nl2sql_templates(sql_kw_file: str,
                             input_file: str,
                             output_file: str,
                             output_in_csv: bool,
                             duorat_preprocessor: AbstractPreproc,
                             with_op_denotation: bool = True,
                             with_sc_denotation: bool = True,
                             top_k_t: int = 0,
                             top_k_e: int = 0,
                             debug_n: int = 0):
    def _maybe_correct_val_mask_dict(g_sql: str, p_sql: str, m_dict: bidict):
        def _get_potential_op_values(sql: str) -> List[str]:
            op_values = []
            sql_tokens = sql.split()
            see_where = False
            for ind, stok in enumerate(sql_tokens):
                if stok in ["WHERE", "Where", "where"]:
                    see_where = True

                if see_where and stok in SQL_OP_LIST:
                    j = ind + 1
                    if sql_tokens[j] == '\'' or sql_tokens[j] == '\"' \
                            or sql_tokens[j].startswith('\'') or sql_tokens[j].startswith('"'):
                        if sql_tokens[j] == '\'' or sql_tokens[j] == '\"':
                            j += 1

                        val_list = []
                        while j < len(sql_tokens):
                            if sql_tokens[j] == '\'' or sql_tokens[j] == '\"':
                                break

                            if sql_tokens[j].endswith(')') and len(sql_tokens[j]) > 1:
                                sql_tokens[j] = sql_tokens[j][:-1]

                            val_list.append(sql_tokens[j])

                            if sql_tokens[j].endswith('\'') or sql_tokens[j].endswith('"'):
                                break

                            j += 1
                        val_str = remove_quotes(text=' '.join(val_list))
                    else:
                        val_str = remove_quotes(text=sql_tokens[j])

                    op_values.append(val_str)
            return op_values

        p_op_values = _get_potential_op_values(sql=p_sql)
        g_op_values = _get_potential_op_values(sql=g_sql)
        if len(p_op_values) != len(g_op_values):
            return

        for p_op_val, g_op_val in zip(p_op_values, g_op_values):
            p_op_val = p_op_val.replace('%', '').replace(' _ ', '_').strip()
            g_op_val = g_op_val.replace('%', '').replace(' _ ', '_').strip()
            if p_op_val in m_dict and g_op_val not in m_dict:
                p_op_val_mask = m_dict[p_op_val]
                del m_dict[p_op_val]
                m_dict[g_op_val] = p_op_val_mask

        return

    # read SQL keywords
    sql_keyword_set = set()
    with open(sql_kw_file) as f:
        for line in f:
            line = line.strip()
            for kw_token in line.split():
                sql_keyword_set.add(kw_token)

    # read data
    with open(input_file) as f:
        data = json.load(f)

    template_collection = {}
    templates_by_hardness = {"easy": {}, "medium": {}, "hard": {}, "extra": {}}
    templates_by_sql = {}
    for ind, item in enumerate(tqdm.tqdm(data["per_item"])):
        if 0 < debug_n <= ind + 1:
            break

        question = postprocess_question(question=item["question"])
        # if 'display the employee number, name( first name and last name ) and jo' not in question:
        #     continue  # for debugging only

        gold_sql = postprocess_sql(sql=item["gold"])
        predicted_sql = postprocess_sql(sql=item["predicted"])
        predicted_parse_error = bool(item["predicted_parse_error"])
        exact = bool(item["exact"])
        db_path = item["db_path"]
        db_name = item["db_name"]
        hardness = item["hardness"]

        # *** Extract SQL template
        tab_mask_dict = {}
        col_mask_dict = {}
        val_mask_dict = bidict({})
        if exact and not predicted_parse_error:
            prev_sql_token = ''
            predicted_sql_tokens = predicted_sql.split()
            template_sql_token_list = []
            op_counter = 0
            start_quote = False
            for sql_token in predicted_sql_tokens:
                if sql_token in ['\'', '"']:
                    start_quote = not start_quote

                if '.' in sql_token and len(sql_token.split('.')) == 2 and not start_quote:
                    dot_pos = sql_token.find('.')
                    open_bracket_pos = sql_token.find('(') + 1

                    table_name = sql_token[open_bracket_pos: dot_pos]

                    def _replace_table_name(tok: str, rep_tok: str, rep_val: str) -> str:
                        toks = tok.split('.')
                        toks[0] = toks[0].replace(rep_tok, rep_val)
                        return '.'.join(toks)

                    # sql_token = sql_token.replace(table_name,
                    #                               get_table_mask_sid(mask_dict=tab_mask_dict,
                    #                                                  tab_name=table_name))
                    sql_token = _replace_table_name(tok=sql_token,
                                                    rep_tok=table_name,
                                                    rep_val=get_table_mask_sid(mask_dict=tab_mask_dict,
                                                                               tab_name=table_name))

                    dot_pos = sql_token.find('.')
                    if open_bracket_pos == 0:
                        if sql_token.find(')') == -1:
                            if sql_token.find(',') == -1:
                                col_name = sql_token[dot_pos + 1:]
                            else:
                                col_name = sql_token[dot_pos + 1: sql_token.find(',')]
                        else:
                            col_name = sql_token[dot_pos + 1: sql_token.find(')')]
                    else:
                        col_name = sql_token[dot_pos + 1: sql_token.find(')')]

                    sql_token = sql_token.replace(col_name, get_column_mask_sid(mask_dict=col_mask_dict,
                                                                                tab_name=table_name,
                                                                                col_name=col_name))
                elif prev_sql_token.upper() == 'FROM' or prev_sql_token.upper() == 'JOIN':
                    if sql_token.find(")") == -1:
                        sql_token = get_table_mask_sid(mask_dict=tab_mask_dict, tab_name=sql_token)
                    else:
                        sql_token = f"{get_table_mask_sid(mask_dict=tab_mask_dict, tab_name=sql_token[:-1])})"
                elif sql_token.upper() in SQL_OP_LIST and with_op_denotation:
                    sql_token = f'OP#{op_counter}'
                    op_counter += 1
                elif sql_token.upper() in ['ASC', 'DESC'] and with_sc_denotation:
                    sql_token = 'SC'
                elif not is_sql_keyword(text=sql_token,
                                        sql_keyword_set=sql_keyword_set) \
                        and sql_token.upper() not in SQL_OP_LIST and sql_token not in ['\'', '"']:
                    if 'OP#' in prev_sql_token or (prev_sql_token in ['\'', '"'] and sql_token != ')'):
                        if sql_token in val_mask_dict:
                            value_str = val_mask_dict[sql_token]
                        else:
                            value_str = f"VALUE#{len(val_mask_dict)}"

                        if sql_token[-1] == ')':
                            value_str = f"{value_str})"

                        val_mask_dict[sql_token.replace(')', '')] = value_str.replace(')', '')
                        sql_token = value_str
                    elif 'VALUE#' in prev_sql_token:
                        last_sql_token = template_sql_token_list.pop()
                        last_val_token = val_mask_dict.inverse[last_sql_token]
                        del val_mask_dict.inverse[last_sql_token]

                        mask_key = remove_quotes(text=f"{last_val_token} {sql_token}").replace(')', '').replace('%',
                                                                                                                '').replace(
                            '/', '').strip()
                        if mask_key not in val_mask_dict:
                            value_str = f"VALUE#{len(val_mask_dict)}"
                            if sql_token[-1] == ')':
                                value_str = f"{value_str})"

                            val_mask_dict.inverse[last_sql_token] = mask_key
                            sql_token = value_str
                        else:
                            value_str = val_mask_dict[mask_key]
                            if sql_token[-1] == ')':
                                value_str = f"{value_str})"

                            sql_token = value_str

                prev_sql_token = sql_token
                template_sql_token_list.append(sql_token)

            def _maybe_postprocess_parentheses_sql(sql: str) -> str:
                sql = sql.replace('( ', '(').replace(' )', ')')

                return sql

            sql_template = _maybe_postprocess_parentheses_sql(
                sql=postprocess_sql(sql=' '.join(template_sql_token_list)))
            # print(f"Predicted SQL: {predicted_sql}")
            # print(f"SQL Template: {sql_template}")

            # *** Extract NL template
            # print(tab_mask_dict)
            # print(col_mask_dict)
            _maybe_correct_val_mask_dict(p_sql=predicted_sql, g_sql=gold_sql, m_dict=val_mask_dict)
            nl_template = extract_nl_template(duorat_preprocessor=duorat_preprocessor,
                                              tab_mask_dict=tab_mask_dict,
                                              col_mask_dict=col_mask_dict,
                                              val_mask_dict=val_mask_dict,
                                              question=question,
                                              db_path=db_path)

            def _maybe_postprocess_last(nl_text: str, mask_dict: bidict) -> str:
                nl_text = nl_text.replace(' . ', '. ')
                if '@' in nl_text and '.' in nl_text:
                    nl_text = nl_text.replace('. ', '.').replace(' @ ', '@')

                for k, v in mask_dict.items():

                    def _maybe_handle_abbreviations(kval: str) -> str:
                        if kval in ['m', 'M']:
                            return 'male'
                        if kval in ['f', 'F']:
                            return 'female'
                        return kval

                    new_k = _maybe_handle_abbreviations(kval=k)
                    if new_k != '':
                        pos = nl_text.find(new_k)
                        if pos != -1:
                            if pos > 0 and pos + len(new_k) < len(nl_text) \
                                    and nl_text[pos - 1] in [' ', '"', "'"] and nl_text[pos + len(new_k)] in [' ', '"',
                                                                                                              "'"]:
                                nl_text = nl_text.replace(new_k, v)
                return nl_text

            nl_template = _maybe_postprocess_last(nl_text=nl_template, mask_dict=val_mask_dict)

            # print(f"NL: {question}")
            # print(f"NL Template: {nl_template}")
            #
            # print("------------------------")

            # *** Archive it without and with hardness level
            if (nl_template, sql_template) in template_collection:
                template_collection[(nl_template, sql_template)].append((question, gold_sql, db_name))
            else:
                template_collection[(nl_template, sql_template)] = [(question, gold_sql, db_name)]
            if (nl_template, sql_template) in templates_by_hardness[hardness]:
                templates_by_hardness[hardness][(nl_template, sql_template)].append((question, gold_sql, db_name))
            else:
                templates_by_hardness[hardness][(nl_template, sql_template)] = [(question, gold_sql, db_name)]
            if sql_template in templates_by_sql:
                templates_by_sql[sql_template].append({'original': question,
                                                       'delexicalized': nl_template,
                                                       'gold_sql': gold_sql
                                                       }
                                                      )
            else:
                templates_by_sql[sql_template] = [{'original': question,
                                                   'delexicalized': nl_template,
                                                   'gold_sql': gold_sql
                                                   }
                                                  ]

    # *** Write to files (.txt or .csv)
    print(f"Done! There are {len(template_collection)} NL<->SQL templates.")
    print(
        f"Writing resulting templates to output file {output_file} (w/ {'raw text format' if not output_in_csv else 'CSV format'})")
    if not output_in_csv:
        with open(f"{output_file}.txt", "w") as fout:
            for template, examples in sorted(template_collection.items(), key=lambda item: len(item[1]), reverse=True):
                for example in examples:
                    fout.write(f"{template[0]}\t{template[1]}\t{example[0]}\t{example[1]}\t{example[2]}\n")
                fout.write("-----------------------------\n")

        for hardness, hardness_template_collection in templates_by_hardness.items():
            with open(f"{output_file}.{hardness}.txt", "w") as fout:
                fout.write(f"{len(hardness_template_collection)}\n")
                for template, examples in sorted(hardness_template_collection.items(), key=lambda item: len(item[1]),
                                                 reverse=True):
                    for example in examples:
                        fout.write(f"{template[0]}\t{template[1]}\t{example[0]}\t{example[1]}\t{example[2]}\n")
                    fout.write("-----------------------------\n")

        with open(f"{output_file}.by_sql_topkt{top_k_t}_topke{top_k_e}.txt", "w") as fout:
            index = 0
            for sql_template, nl_template_list in sorted(templates_by_sql.items(),
                                                         key=lambda item: len(item[1]),
                                                         reverse=True):
                for nl_template in nl_template_list[:top_k_e if top_k_e != 0 else len(nl_template_list)]:
                    fout.write(f"{sql_template}\t{nl_template['delexicalized']}\t{nl_template['original']}\t{nl_template['gold_sql']}\n")

                if 0 < top_k_t <= index + 1:
                    break

                index += 1
    else:
        fieldnames = ['nl_template', 'sql_template', 'examples']
        with open(f"{output_file}.csv", 'w', newline='') as fcsvfile:
            writer = csv.DictWriter(fcsvfile, fieldnames=fieldnames)
            writer.writeheader()
            for template, examples in sorted(template_collection.items(), key=lambda item: len(item[1]), reverse=True):
                writer.writerow({'nl_template': template[0],
                                 'sql_template': template[1],
                                 'examples': str(examples)
                                 }
                                )

        for hardness, hardness_template_collection in templates_by_hardness.items():
            with open(f"{output_file}.{hardness}.csv", "w", newline='') as fcsvfile:
                hardness_writer = csv.DictWriter(fcsvfile, fieldnames=fieldnames)
                hardness_writer.writeheader()
                for template, examples in sorted(hardness_template_collection.items(), key=lambda item: len(item[1]),
                                                 reverse=True):
                    hardness_writer.writerow({'nl_template': template[0],
                                              'sql_template': template[1],
                                              'examples': str(examples)
                                              }
                                             )

        with open(f"{output_file}.by_sql_topkt{top_k_t}_topke{top_k_e}.csv", "w", newline='') as fcsvfile:
            by_sql_writer = csv.DictWriter(fcsvfile, fieldnames=['sql_template', 'nl_template', 'original_nl', 'gold_sql'])
            by_sql_writer.writeheader()
            index = 0
            for sql_template, nl_template_list in sorted(templates_by_sql.items(),
                                                         key=lambda item: len(item[1]),
                                                         reverse=True):
                top_nl_template_list = nl_template_list[:top_k_e if top_k_e != 0 else len(nl_template_list)]
                for nl_dict in top_nl_template_list:
                    by_sql_writer.writerow({'sql_template': sql_template,
                                            'nl_template': nl_dict['delexicalized'],
                                            'original_nl': nl_dict['original'],
                                            'gold_sql': nl_dict['gold_sql']
                                            })

                if 0 < top_k_t <= index + 1:
                    break

                index += 1

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NL2SQL Template Extraction')
    parser.add_argument("--sql-keyword-list-file",
                        help="The SQL keyword list file", required=True)
    parser.add_argument("--duorat-config-file",
                        help="The DuoRAT config", required=True)
    parser.add_argument("--duorat-prediction-file",
                        help="The DuoRAT prediction file", required=True)
    parser.add_argument("--logdir",
                        help="The logging dir", required=True)
    parser.add_argument("--template-output-file",
                        help="The NL2SQL template output file", required=True)
    parser.add_argument("--top-k-t",
                        help="Top-k templates", required=False, type=int, default=0)
    parser.add_argument("--top-k-e",
                        help="Top-k examples by template", required=False, type=int, default=0)
    parser.add_argument("--debug-n",
                        help="Process n examples (for debugging only)", required=False, type=int, default=0)
    parser.add_argument(
        "--output-in-csv",
        default=False,
        action="store_true",
        help="If True, write outputs in CSV format",
    )
    parser.add_argument(
        "--with-stemming",
        default=False,
        action="store_true",
        help="If True, do stemming with schema linker",
    )
    parser.add_argument(
        "--with-op-denotation",
        default=False,
        action="store_true",
        help="If True, do denotation for OP, e.g., >= < = ...",
    )
    parser.add_argument(
        "--with-sc-denotation",
        default=False,
        action="store_true",
        help="If True, do denotation for SC, e.g., ASC, DESC",
    )
    args, _ = parser.parse_known_args()

    # Initialize
    sql_kw_file = args.sql_keyword_list_file
    input_file = args.duorat_prediction_file
    logdir = args.logdir
    duorat_config_file = args.duorat_config_file
    output_file = args.template_output_file

    # DuoRAT config
    print("Initializing DuoRAT config...")
    config = json.loads(_jsonnet.evaluate_file(duorat_config_file))
    config['model']['preproc']['save_path'] = os.path.join(logdir, "data")
    config['model']['preproc']['schema_linker']['with_stemming'] = args.with_stemming

    # DuoRAT preprocessor
    print("Initializing DuoRAT preprocessor...")
    duorat_preprocessor: AbstractPreproc = registry.construct("preproc", config["model"]["preproc"])
    duorat_preprocessor.load()

    # Extract NL2SQL templates
    print(f"Extracting NL2SQL templates from {input_file}...")
    extract_nl2sql_templates(sql_kw_file=sql_kw_file,
                             input_file=input_file,
                             output_file=output_file,
                             output_in_csv=args.output_in_csv,
                             duorat_preprocessor=duorat_preprocessor,
                             with_op_denotation=args.with_op_denotation,
                             with_sc_denotation=args.with_sc_denotation,
                             top_k_t=args.top_k_t,
                             top_k_e=args.top_k_e,
                             debug_n=args.debug_n)

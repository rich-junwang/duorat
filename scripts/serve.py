import os
from os import listdir
from os.path import join, exists

import argparse

import pydantic
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import File, Form, UploadFile
import shutil
from datetime import datetime
import sqlite3
from typing import Dict, Tuple
import json
import re
import copy
from collections import OrderedDict

from moz_sql_parser import parse
from moz_sql_parser import format as format_sql

from nltk.tokenize.treebank import TreebankWordDetokenizer

from duorat.api import DuoratAPI, DuoratOnDatabase
from duorat.utils.evaluation import find_any_config
from third_party.spider.process_sql import get_sql
from third_party.spider.preprocess.schema import Schema

from scripts.train import Logger


# --------------------------------

def dump_db_json_schema(db_file: str, db_id: str) -> Dict:
    """read table, column info, keys, content and dump all to a JSON file"""

    conn = sqlite3.connect(db_file)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    conn.execute("pragma foreign_keys=ON")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {
        "type": "database",
        "name": db_id,
        "security_passed": True,  # no need password by default
        "objects": [],
        # ----
    }

    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        # print(table_name)

        table_info = {
            "type": "table",
            "name": table_name,
            "columns": [],
            "constraints": [],
            "rows": []
        }

        fks = conn.execute(
            "PRAGMA foreign_key_list('{}') ".format(table_name)
        ).fetchall()
        # print(fks)

        fk_holder = []
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        # print(fk_holder)
        fk_entries = []
        for fk in fk_holder:
            fk_entry = {
                "type": "FOREIGN KEY",
                "definition": f"FOREIGN KEY (`{fk[0][1]}`) REFERENCES `{fk[1][0]}`(`{fk[1][1]}`)"
            }
            fk_entries.append(fk_entry)

        pk_holder = []
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            # print(j)
            # print(col)
            if col[5] != 0:  # primary key
                pk_holder.append(col)

            col_entry = {
                "name": col[1],
                "type": col[2]
            }
            table_info["columns"].append(col_entry)

        pk_str = ','.join([f"\"{pk[1]}\"" for pk in pk_holder])
        pk_entries = {
            "type": "PRIMARY KEY",
            "definition": f"PRIMARY KEY ({pk_str})"  # \"Cinema_ID\",\"Film_ID\"
        }
        if len(pk_entries):
            table_info["constraints"].append(pk_entries)
        if len(fk_entries):
            table_info["constraints"].extend(fk_entries)

        cur = conn.execute("SELECT * FROM '{}'".format(table_name))
        for i, row in enumerate(cur.fetchall()):
            # print(i)
            # print(row)
            table_info["rows"].append(list(row))

        data["objects"].append(table_info)

    # print(data)

    return data


# --------------------------------

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8100",
    "http://localhost:8200",
    "http://localhost:8300",
    "http://localhost:8400",
    "http://localhost:8500",
    "http://100.102.86.190:8100",
    "http://100.102.86.190:8300",
    "http://100.102.86.190:8400",
    "http://100.102.86.234:8101",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "./data/database"
DB_PATH_USER = f"{DB_PATH}/user_db"
SPIDER_TRAIN_DBS = set([
    "department_management",
    "farm",
    "student_assessment",
    "bike_1",
    "book_2",
    "musical",
    "twitter_1",
    "product_catalog",
    "flight_1",
    "allergy_1",
    "store_1",
    "journal_committee",
    "customers_card_transactions",
    "race_track",
    "coffee_shop",
    "chinook_1",
    "insurance_fnol",
    "medicine_enzyme_interaction",
    "university_basketball",
    "phone_1",
    "match_season",
    "climbing",
    "body_builder",
    "election_representative",
    "apartment_rentals",
    "game_injury",
    "soccer_1",
    "performance_attendance",
    "college_2",
    "debate",
    "insurance_and_eClaims",
    "customers_and_invoices",
    "wedding",
    "theme_gallery",
    "epinions_1",
    "riding_club",
    "gymnast",
    "small_bank_1",
    "browser_web",
    "wrestler",
    "school_finance",
    "protein_institute",
    "cinema",
    "products_for_hire",
    "phone_market",
    "gas_company",
    "party_people",
    "pilot_record",
    "cre_Doc_Control_Systems",
    "company_1",
    "local_govt_in_alabama",
    "formula_1",
    "machine_repair",
    "entrepreneur",
    "perpetrator",
    "csu_1",
    "candidate_poll",
    "movie_1",
    "county_public_safety",
    "inn_1",
    "local_govt_mdm",
    "party_host",
    "storm_record",
    "election",
    "news_report",
    "restaurant_1",
    "customer_deliveries",
    "icfp_1",
    "sakila_1",
    "loan_1",
    "behavior_monitoring",
    "assets_maintenance",
    "station_weather",
    "college_1",
    "sports_competition",
    "manufacturer",
    "hr_1",
    "music_1",
    "baseball_1",
    "mountain_photos",
    "program_share",
    "e_learning",
    "insurance_policies",
    "hospital_1",
    "ship_mission",
    "student_1",
    "company_employee",
    "film_rank",
    "cre_Doc_Tracking_DB",
    "club_1",
    "tracking_grants_for_research",
    "network_2",
    "decoration_competition",
    "document_management",
    "company_office",
    "solvency_ii",
    "entertainment_awards",
    "customers_campaigns_ecommerce",
    "college_3",
    "department_store",
    "aircraft",
    "local_govt_and_lot",
    "school_player",
    "store_product",
    "soccer_2",
    "device",
    "cre_Drama_Workshop_Groups",
    "music_2",
    "manufactory_1",
    "tracking_software_problems",
    "shop_membership",
    "voter_2",
    "products_gen_characteristics",
    "swimming",
    "railway",
    "customers_and_products_contacts",
    "dorm_1",
    "customer_complaints",
    "workshop_paper",
    "tracking_share_transactions",
    "cre_Theme_park",
    "game_1",
    "customers_and_addresses",
    "music_4",
    "roller_coaster",
    "ship_1",
    "city_record",
    "e_government",
    "school_bus",
    "flight_company",
    "cre_Docs_and_Epenses",
    "scientist_1",
    "wine_1",
    "train_station",
    "driving_school",
    "activity_1",
    "flight_4",
    "tracking_orders",
    "architecture",
    "culture_company",
    "geo",
    "scholar",
    "yelp",
    "academic",
    "imdb",
    "restaurants"])
SPIDER_DEV_DBS = set([
    "concert_singer",
    "pets_1",
    "car_1",
    "flight_2",
    "employee_hire_evaluation",
    "cre_Doc_Template_Mgt",
    "course_teach",
    "museum_visit",
    "wta_1",
    "battle_death",
    "student_transcripts_tracking",
    "tvshow",
    "poker_player",
    "voter_1",
    "world_1",
    "orchestra",
    "network_1",
    "dog_kennels",
    "singer",
    "real_estate_properties"])
DBS_REQUIRE_PASSWORDS = {}
duorat_model = None
logger = None
do_sql_post_processing = False
detokenizer = TreebankWordDetokenizer()


class Text2SQLInferenceRequest(pydantic.BaseModel):
    text_question: str
    db_id: str
    db_type: str


class Text2SQLWithFollowUpInferenceRequest(Text2SQLInferenceRequest):
    history: Dict[str, str]


class Text2SQLInferenceResponse(pydantic.BaseModel):
    sql_query: str
    score: str
    execution_result: str


class Text2SQLQueryDBRequest(pydantic.BaseModel):
    query_type: str
    db_id: str
    password: str


class Text2SQLQueryDBResponse(pydantic.BaseModel):
    db_id: str
    db_json_content: str


class Text2SQLValidationRequest(pydantic.BaseModel):
    sql_query: str
    db_id: str
    db_type: str


class Text2SQLValidationResponse(pydantic.BaseModel):
    sql_query: str
    db_id: str
    validation_result: str
    execution_result: str


def show_schema(duorat_on_db: DuoratOnDatabase):
    for table in duorat_on_db.schema.tables:
        if logger:
            logger.log(f"Table: {table.name} {table.orig_name}")
        else:
            print("Table", f"{table.name} ({table.orig_name})")
        for column in table.columns:
            if logger:
                logger.log(f"    Column {column.name} {column.orig_name}")
            else:
                print("    Column", f"{column.name} ({column.orig_name})")


NEW_VALID = re.compile(r"^[a-zA-Z_%]\w*$")


def postprocess_sql_for_like_clause(sql: str) -> str:
    """
        A heuristics-based SQL post-processing function for converting EQ comparison into LIKE
        Args:
            sql: a raw input sql
        Returns:
            a post-processed sql
    """

    def _detokenize(txt: str) -> str:
        return detokenizer.detokenize(txt.strip().split(" ")).replace(" .", ".")  # This is a temporary fix.

    def _remove_duplicates(txt: str) -> str:
        return ' '.join(list(OrderedDict.fromkeys(txt.replace('%', '').split(' '))))

    def _put_like_operator(txt: str) -> str:
        return ' '.join([f"%{word}%" for word in txt.split(' ')])

    def _mask_dot(txt: str, rev: bool = False) -> str:
        if rev:
            return txt.replace("[DOT]", ".")
        return txt.replace(".", "[DOT]")

    def _should_quote(identifier):
        """
        Return true if a given identifier should be quoted.

        This is usually true when the identifier:

          - is a reserved word
          - contain spaces
          - does not match the regex `[a-zA-Z_]\\w*`

        """
        return identifier != "*" and (not NEW_VALID.match(identifier))

    def _replace_eq_by_like(eq: Dict):
        eq_clause = eq['eq']
        if isinstance(eq_clause[1], str):
            eq_clause[1] = _put_like_operator(_mask_dot(txt=_detokenize(txt=_remove_duplicates(txt=str(eq_clause[1]))),
                                                        rev=False))
        else:
            return

        tmp_eq_clause = copy.deepcopy(eq_clause)
        eq.pop('eq', None)
        eq['like'] = tmp_eq_clause

        return

    try:
        parsed_sql_dict = parse(sql)
        if 'where' in parsed_sql_dict:
            where_clause = parsed_sql_dict['where']
            if 'and' in where_clause or 'or' in where_clause:
                and_or_clause = where_clause['and' if 'and' in where_clause else 'or']
                for eq in and_or_clause:
                    if 'eq' in eq:
                        _replace_eq_by_like(eq=eq)
            else:
                if 'eq' in where_clause:
                    _replace_eq_by_like(eq=where_clause)

            if 'like' in where_clause:
                like_clause = where_clause['like']
                like_clause[1] = _put_like_operator(txt=_detokenize(txt=_remove_duplicates(txt=like_clause[1])))

        return _mask_dot(txt=format_sql(parsed_sql_dict, should_quote=_should_quote), rev=True)
    except Exception as e:
        if logger:
            logger.log(f"[ERROR] - {str(e)}")
        else:
            print(f"[ERROR] - {str(e)}")
        return sql


def postprocess_sql_for_star_column(sql: str, db_path: str) -> str:
    """
        A heuristics-based SQL post-processing function for replacing * column in SELECT by meaningful columns
        Args:
            sql: a raw input sql
        Returns:
            a post-processed sql

        Example:
            postprocess_sql_for_star_column(sql='SELECT * FROM  job_history AS T1 JOIN employees AS T2 ON
                                                 T1.employee_id  =  T2.employee_id WHERE T2.salary  >=  12000',
                                            db_path='./data/database/hr_1/hr_1.sqlite')
    """

    def _get_schema_from_json(db_path: str) -> Tuple[Dict, Dict, Dict]:
        db_path_splits = db_path.split('/')
        json_file_path = os.path.join('/'.join(db_path_splits[:-1]), 'tables.json')
        if not os.path.exists(json_file_path):
            return None
        with open(json_file_path) as f:
            data = json.load(f)
            db = data[0]

            schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
            schema_ori = {}  # similar to schema with original table/column names
            column_names_original = db["column_names_original"]
            table_names_original = db["table_names_original"]
            tables = {
                "column_names_original": column_names_original,
                "table_names_original": table_names_original,
            }
            for i, tabn in enumerate(table_names_original):
                table = str(tabn).lower()
                cols = [str(col).lower() for td, col in column_names_original if td == i]
                schema[table] = cols

                table = str(tabn)
                cols = [str(col) for td, col in column_names_original if td == i]
                schema_ori[table] = cols

            return schema, schema_ori, tables

    def _get_schema_mapping(tables: Dict) -> Tuple[Dict[id, str], Dict[id, str]]:
        tab2id = {}
        col2id = {}

        column_names_original = tables["column_names_original"]
        table_names_original = tables["table_names_original"]
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                col2id[i] = '*'
            else:
                key = table_names_original[tab_id]
                val = col
                col2id[i] = key + "." + val

        for i, tab in enumerate(table_names_original):
            key = tab
            tab2id[i] = key

        return tab2id, col2id

    schema_obj = _get_schema_from_json(db_path=db_path)
    if schema_obj is None:
        return sql
    schema, schema_ori, tables = schema_obj
    tab2id, col2id = _get_schema_mapping(tables=tables)
    spider_sql: dict = get_sql(schema=Schema(schema=schema, table=tables), query=sql)  # a dictionary

    # for from clause
    from_clause = spider_sql['from']['table_units']
    # print(from_clause)
    table_name = None
    if len(from_clause) > 0:
        table_ref = from_clause[0][1]
        if isinstance(table_ref, int):
            table_name = tab2id[table_ref]

    # for select clause
    if table_name:
        select_clause = spider_sql['select']
        # print(select_clause)
        if len(select_clause[1]) == 1:  # only if SELECT *
            star_col = select_clause[1][0]
            if isinstance(star_col, tuple) or isinstance(star_col, list):
                if star_col[0] == 0:  # not a COUNT, MIN, MAX, AVG
                    # only for col_unit1
                    col_id = star_col[1][1][1]
                    if isinstance(col_id, int):
                        col_name = col2id[col_id]
                        if col_name == '*':
                            rep_col_names = ', '.join([f"{table_name}.{col_n}"for col_n in schema_ori[table_name]])
                            # naive replacement
                            sql = sql.replace('SELECT * ', f'SELECT {rep_col_names} ')

    return sql


def postprocess_sql(sql: str, db_path: str) -> str:
    """
    A heuristics-based SQL post-processing function
    Args:
        sql: a raw input sql
        db_path: db path to .sqlite/.db
    Returns:
        a post-processed sql
    """

    try:
        # convert all eq to like for fuzzy string matching
        final_sql = postprocess_sql_for_like_clause(sql=sql)

        # convert * column in SELECT clause into meaningful columns
        final_sql = postprocess_sql_for_star_column(sql=final_sql, db_path=db_path)
    except:
        final_sql = sql  # revert back to original result if getting any exception.

    return final_sql


def ask_any_question(question: str,
                     duorat_on_db: DuoratOnDatabase) -> Text2SQLInferenceResponse:
    if '@EXECUTE' not in question and '@execute' not in question:
        if ('<tm' in question and '</tm>' in question) \
                or ('<cm' in question and '</cm>' in question) \
                or ('<vm' in question and '</vm>' in question):
            model_results = duorat_on_db.infer_query(question='', slml_question=question)
        else:
            model_results = duorat_on_db.infer_query(question=question, slml_question=None)
        sql = model_results['query']
        score = str(model_results["score"])
    else:  # an implicit db execution query (for debugging only)
        sql = re.compile(re.escape('@execute'), re.IGNORECASE).sub('', question).strip()
        score = "n/a"

    if do_sql_post_processing:
        logger.log(f"SQL: {sql}")
        sql = postprocess_sql(sql=sql, db_path=duorat_on_db.db_path)

    if logger:
        logger.log(f"Question: {question}")
        logger.log(f"Final SQL: {sql}")

    try:
        exe_results = duorat_on_db.execute(sql)
        if isinstance(exe_results, list):
            exe_results = [tuple([str(r) if not isinstance(r, str) else r for r in list(res)]) for res in exe_results]
        if logger:
            logger.log(f"Execution results: {exe_results}")
        return Text2SQLInferenceResponse(sql_query=sql,
                                         score=score,
                                         execution_result=f"{exe_results}"
                                         )
    except Exception as e:
        print(str(e))

    if logger:
        logger.log("Execution results: [UNEXECUTABLE]")
    return Text2SQLInferenceResponse(sql_query=sql,
                                     score=score,
                                     execution_result="[UNEXECUTABLE]"
                                     )


def ask_any_question_with_followup(question: str,
                                   history: Dict[str, str],  # currently support one previous question only
                                   duorat_on_db: DuoratOnDatabase) -> Text2SQLInferenceResponse:
    duorat_history = None
    if duorat_model.config['model']['preproc']['interaction_type'] == 'source':
        duorat_history = [(history['text'], None, '')]
    elif duorat_model.config['model']['preproc']['interaction_type'] == 'target':
        duorat_history = [('', None, history['sql'])]
    elif duorat_model.config['model']['preproc']['interaction_type'] == 'source&target':
        duorat_history = [(history['text'], None, history['sql'])]

    if '@EXECUTE' not in question and '@execute' not in question:
        if ('<tm' in question and '</tm>' in question) \
                or ('<cm' in question and '</cm>' in question) \
                or ('<vm' in question and '</vm>' in question):
            model_results = duorat_on_db.infer_query(question='', slml_question=question, history=duorat_history)
        else:
            model_results = duorat_on_db.infer_query(question, history=duorat_history)
        sql = model_results['query']
        score = str(model_results["score"])
    else:  # an implicit db execution query (for debugging only)
        sql = re.compile(re.escape('@execute'), re.IGNORECASE).sub('', question).strip()
        score = "n/a"

    if do_sql_post_processing:
        logger.log(f"SQL: {sql}")
        sql = postprocess_sql(sql=sql, db_path=duorat_on_db.db_path)

    if logger:
        logger.log(f"Question: {question}")
        logger.log(f"Final SQL: {sql}")

    try:
        exe_results = duorat_on_db.execute(sql)
        if isinstance(exe_results, list):
            exe_results = [tuple([str(r) if not isinstance(r, str) else r for r in list(res)]) for res in exe_results]
        if logger:
            logger.log(f"Execution results: {exe_results}")
        return Text2SQLInferenceResponse(sql_query=sql,
                                         score=score,
                                         execution_result=f"{exe_results}"
                                         )
    except Exception as e:
        print(str(e))

    if logger:
        logger.log("Execution results: [UNEXECUTABLE]")
    return Text2SQLInferenceResponse(sql_query=sql,
                                     score=score,
                                     execution_result="[UNEXECUTABLE]"
                                     )


def _get_proper_db_id(db_info: str) -> str:
    return db_info.split()[0]  # @Vu Hoang: This is very buggy. We should redesign this in the future.


@app.post('/text2sql/query_db', response_class=JSONResponse)
async def query_db(request: Text2SQLQueryDBRequest):
    print(f'Attempting for a request: {request.query_type}')

    def _get_db_examples(db_name: str) -> int:
        json_example_path = os.path.join(DB_PATH, db_name, "examples.json")
        examples = json.load(open(json_example_path))
        return len(examples)

    def _get_db_file_size(db_name: str) -> str:
        db_file_path = os.path.join(DB_PATH, db_name, f"{db_name}.sqlite")
        db_file_stats = os.stat(db_file_path)
        db_file_size = db_file_stats.st_size
        if db_file_size < 1024:
            return f"{db_file_size} bytes"
        elif 1024 <= db_file_size < 1024 * 1024:
            return f"{int(db_file_size / 1024)} Kb"  # Kb
        elif 1024 * 1024 <= db_file_size < 1024 * 1024 * 1024:
            return f"{db_file_size / (1024 * 1024):.2f} Mb"  # in Mb
        else:
            return f"{db_file_size / (1024 * 1024 * 1024):.2f} Gb"  # in Gb

    if request.query_type == '[ALL_DB]':
        db_names = [
            df for df in listdir(path=DB_PATH) \
            if exists(join(DB_PATH, df, df + ".sqlite")) and "user_db" not in df and "_test" not in df
        ]

        # add meta info to db_names
        new_db_names = []
        for db_name in db_names:
            require_password = False
            if db_name in DBS_REQUIRE_PASSWORDS:
                require_password = True

            if db_name in SPIDER_TRAIN_DBS:
                new_db_name = f"{db_name} (Spider, trained, {_get_db_examples(db_name=db_name)} examples, {_get_db_file_size(db_name=db_name)})"
            elif db_name in SPIDER_DEV_DBS:
                new_db_name = f"{db_name} (Spider, unseen test, {_get_db_examples(db_name=db_name)} examples, {_get_db_file_size(db_name=db_name)})"
            else:
                new_db_name = f"{db_name} (new, unseen, {_get_db_file_size(db_name=db_name)})"
            # As of 27 July 2020, we add security to each database.
            new_db_name = {"name": new_db_name, "password": require_password}
            new_db_names.append(new_db_name)
        db_names = new_db_names

        return jsonable_encoder(
            Text2SQLQueryDBResponse(db_id='[ALL_DB]', db_json_content=json.dumps(db_names, indent=4)))
    elif request.query_type == '[CUR_DB]':
        db_id = _get_proper_db_id(db_info=request.db_id)

        continued = True
        if db_id in DBS_REQUIRE_PASSWORDS:
            if DBS_REQUIRE_PASSWORDS[db_id] != request.password:
                continued = False

        if continued:
            db_file = join(DB_PATH, db_id, f"{db_id}.sqlite")
            if exists(db_file):
                db_json_content = dump_db_json_schema(db_file=db_file, db_id=request.db_id)
            else:
                db_json_content = {}
        else:  # cannot continue due to security issue.
            db_json_content = {
                "type": "database",
                "name": request.db_id,
                "security_passed": False,
                "objects": [],
            }

        return jsonable_encoder(
            Text2SQLQueryDBResponse(db_id=request.db_id, db_json_content=json.dumps(db_json_content, indent=4)))

    return jsonable_encoder(Text2SQLQueryDBResponse(db_id='', db_json_content=''))


@app.post("/text2sql/query_db_file", response_class=JSONResponse)
async def query_db_file(
        db_file: UploadFile = File(...)
):
    print(f'Attempting for a request with db_file={db_file.filename}')
    curtime = datetime.now().strftime("%d%m%Y%H%M%S")
    new_db_id = f"{db_file.filename.replace('.sqlite', '').replace('.db', '')}_userdb_{curtime}"
    user_db_path = f"{DB_PATH_USER}/{new_db_id}.sqlite"
    with open(user_db_path, "wb") as buffer:
        shutil.copyfileobj(db_file.file, buffer)

    return jsonable_encoder(
        Text2SQLQueryDBResponse(db_id=new_db_id, db_json_content=json.dumps(dump_db_json_schema(db_file=user_db_path,
                                                                                                db_id=new_db_id),
                                                                            indent=4)))


@app.post("/text2sql/infer_file", response_class=JSONResponse)
async def text2sql_infer_file(
        db_file: UploadFile = File(...), text_question: str = Form(...)
):
    print(f'Attempting for a request with text="{text_question}" and db_file={db_file.filename}')
    curtime = datetime.now().strftime("%d%m%Y%H%M%S")
    db_path = f"{DB_PATH_USER}/{db_file.filename.replace('.sqlite', '').replace('.db', '')}_{curtime}.sqlite"
    with open(db_path, "wb") as buffer:
        shutil.copyfileobj(db_file.file, buffer)

    if logger:
        logger.log(f'DB path: {db_path}')

    duorat_on_db = DuoratOnDatabase(duorat=duorat_model,
                                    db_path=db_path,
                                    schema_path='')

    show_schema(duorat_on_db=duorat_on_db)
    results = ask_any_question(question=text_question, duorat_on_db=duorat_on_db)
    if not logger:
        print(results)

    return jsonable_encoder(results)


@app.post('/text2sql/infer', response_class=JSONResponse)
async def text2sql_infer(request: Text2SQLInferenceRequest):
    print(f'Attempting for a request: {request}')

    if request.db_type == 'u_db':
        db_path = f"{DB_PATH_USER}/{request.db_id}.sqlite"
        schema_path = ''
    elif request.db_type == 'c_db':
        db_id = _get_proper_db_id(db_info=request.db_id)
        db_path = f"{DB_PATH}/{db_id}/{db_id}.sqlite"
        schema_path = f"{DB_PATH}/{db_id}/tables.json"
        if not os.path.exists(schema_path):
            schema_path = ""

    if logger:
        logger.log(f'DB path: {db_path}')
        logger.log(f'Schema path: {schema_path}')

    duorat_on_db = DuoratOnDatabase(duorat=duorat_model,
                                    db_path=db_path,
                                    schema_path=schema_path)

    show_schema(duorat_on_db=duorat_on_db)
    results = ask_any_question(question=request.text_question, duorat_on_db=duorat_on_db)
    if not logger:
        print(results)

    return jsonable_encoder(results)


@app.post('/text2sql/infer_followup', response_class=JSONResponse)
async def text2sql_infer_followup(request: Text2SQLWithFollowUpInferenceRequest):
    print(f'Attempting for a request with a follow-up: {request}')

    if request.db_type == 'u_db':
        db_path = f"{DB_PATH_USER}/{request.db_id}.sqlite"
        schema_path = ''
    elif request.db_type == 'c_db':
        db_id = _get_proper_db_id(db_info=request.db_id)
        db_path = f"{DB_PATH}/{db_id}/{db_id}.sqlite"
        schema_path = f"{DB_PATH}/{db_id}/tables.json"
        if not os.path.exists(schema_path):
            schema_path = ""

    if logger:
        logger.log(f'DB path: {db_path}')
        logger.log(f'Schema path: {schema_path}')

    duorat_on_db = DuoratOnDatabase(duorat=duorat_model,
                                    db_path=db_path,
                                    schema_path=schema_path)

    show_schema(duorat_on_db=duorat_on_db)
    results = ask_any_question_with_followup(question=request.text_question,
                                             history=request.history,
                                             duorat_on_db=duorat_on_db)
    if not logger:
        print(results)

    return jsonable_encoder(results)


@app.post('/text2sql/validate_sql', response_class=JSONResponse)
async def text2sql_validate_sql(request: Text2SQLValidationRequest):
    def _is_parsable(sql: str) -> bool:
        try:
            _ = parse(sql)
        except:
            return False
        return True

    print(f'Attempting for a request: {request}')

    schema_path = ''
    if request.db_type == 'u_db':
        db_path = f"{DB_PATH_USER}/{request.db_id}.sqlite"
    elif request.db_type == 'c_db':
        db_id = _get_proper_db_id(db_info=request.db_id)
        db_path = f"{DB_PATH}/{db_id}/{db_id}.sqlite"

    if logger:
        logger.log(f'DB path: {db_path}')

    duorat_on_db = DuoratOnDatabase(duorat=duorat_model,
                                    db_path=db_path,
                                    schema_path=schema_path)

    validation_result = Text2SQLValidationResponse(sql_query=request.sql_query,
                                                   db_id=request.db_id,
                                                   validation_result="SQL is incorrect!",
                                                   execution_result="")
    try:
        if _is_parsable(sql=request.sql_query):
            exe_result = duorat_on_db.execute(query=request.sql_query)
            if isinstance(exe_result, list):
                exe_result = [tuple([str(r) if not isinstance(r, str) else r for r in list(res)]) for res in exe_result]
            else:
                exe_result = ""
            if exe_result != "":
                validation_result = Text2SQLValidationResponse(sql_query=request.sql_query,
                                                               db_id=request.db_id,
                                                               validation_result="SQL is correct!",
                                                               execution_result=f"{exe_result}")
        else:
            validation_result.validation_result = "SQL has syntax error(s)."
    except:
        print("Execution error encountered.")

    return jsonable_encoder(validation_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DuoRAT Text2SQL Inference Server Version 2.0')
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--server-port", required=False, type=int, default=8000)
    parser.add_argument("--config",
                        help="The configuration file. By default, an arbitrary configuration from the logdir is loaded")
    parser.add_argument("--db-path",
                        help="The database path. By default, ./data/database", default=DB_PATH)
    parser.add_argument(
        "--do-sql-post-processing",
        default=False,
        action="store_true",
        help="If True, do fuzzy string matching mechanism as a post-processing step for generated SQL.",
    )
    parser.add_argument(
        "--do-logging",
        default=False,
        action="store_true",
        help="If True, do logging; otherwise just print",
    )
    parser.add_argument(
        "--log-append",
        default=False,
        action="store_true",
        help="If True, append the content to the log file else re-open a new one.",
    )
    parser.add_argument("--log-file-name",
                        help="The logging file path.", default='serve.log')
    parser.add_argument("--db-passwords-file",
                        help="The DB passwords file path.", default='passwords.sec', required=False)

    args, _ = parser.parse_known_args()

    DB_PATH = args.db_path
    DB_PATH_USER = f"{DB_PATH}/user_db"
    try:
        os.makedirs(DB_PATH_USER, exist_ok=True)
    except OSError as error:
        print(error)

    # Initialize the logger
    if args.do_logging:
        log_file = os.path.join(args.logdir, args.log_file_name)
        if not args.log_append:
            with open(log_file, "w") as f:
                f.close()
        logger = Logger(log_path=log_file, reopen_to_flush=False)
        logger.log("Logging to {}".format(log_file))

    if args.do_sql_post_processing:
        do_sql_post_processing = True

    if os.path.exists(args.db_passwords_file):
        with open(args.db_passwords_file) as pwd_f:
            for line in pwd_f:  # @Vu Hoang: this is not an encripted file.
                line = line.strip()
                parts = line.split('\t')  # DB_ID\tACCESS_PASSWORD
                DBS_REQUIRE_PASSWORDS[parts[0]] = parts[1]

    # Initialize the model
    print('Initializing Text2SQL Inference Service...')
    duorat_model = DuoratAPI(args.logdir, find_any_config(args.logdir) if args.config is None else args.config)

    # Start serving service
    uvicorn.run(app, host="0.0.0.0", port=args.server_port)

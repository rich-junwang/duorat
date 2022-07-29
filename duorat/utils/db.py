import re
import os
import subprocess
import sqlite3


def fix_detokenization(query: str):
    query = query.replace('" ', '"').replace(' "', '"')
    query = query.replace("% ", "%").replace(" %", "%")
    query = re.sub("(\d) . (\d)", "\g<1>.\g<2>", query)
    return query


def add_collate_nocase(query: str):
    value_regexps = ['"[^"]*"', "'[^']*'"]
    value_strs = []
    for regex in value_regexps:
        value_strs += re.findall(regex, query)
    for str_ in set(value_strs):
        query = query.replace(str_, str_ + " COLLATE NOCASE ")
    return query


def convert_csv_to_sqlite(csv_path: str):
    # TODO: infer types when importing
    db_path = csv_path + ".sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
    subprocess.run(["sqlite3", db_path, ".mode csv", f".import {csv_path} Data"])
    return db_path


def execute(query: str, db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        # Temporary Hack: makes sure all literals are collated in a case-insensitive way
        query = add_collate_nocase(query)
        results = conn.execute(query).fetchall()
    except:
        results = "[ERROR]"
    finally:
        conn.close()
    return results

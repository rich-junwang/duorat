# get_tables_wikisql.py
# Vu Hoang
# @Oracle

import json
import os
import sys
import sqlite3

from duorat.preproc.utils import refine_schema_names


def dump_wikisql_db_json_schema(db_file, table_file):
    """read table and column info"""

    # read .tables.jsonl file
    table_cols_dict = {}
    f_table_file = open(table_file, 'r')
    for line in f_table_file:
        tab_info = json.loads(line.strip())
        table_cols_dict[f"table_{tab_info['id'].replace('-', '_')}"] = tab_info["header"]

    # read .db file
    conn = sqlite3.connect(db_file)
    conn.execute("pragma foreign_keys=ON")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    all_tables = []

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]

        table_info = {
            "db_id": f"{table_name}",
            "table_names_original": [],
            "table_names": [],
            "column_names_original": [(-1, "*")],
            "column_names": [(-1, "*")],
            "column_types": ["text"],
            "primary_keys": [],
            "foreign_keys": [],
        }

        table_info["table_names_original"].append(table_name)
        table_info["table_names"].append(table_name.lower().replace("_", " "))
        fks = conn.execute(
            "PRAGMA foreign_key_list('{}') ".format(table_name)
        ).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            if table_name in table_cols_dict:
                col_id = int(col[1].replace('col', ''))
                if 0 <= col_id < len(table_cols_dict[table_name]):
                    col_name = table_cols_dict[table_name][col_id]
                else:
                    print(f"WARNING: {col[1]} is not valid in {table_cols_dict[table_name]}.")
                    col_name = col[1]
            else:
                print(f"WARNING: {table_name} does not exist in {table_file}.")
                col_name = col[1]

            # post-process col_name
            col_name = str(col_name).strip()
            # col_name = "Fleet Series (Quantity)", Powertrain (Engine/Transmission)
            # --> Fleet Series Quantity,  Powertrain Engine Transmission
            col_name = col_name.replace('.', '').replace(',', '').replace(':', '').replace('&', '').replace('*', '').replace('!', '').replace('\'', '').replace('"', '').replace('?', '').replace(';', '').strip()
            col_name = col_name.replace('%', 'PERCENTAGE').strip()
            col_name = col_name.replace("’s", '').strip()
            col_name = col_name.replace("#", 'No').strip()
            col_name = col_name.replace("@", ' ').strip()
            col_name = col_name.replace(">", 'Greater Than').strip()
            col_name = col_name.replace("<", 'Less Than').strip()
            col_name = col_name.replace("from", 'from_').replace("From", "From_").strip()
            col_name = col_name.replace("avg", 'avg_').replace("Avg", "Avg_").strip()
            col_name = col_name.replace("count", 'count_').replace("Count", "Count_").strip()
            col_name = col_name.replace("none", 'none_').replace("None", "None_").strip()
            col_name = col_name.replace("’", '').replace('’', '').replace('‘', '').replace('“', '').replace('”', '').strip()
            col_name = col_name.replace("$", 'Dollars').strip()
            col_name = col_name.replace("(", '').replace(')', '').strip()
            col_name = col_name.replace("[", '').replace(']', '').strip()
            col_name = col_name.replace("{", '').replace('}', '').strip()
            col_name = col_name.replace('\\', '').strip()
            col_name = col_name.replace('/', ' ').replace('-', ' ').strip()
            col_name = col_name.replace("  ", " ").strip()

            table_info["column_names_original"].append((0, col_name.upper().replace(" ", "_")))
            table_info["column_names"].append((0, col_name.lower()))

            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if (
                    "char" in col_type
                    or col_type == ""
                    or "text" in col_type
                    or "var" in col_type
            ):
                table_info["column_types"].append("text")
            elif (
                    "int" in col_type
                    or "numeric" in col_type
                    or "decimal" in col_type
                    or "number" in col_type
                    or "id" in col_type
                    or "real" in col_type
                    or "double" in col_type
                    or "float" in col_type
            ):
                table_info["column_types"].append("number")
            elif "date" in col_type or "time" in col_type or "year" in col_type:
                table_info["column_types"].append("time")
            elif "boolean" in col_type:
                table_info["column_types"].append("boolean")
            else:
                table_info["column_types"].append("others")

            if col[5] == 1:
                table_info["primary_keys"].append(len(table_info["column_names"]) - 1)

        all_tables.append(table_info)

    return all_tables


if __name__ == "__main__":
    """
    Extract tables.json for a single sqlite DB
    """
    if len(sys.argv) < 2:
        print(
            "Usage: python get_tables_wikisql.py [sqlite file] [JSON table file] [output file name e.g. output.json]"
        )
        sys.exit()
    sqlite_file = sys.argv[1]
    table_file = sys.argv[2]
    output_file = sys.argv[3]

    assert sqlite_file.endswith('.db')
    db_id = os.path.basename(sqlite_file)[:-3]

    schemas = dump_wikisql_db_json_schema(sqlite_file, table_file)
    schemas = [refine_schema_names(schema) for schema in schemas]

    with open(output_file, "wt") as out:
        json.dump(schemas, out, sort_keys=True, indent=2, separators=(",", ": "))

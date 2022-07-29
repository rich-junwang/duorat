# sparc.py
# Vu Hoang @ Oracle Corp.

import json
from typing import Dict, Tuple, List, Optional
import re

from pydantic.dataclasses import dataclass

from duorat.datasets.spider import SpiderDataset, SpiderItem, SpiderSchema, schema_dict_to_spider_schema
from duorat.utils import registry
from third_party.spider import evaluation
from third_party.spider.preprocess.schema import _get_schemas_from_json, Schema
from third_party.spider.process_sql import get_sql


def load_original_schemas(tables_files: List[str]) -> Dict[str, Schema]:
    all_schemas = {}
    for tables_file in tables_files:
        raw_data = json.load(open(tables_file))
        for table_info_dict in raw_data:
            schemas, db_ids, tables = _get_schemas_from_json(data=[table_info_dict])
            for db_id in db_ids:
                all_schemas[db_id] = Schema(schemas[db_id], tables[db_id])
    return all_schemas


def load_tables(tables_files: List[str]) -> Tuple[Dict[str, SpiderSchema], Dict[str, Dict[str, str]]]:
    schemas = {}
    eval_foreign_key_maps = {}
    for tables_file in tables_files:
        schema_dicts = json.load(open(tables_file))
        for schema_dict in schema_dicts:
            db_id = schema_dict["db_id"]
            assert db_id not in schemas
            schemas[db_id] = schema_dict_to_spider_schema(schema_dict=schema_dict)
            eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(entry=schema_dict)

    return schemas, eval_foreign_key_maps


@dataclass
class SparcItem(SpiderItem):
    interaction: Optional[List[SpiderItem]] = None


@registry.register("dataset", "sparc")
class SparcDataset(SpiderDataset):
    def __init__(self, examples_files: List[str],
                 tables_files: List[str],
                 db_path: str,
                 ignore_patterns: Optional[str] = ""):
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_files=tables_files)
        original_schemas = load_original_schemas(tables_files=tables_files)

        # read examples_file
        for example_file in examples_files:
            raw_data = json.load(open(example_file))
            for entry in raw_data:  # entry={"database_id": "...", "interaction": [{"query": "...", "utterance_toks": "...", "utterance": "..., "sql": "...", }]
                interaction = []
                for utter_info in entry["interaction"]:
                    if "sql" not in utter_info:
                        entry["sql"] = get_sql(
                            original_schemas[entry["database_id"]], utter_info["query"]
                        )

                    if ignore_patterns and re.search(ignore_patterns, utter_info["utterance"]):
                        print(f"Ignoring utterance: {utter_info['utterance']}")
                        continue

                    utter_info["utterance"] = utter_info["utterance"].replace('*', '')
                    item = SparcItem(
                        question=utter_info["utterance"],
                        slml_question=None,
                        query=utter_info["query"],
                        spider_sql=utter_info["sql"],
                        spider_schema=self.schemas[entry["database_id"]],
                        db_path=self.get_db_path(db_id=entry["database_id"]),
                        orig=utter_info,
                        interaction=interaction
                    )
                    self.examples.append(item)
                    interaction.append(SpiderItem(
                        question=utter_info["utterance"],
                        slml_question=None,
                        query=utter_info["query"],
                        spider_sql=utter_info["sql"],
                        spider_schema=self.schemas[entry["database_id"]],
                        db_path=self.get_db_path(db_id=entry["database_id"]),
                        orig=utter_info)
                    )

    def __getitem__(self, idx) -> SparcItem:
        return self.examples[idx]

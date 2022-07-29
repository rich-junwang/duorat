# interactive
# Raymond Li, 2020-04-27
# Copyright (c) 2020 Element AI Inc. All rights reserved.
import json
import os
from typing import Optional, List, Union, Tuple

import _jsonnet
import torch

from duorat.asdl.asdl_ast import AbstractSyntaxTree
from duorat.datasets.spider import (
    SpiderItem,
    load_tables,
    SpiderSchema,
    schema_dict_to_spider_schema,
)
from duorat.datasets.sparc import (
    SparcItem
)
from duorat.preproc.utils import preprocess_schema_uncached, refine_schema_names
from duorat.types import RATPreprocItem, SQLSchema, Dict
from duorat.utils import registry
import duorat.models  # *** COMPULSORY: for registering classes. PLEASE DON'T REMOVE THIS.
from duorat.utils import saver as saver_mod
from duorat.utils.db import fix_detokenization, convert_csv_to_sqlite, execute
from third_party.spider.preprocess.get_tables import dump_db_json_schema


class ModelLoader:
    def __init__(self, config, from_heuristic: bool = False):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)
        self.from_heuristic = from_heuristic
        if from_heuristic:
            config["model"]["preproc"]["grammar"]["output_from"] = False

        # 0. Construct preprocessors
        self.model_preproc = registry.construct(
            "preproc", self.config["model"]["preproc"],
        )
        self.model_preproc.load()

    def load_model(self, logdir, step, allow_untrained=False, load_best=True):
        """Load a model (identified by the config used for construction) and return it"""
        # 1. Construct model
        model = registry.construct(
            "model", self.config["model"], preproc=self.model_preproc,
        )
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        # 2. Restore its parameters
        saver = saver_mod.Saver(model, None)
        last_step, best_validation_metric = saver.restore(
            logdir, step=step, map_location=self.device, load_best=load_best
        )
        if not allow_untrained and not last_step:
            raise Exception("Attempting to infer on untrained model")
        return model


class DuoratAPI(object):
    """Adds minimal preprocessing code to the DuoRAT model."""

    def __init__(self, logdir: str, config_path: str):
        self.config = json.loads(_jsonnet.evaluate_file(config_path))
        self.config['model']['preproc']['save_path'] = os.path.join(logdir, "data")
        self.inferer = ModelLoader(self.config)
        self.preproc = self.inferer.model_preproc
        self.model = self.inferer.load_model(logdir, step=None)

    def infer_query(self,
                    question: str,
                    spider_schema: SpiderSchema,
                    preprocessed_schema: SQLSchema,
                    slml_question: Optional[str] = None,
                    history: Optional[Union[List[str], List[Tuple[str, str, str]]]] = None,
                    beam_size: Optional[int] = 1,
                    decode_max_time_step: Optional[int] = 500
                    ):
        # TODO: we should only need the preprocessed schema here
        if history is not None:
            interaction = [SpiderItem(question=prev_question[0] if isinstance(prev_question, tuple) else prev_question,
                                      slml_question=prev_question[1] if isinstance(prev_question, tuple) else None,
                                      query=prev_question[2] if isinstance(prev_question, tuple) else "",
                                      spider_sql={},
                                      spider_schema=spider_schema,
                                      db_path="",
                                      orig={}) for prev_question in history]
            input_item = SparcItem(
                question=question,
                slml_question=slml_question,
                query="",
                spider_sql={},
                spider_schema=spider_schema,
                db_path="",
                orig={},
                interaction=interaction
            )
        else:
            input_item = SpiderItem(
                question=question,
                slml_question=slml_question,
                query="",
                spider_sql={},
                spider_schema=spider_schema,
                db_path="",
                orig={},
            )
        preproc_item: RATPreprocItem = self.preproc.preprocess_item(
            input_item,
            preprocessed_schema,
            AbstractSyntaxTree(production=None, fields=(), created_time=None),
        )
        finished_beams = self.model.parse(
            [preproc_item],
            decode_max_time_step=decode_max_time_step,
            beam_size=beam_size
        )

        if not finished_beams:
            return {
                "slml_question": input_item.slml_question,
                "query": "",
                "tokenized_query": "",
                "score": -1,
                "beams": [],
            }

        parsed_query = self.model.preproc.transition_system.ast_to_surface_code(
            asdl_ast=finished_beams[0].ast  # best on beams
        )
        parsed_query = self.model.preproc.transition_system.spider_grammar.unparse(
            parsed_query, spider_schema=spider_schema
        )
        return {
            "slml_question": input_item.slml_question,
            "query": fix_detokenization(parsed_query),
            "tokenized_query": parsed_query,
            "score": finished_beams[0].score,
            "beams": finished_beams
        }


class DuoratOnDatabase(object):
    """Run DuoRAT model on a given database."""

    def __init__(self, duorat: DuoratAPI, db_path: str, schema_path: Optional[str]):
        self.duorat = duorat
        self.db_path = db_path

        if self.db_path.endswith(".sqlite"):
            pass
        elif self.db_path.endswith(".csv"):
            self.db_path = convert_csv_to_sqlite(self.db_path)
        else:
            raise ValueError("expected either .sqlite or .csv file")

        # Get SQLSchema
        if schema_path:
            schemas, _ = load_tables([schema_path])
            if len(schemas) != 1:
                raise ValueError()
            self.schema: Dict = next(iter(schemas.values()))
        else:
            db_path_splits = self.db_path.split('/')
            db_id = db_path_splits[-1].replace('.sqlite', '').replace('.db', '')
            self.schema: Dict = dump_db_json_schema(self.db_path, db_id)
            schema_json_file = os.path.join('/'.join(db_path_splits[:-1]), 'tables.json')
            if not os.path.exists(schema_json_file):  # Vu Hoang: write to JSON schema file if not exists.
                with open(schema_json_file, "w") as f_json:
                    json.dump([self.schema], f_json, indent=4, sort_keys=True)
            self.schema: SpiderSchema = schema_dict_to_spider_schema(
                refine_schema_names(self.schema)
            )

        self.preprocessed_schema: SQLSchema = preprocess_schema_uncached(
            schema=self.schema,
            db_path=self.db_path,
            tokenize=self.duorat.preproc._schema_tokenize,
        )

    def infer_query(self, question, slml_question=None, history=None, beam_size=1, decode_max_time_step=500):
        return self.duorat.infer_query(question,
                                       spider_schema=self.schema,
                                       preprocessed_schema=self.preprocessed_schema,
                                       slml_question=slml_question,
                                       history=history,
                                       beam_size=beam_size,
                                       decode_max_time_step=decode_max_time_step)

    def execute(self, query):
        return execute(query=query, db_path=self.db_path)

# wikisql.py
# Vu Hoang @ Oracle Corp.

from typing import List, Optional

from duorat.datasets.spider import SpiderDataset
from duorat.utils import registry
from third_party.spider import evaluation


class WikiSQLEvaluator(evaluation.Evaluator):
    def __init__(self, db_path, schemas, kmaps, etype):
        self.kmaps = kmaps
        self.etype = etype

        self.db_paths = {}
        self.schemas = {}
        for db_name in self.kmaps.keys():
            self.db_paths[db_name] = db_path
            self.schemas[db_name] = schemas[db_name]

        self.scores = {
            level: {
                "count": 0,
                "partial": {
                    type_: {
                        "acc": 0.0,
                        "rec": 0.0,
                        "f1": 0.0,
                        "acc_count": 0,
                        "rec_count": 0,
                    }
                    for type_ in evaluation.PARTIAL_TYPES
                },
                "exact": 0.0,
                "exec": 0,
            }
            for level in evaluation.LEVELS
        }


@registry.register("dataset", "wikisql")
class WikiSQLDataset(SpiderDataset):
    def __init__(self,
                 examples_files: List[str],
                 tables_files: List[str],
                 db_path: str):
        super().__init__(paths=examples_files,
                         tables_paths=tables_files,
                         db_path=db_path)

    def get_db_path(self, db_id: Optional[str] = ''):
        # In the WikiSQL dataset, all tables are stored in one DB file which is self.db_path.
        return self.db_path

    class Metrics(SpiderDataset.Metrics):
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = WikiSQLEvaluator(
                self.dataset.db_path, self.dataset.original_schemas, self.foreign_key_maps, "match"
            )
            self.results = []

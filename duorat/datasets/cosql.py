# cosql.py
# Vu Hoang @ Oracle Corp.

from typing import List, Optional

from duorat.datasets.sparc import SparcDataset
from duorat.utils import registry


@registry.register("dataset", "cosql")
class CoSQLDataset(SparcDataset):
    def __init__(self,
                 examples_files: List[str],
                 tables_files: List[str],
                 db_path: str,
                 ignore_patterns: Optional[str] = ""):
        super().__init__(examples_files=examples_files,
                         tables_files=tables_files,
                         db_path=db_path,
                         ignore_patterns=ignore_patterns)

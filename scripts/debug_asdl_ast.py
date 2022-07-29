from third_party.spider.process_sql import get_sql

from duorat.utils import registry
from duorat.datasets.spider import load_original_schemas
from duorat.asdl.lang.spider.spider_transition_system import (
    SpiderTransitionSystem,
    asdl_ast_to_dict
)


if __name__ == "__main__":
    print('Loading schemas and transition system...')
    original_schemas = load_original_schemas(['./data/spider/tables.json'])
    transition_system = {
        'name': 'SpiderTransitionSystem',
        'asdl_grammar_path': 'duorat/asdl/lang/spider/spider_asdl.txt',
        'tokenizer': {
            'name': 'BERTTokenizer',
            'pretrained_model_name_or_path': 'bert-large-uncased-whole-word-masking',
        },
        'output_from': True,
        'use_table_pointer': True,
        'include_literals': True,
        'include_columns': True,
    }
    transition_system: SpiderTransitionSystem = registry.construct(
        kind="transition_system", config=transition_system
    )
    print(transition_system)

    print("Done! Entering a loop: ")
    while True:
        db_id = input("db_id: ")
        print(original_schemas[db_id].schema)
        print(original_schemas[db_id].idMap)

        raw_sql = input("query: ")
        spider_sql = get_sql(original_schemas[db_id], raw_sql)
        print(spider_sql)
        asdl_ast = transition_system.surface_code_to_ast(code=spider_sql)
        print(asdl_ast)
        print(asdl_ast.pretty(string_buffer=None))
        asdl_ast_in_json = asdl_ast_to_dict(asdl_ast=asdl_ast, grammar=transition_system.grammar)
        print(asdl_ast_in_json)

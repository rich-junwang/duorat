(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            transition_system+: {
                asdl_grammar_path: '/mnt/shared/vchoang/works/projects/oda/text2sql/code/duorat/duorat/asdl/lang/spider/spider_asdl.txt',
            },
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            }
        }
    },
}

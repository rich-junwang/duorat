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
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            }
        }
    },

    train+: {
        data_seed: 1,
        model_seed: 1,
        init_seed: 1,
    },
}

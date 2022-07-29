(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX_SPIDER = './data/',

    data: {
        name: 'Spider',
        train: (import '../../data/train_with_schema_custom_ner_silver_data.libsonnet')(prefix=PREFIX_SPIDER),
        val: (import '../../data/val_with_schema_custom_ner_silver_data.libsonnet')(prefix=PREFIX_SPIDER),
        type: 'original'
    },

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
}

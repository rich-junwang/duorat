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
        initialize_from: {
            pretrained_model_path: './logdir/duorat-spider-new-db-content-synthetic-data-template-scfg-100s',
            model_weight_filters: ['initial_encoder']
        },
        pin_memory: true,
        num_workers: 4,
    }
}

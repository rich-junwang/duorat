(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = './data/cosql/sql_state_tracking',

    data: {
        train: (import '../../data/train_cosql.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val_cosql.libsonnet')(prefix=PREFIX),
        train_sample_size: 1000,
        val_sample_size: 1007,
    },

    model+: {
        encoder+: {
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            },
            interaction_size: 1,
            max_source_length: 200,
        },
        preproc+: {
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            interaction_type: 'source',
        }
    },

    train+: {
        data_seed: 1,
        model_seed: 1,
        init_seed: 1,
        other_seed: 1,
        batch_size: 2,
        eval_every_n: 200,
        infer_min_n: 200,
        deterministic: true,
        num_eval_items: 1007,
    },
}

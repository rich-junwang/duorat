(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX_SPIDER = './data/',
    local PREFIX_SPARC = './data/sparc',
    local PREFIX_COSQL = './data/cosql/sql_state_tracking',

    data: [
        {
            name: 'Spider',
            train: (import '../../data/train.libsonnet')(prefix=PREFIX_SPIDER),
            val: (import '../../data/val.libsonnet')(prefix=PREFIX_SPIDER),
        },
        {
            name: 'Sparc',
            train: (import '../../data/train_sparc.libsonnet')(prefix=PREFIX_SPARC),
            val: (import '../../data/val_sparc.libsonnet')(prefix=PREFIX_SPARC),
        },
        {
            name: 'CoSQL',
            train: (import '../../data/train_cosql.libsonnet')(prefix=PREFIX_COSQL),
            val: (import '../../data/val_cosql.libsonnet')(prefix=PREFIX_COSQL),
        }
    ],

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
            interaction_type: 'target',
        }
    },

    train+: {
        num_eval_items: 3242,
    }
}

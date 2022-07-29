(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX_SPIDER = './data/',
    local PREFIX_SPARC = './data/sparc',
    local PREFIX_COSQL = './data/cosql/sql_state_tracking',

    data: [
        {
            name: 'Spider',
            train: (import '../../data/train.libsonnet')(prefix=PREFIX_SPIDER),
            val: (import '../../data/val.libsonnet')(prefix=PREFIX_SPIDER),
            type: 'original'
        },
        {
            name: 'Spider_Synthetic_TemplateSCFG',
            train: (import '../../data/train_spider_synthetic_data_template_scfg_v3_fixed_val_db_only_100shot.libsonnet')(prefix=PREFIX_SPIDER),
            type: 'synthetic'
        },
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
            interaction_type: 'source',
        }
    },

    train+: {
        num_eval_items: 1034,
        batch_balancing: false,
        pin_memory: true,
        num_workers: 4,
    }
}

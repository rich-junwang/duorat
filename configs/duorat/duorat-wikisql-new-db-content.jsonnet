(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = './data/wikisql',

    data: {
        train: (import '../../data/train_wikisql.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val_wikisql.libsonnet')(prefix=PREFIX),
        test: (import '../../data/test_wikisql.libsonnet')(prefix=PREFIX),
    },

    model+: {
        encoder+: {
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            },
            schema_input_token_ordering: '[column]',
            schema_source_token_ordering: '[column]',
            max_source_length: 200,
        },
        preproc+: {
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            }
        }
    },

    train+: {
        batch_size: 32,
        n_grad_accumulation_steps: 8,
        eval_batch_size: 32,
        num_eval_items: 8421,
    },
}

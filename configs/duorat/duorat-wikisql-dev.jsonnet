(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = './data/wikisql',

    data: {
        train: (import '../../data/train_wikisql.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val_wikisql.libsonnet')(prefix=PREFIX),
        test: (import '../../data/test_wikisql.libsonnet')(prefix=PREFIX),
        train_sample_size: 500,
        val_sample_size: 100,
        test_sample_size: 100
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
        data_seed: 1,
        model_seed: 1,
        init_seed: 1,
        other_seed: 1,
        batch_size: 16,
        n_grad_accumulation_steps: 8,
        eval_every_n: 200,
        infer_min_n: 200,
        deterministic: true,
        num_eval_items: 100,
    },
}

(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = 'data/',
    data: {
        train: (import '../../data/train.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val.libsonnet')(prefix=PREFIX),
        train_sample_size: 100,
        val_sample_size: 100,
    },

    model+: {
        encoder+: {
            initial_encoder+: {
                pretrained_model_name_or_path: 'google/electra-base-discriminator',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            },
            schema_input_token_ordering: '[table column]',
            schema_source_token_ordering: '[table column]',
        },
        preproc+: {
            use_column_type: true,
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                pretrained_model_name_or_path: 'google/electra-base-discriminator',
            },
            transition_system+: {
                tokenizer+: {
                    pretrained_model_name_or_path: 'google/electra-base-discriminator',
                }
            }
        }
    },

    lr_scheduler+: {
        decay_steps: 148000,
    },

    train+: {
        batch_size: 1,
        n_grad_accumulation_steps: 1,
        max_steps: 150000,
        "eval_batch_size": 1,
        "eval_every_n": 50,
        "eval_nproc": 1,
        "eval_on_train": false,
        "eval_on_val": true,
        "infer_min_n": 50,
        "num_eval_items": 100,
        "report_every_n": 1,
        "debug": true,
    }
}

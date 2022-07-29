(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = './data/',
    data: {
        train: (import '../../data/train.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val.libsonnet')(prefix=PREFIX),
        train_sample_size: 100,
        val_sample_size: 100,
    },

    model+: {
        encoder+: {
            initial_encoder+: {
                name: 'T5Encoder',
                pretrained_model_name_or_path: 't5-small',
                use_outputs_from: 'enc',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            },
            "schema_input_token_ordering": "[table][column]",
            "schema_source_token_ordering": "[table][column]",
        },
        preproc+: {
            add_cls_token: false,  # T5 does not use CLS token.
            add_sep_token: true,
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                name: 'T5Tokenizer',
                pretrained_model_name_or_path: 't5-small',
                sep_token: '</s>',  # We replace cls_token with eos_token in T5.
            },
            transition_system+: {
                tokenizer+: {
                    name: 'T5Tokenizer',
                    pretrained_model_name_or_path: 't5-small',
                    sep_token: '</s>',  # We replace cls_token with eos_token in T5.
                }
            }
        }
    },

    train+: {
        "batch_size": 8,
        "n_grad_accumulation_steps": 6,
        "eval_batch_size": 20,
        "eval_every_n": 5000,
        "eval_on_val": true,
        "infer_min_n": 5000,
        "num_eval_items": 1034,
        "report_every_n": 10
    }
}

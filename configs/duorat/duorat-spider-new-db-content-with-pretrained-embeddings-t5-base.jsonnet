(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                name: 'T5Encoder',
                pretrained_model_name_or_path: 't5-base',
                use_outputs_from: 'enc',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            add_cls_token: true,  # T5 does not use CLS token.
            add_sep_token: false,
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                name: 'T5Tokenizer',
                pretrained_model_name_or_path: 't5-base',
                cls_token: '</s>'  # We replace cls_token with eos_token in T5.
            },
            transition_system+: {
                tokenizer+: {
                    name: 'T5Tokenizer',
                    pretrained_model_name_or_path: 't5-base',
                    cls_token: '</s>'  # We replace cls_token with eos_token in T5.
                }
            }
        }
    },

    train+: {
        "batch_size": 8,
        "n_grad_accumulation_steps": 6,
    }
}

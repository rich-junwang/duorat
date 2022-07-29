(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                name: 'BartEncoder',
                pretrained_model_name_or_path: 'facebook/bart-base',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            add_cls_token: false,
            add_sep_token: true,

            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                name: 'RoBERTaTokenizer',
                pretrained_model_name_or_path: 'facebook/bart-base',
            },
            transition_system+: {
                tokenizer+: {
                    name: 'RoBERTaTokenizer',
                    pretrained_model_name_or_path: 'facebook/bart-base',
                }
            }
        }
    },

    lr_scheduler+: {
        decay_steps: 148000,
    },

    train+: {
        batch_size: 8,
        n_grad_accumulation_steps: 6,
        max_steps: 150000,
    }
}

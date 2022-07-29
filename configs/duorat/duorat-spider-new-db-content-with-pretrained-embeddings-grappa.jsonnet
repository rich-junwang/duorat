(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                pretrained_model_name_or_path: 'Salesforce/grappa_large_jnt',
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
            add_cls_token: false,
            add_sep_token: true,
            use_column_type: false,
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                name: 'RoBERTaTokenizer',
                pretrained_model_name_or_path: 'Salesforce/grappa_large_jnt',
            },
            transition_system+: {
                tokenizer+: {
                    name: 'RoBERTaTokenizer',
                    pretrained_model_name_or_path: 'Salesforce/grappa_large_jnt',
                }
            }
        }
    },

    train+: {
        "batch_size": 4,
        "n_grad_accumulation_steps": 7,
    }
}

(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                pretrained_model_name_or_path: 'roberta-large',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                name: 'RoBERTaTokenizer',
                pretrained_model_name_or_path: 'roberta-large',
            },
            transition_system+: {
                tokenizer+: {
                    name: 'RoBERTaTokenizer',
                    pretrained_model_name_or_path: 'roberta-large',
                }
            }
        }
    },

    train+: {
        "batch_size": 4,
        "n_grad_accumulation_steps": 7,
    }
}

(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                name: 'DistilBert',
                pretrained_model_name_or_path: 'distilbert-base-uncased',
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
                pretrained_model_name_or_path: 'distilbert-base-uncased',
            },
            transition_system+: {
                tokenizer+: {
                    pretrained_model_name_or_path: 'distilbert-base-uncased',
                }
            }
        }
    },

    train+: {
        "batch_size": 6,
        "n_grad_accumulation_steps": 8,
    }
}

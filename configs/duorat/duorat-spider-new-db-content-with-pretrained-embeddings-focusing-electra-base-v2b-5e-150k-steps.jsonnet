(import 'duorat-finetune-bert-large.jsonnet') {
    model+: {
        encoder+: {
            initial_encoder+: {
                pretrained_model_name_or_path: './pretrained_models/focusing/focusing_electra_base_v2b_5epoch',
            },
            source_relation_types: {
                use_schema_linking: true,
                high_confidence_db_content_schema_linking: true,
                low_confidence_db_content_schema_linking: true,
            }
        },
        preproc+: {
            use_column_type: false,
            schema_linker+: {
                whole_entry_db_content_confidence: 'high',
                partial_entry_db_content_confidence: 'low'
            },
            tokenizer+: {
                pretrained_model_name_or_path: './pretrained_models/focusing/focusing_electra_base_v2b_5epoch',
            },
            transition_system+: {
                tokenizer+: {
                    pretrained_model_name_or_path: './pretrained_models/focusing/focusing_electra_base_v2b_5epoch',
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

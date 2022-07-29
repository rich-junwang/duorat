(import 'duorat-finetune-bert-large.jsonnet') {
    local PREFIX = 'data/',
    data: {
        name: 'Spider',
        train: (import '../../data/train_plus_dev25.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val_dev55.libsonnet')(prefix=PREFIX),
        type: 'original'
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
            }
        },
        preproc+: {
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
        batch_size: 8,
        n_grad_accumulation_steps: 6,
        max_steps: 150000,
    }
}

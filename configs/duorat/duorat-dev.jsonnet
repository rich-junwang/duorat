(import 'duorat-base.libsonnet')(output_from=true) {
    local PREFIX = 'data/',

    data: {
        train: (import '../../data/train.libsonnet')(prefix=PREFIX),
        val: (import '../../data/val.libsonnet')(prefix=PREFIX),
        train_sample_size: 500,
        val_sample_size: 100
    },

    model+: {
        name: 'LMDuoRAT',
        encoder: {
            initial_encoder: {
                name: 'Bert',
                pretrained_model_name_or_path: 'bert-large-uncased-whole-word-masking',
                trainable: true,
                num_return_layers: 1,
                embed_dim: 256,
                use_dedicated_gpu: false,
                use_affine_transformation: false,
                use_attention_mask: false,
                use_token_type_ids: false,
                use_position_ids: false,
                use_segments: false,
                use_outputs_from: 'enc',
            },
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 1024,
            "rat_num_heads": 8,
            "rat_num_layers": 8,
            "rat_relu_dropout": 0.1,
            source_relation_types: {
                use_schema_linking: true,
            },
            schema_input_token_ordering: '[column][table]',
            schema_source_token_ordering: '[column][table]',
            max_source_length: 200,
        },
        decoder: {
            "action_embed_dim": 64,
            "field_embed_dim": 64,
            "type_embed_dim": 64,
            "p_mask": 0.2,
            "rat_attention_dropout": 0.1,
            "rat_dropout": 0.1,
            "rat_ffn_dim": 256,
            "rat_num_heads": 8,
            "rat_num_layers": 2,
            "rat_relu_dropout": 0.1,
            pointer: {
                name: 'BahdanauMemEfficient',
                proj_size: 50,
            },
        },
        preproc+: {
            name: 'BertDuoRAT',
            add_cls_token: true,
            add_sep_token: false,

            min_freq: 5,
            max_count: 5000,

            tokenizer: {
                name: 'BERTTokenizer',
                pretrained_model_name_or_path: 'bert-large-uncased-whole-word-masking',
            },
            transition_system+: {
                tokenizer: {
                    name: 'BERTTokenizer',
                    pretrained_model_name_or_path: 'bert-large-uncased-whole-word-masking',
                }
            }
        },
    },
    train+: {
        batch_size: 2
    },
}
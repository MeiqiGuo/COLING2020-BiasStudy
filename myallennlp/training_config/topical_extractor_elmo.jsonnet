{
    "dataset_reader": {
        "type": "tweet_reader",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "iterator": {
        "type": "bucket",
        "batch_size": 64,
        "sorting_keys": [
            [
                "tokens",
                "num_tokens"
            ]
        ]
    },
    "model": {
        "type": "topical_extractor",
        "dropout": 0.5,
        "encoder": {
            "type": "lstm",
            "bidirectional": true,
            "hidden_size": 300,
            "input_size": 1024,
            "num_layers": 1
        },
        "initializer": [
            [
                ".*linear_layers.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*linear_layers.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ]
        ],
        "output_feedforward": {
            "activations": "relu",
            "dropout": 0.5,
            "hidden_dims": 300,
            "input_dim": 600,
            "num_layers": 1
        },
        "output_logit": {
            "activations": "linear",
            "hidden_dims": 2,
            "input_dim": 300,
            "num_layers": 1
        },
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "do_layer_norm": false,
                    "dropout": 0.2,
                    "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
                }
            }
        }
    },
    "train_data_path": TRAIN_FILE,
    "validation_data_path": DEV_FILE,
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 10,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "num_epochs": 75,
        "num_serialized_models_to_keep": 2,
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "patience": 10,
        "validation_metric": "+fscore"
    }
}
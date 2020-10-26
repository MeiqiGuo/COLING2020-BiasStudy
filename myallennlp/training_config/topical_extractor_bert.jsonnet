{
    "dataset_reader": {
        "lazy": false,
        "type": "tweet_reader",
        "tokenizer": {
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model": "bert-base-uncased",
              "max_pieces": 128
            },
        }
    },
    "train_data_path": TRAIN_FILE,
    "validation_data_path": DEV_FILE,
    "iterator": {
        "type": "basic",
        "batch_size": 32
    },
    "model": {
        "type": "topical_extractor_bert",
        "bert_model": "bert-base-uncased",
        "dropout": 0.0
    },
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 10,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        },
        "num_epochs": 5,
        "num_serialized_models_to_keep": 2,
        "optimizer": {
            "type": "adam",
            "lr": 0.00003
        },
        "patience": 3,
        "validation_metric": "+fscore"
    }
}
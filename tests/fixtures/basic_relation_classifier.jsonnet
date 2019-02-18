{
    "dataset_reader": {
        "type": "semeval2010_task8",
        "max_len": 50,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true,
            },
        },
    },
    "train_data_path": "tests/fixtures/semeval2010_task8.jsonl",
    "validation_data_path": "tests/fixtures/semeval2010_task8.jsonl",
    "model": {
        "type": "basic_relation_classifier",
        "f1_average": "macro",
        "verbose_metrics": false,
        "embedding_dropout": 0.1,
        "word_dropout": 0.1,
        "encoding_dropout": 0.1,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 2,
                "trainable": false,
            },
        },
        "offset_embedder_head": {
            "type": "relative",
            "n_position": 50,
            "embedding_dim": 2,
        },
        "offset_embedder_tail": {
            "type": "relative",
            "n_position": 50,
            "embedding_dim": 2,
        },
        "text_encoder": {
            "type": "cnn",
            "embedding_dim": 6,
            "num_filters": 2,
            "ngram_filter_sizes": [2],
        },
        "classifier_feedforward": {
            "input_dim": 2,
            "num_layers": 1,
            "hidden_dims": [7],
            "activations": ["linear"],
            "dropout": [0.0],
        },
        "regularizer": [
            ["text_encoder.conv_layer_.*weight", {"type": "l2", "alpha": 1e-5}],
        ],
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["text", "num_tokens"]],
        "padding_noise": 0,
        "batch_size": 20,
    },
    "trainer": {
        "num_epochs": 1,
        "cuda_device": -1,
        "grad_clipping": 5.0,
        "num_serialized_models_to_keep": 1,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adadelta",
        },
    }
}
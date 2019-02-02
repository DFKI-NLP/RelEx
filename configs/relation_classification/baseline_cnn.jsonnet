{
  "dataset_reader": {
    "type": "semeval2010_task8",
    "max_len": 200,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "offset_head": {
        "type": "offset",
        "token_attribute": "offset_head"
      },
      "offset_tail": {
        "type": "offset",
        "token_attribute": "offset_tail"
      },
    },
  },
  
  "train_data_path": "../rexplained/data/semeval_2010_task_8/train.jsonl",
  "validation_data_path": "../rexplained/data/semeval_2010_task_8/test.jsonl",

  "model": {
    "type": "basic_relation_classifier",
    "verbose_metrics": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
        "embedding_dim": 50,
        "trainable": false
      },
      "offset_head": {
        "type": "embedding",
        "embedding_dim": 25,
        "trainable": true
      },
      "offset_tail": {
        "type": "embedding",
        "embedding_dim": 25,
        "trainable": true
      },
    },
    "text_encoder": {
      "type": "cnn",
      "embedding_dim": 100,
      "num_filters": 50,
      "ngram_filter_sizes": [2,3,4,5]
    },
    "classifier_feedforward": {
      "input_dim": 200,
      "num_layers": 2,
      "hidden_dims": [200, 19],
      "activations": ["relu", "linear"],
      "dropout": [0.5, 0.0]
    },
  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 16
  },

  "trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    // "grad_clipping": 5.0,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adam",
      "lr": 1e-3
    }
  }
}

function (embedding_dim = 300, use_embedding_projection = true, embedding_projection_dim = 1024,
          use_offset_embeddings = true, offset_type = "sine", offset_embedding_dim = 300, freeze_offset_embeddings = true,
          max_len = 200) {
  
  local unprojected_embedding_dim = embedding_dim + (if use_offset_embeddings && (offset_type != "sine") then 2 * offset_embedding_dim else 0),
  local text_encoder_input_dim = if use_embedding_projection then embedding_projection_dim else unprojected_embedding_dim,

  "dataset_reader": {
    "type": "semeval2010_task8",
    "max_len": max_len,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  
  "train_data_path": "../relex-data/semeval_2010_task_8/train.jsonl",
  "validation_data_path": "../relex-data/semeval_2010_task_8/dev.jsonl",

  "model": {
    "type": "basic_relation_classifier",
    [if use_embedding_projection then "embedding_projection_dim"]: embedding_projection_dim,
    "verbose_metrics": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
        "embedding_dim": embedding_dim,
        "trainable": false,
      },
    },
    [if use_offset_embeddings then "offset_embedder_head"]: {
      "type": offset_type,
      "n_position": max_len,
      "embedding_dim": offset_embedding_dim,
    },
    [if use_offset_embeddings then "offset_embedder_tail"]: {
      "type": offset_type,
      "n_position": max_len,
      "embedding_dim": offset_embedding_dim,
    },
    "text_encoder": {
      "type": "cnn",
      "embedding_dim": text_encoder_input_dim,
      "num_filters": 1024,
      "ngram_filter_sizes": [2,3,4,5],
    },
    "classifier_feedforward": {
      "input_dim": 4096,
      "num_layers": 1,
      "hidden_dims": [19],
      "activations": ["linear"],
      "dropout": [0.0],
    },
    "initializer": [
      ["text_encoder.conv_layer_.*.weight.*", "kaiming_uniform"],
    ],
  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 64,
  },

  "trainer": {
    "num_epochs": 200,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    // "grad_clipping": 5.0,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adam",
      "lr": 5e-4,
    },
    "no_grad": [
      "text_encoder.*",
    ] + (if freeze_offset_embeddings then ["offset_embedder.*"] else []) 
      + (if use_embedding_projection then [".*embedding_projection"] else []),
  }
}

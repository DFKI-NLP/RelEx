function (
  lr = 1, num_epochs = 50,
  embedding_dim = 300, embedding_trainable = true, embedding_dropout = 0.5,
  ner_embedding_dim = null, pos_embedding_dim = null,
  offset_type = "relative", offset_embedding_dim = 50,
  text_encoder_num_filters = 150, text_encoder_ngram_filter_sizes = [2, 3, 4, 5], text_encoder_dropout=0.5,
  dataset = "semeval2010_task8",
  train_data_path = "../relex-data/semeval_2010_task_8/train.jsonl",
  validation_data_path = "../relex-data/semeval_2010_task_8/dev.jsonl",
  max_len = 90) {
  
  local use_offset_embeddings = (offset_embedding_dim != null),
  local use_ner_embeddings = (ner_embedding_dim != null),
  local use_pos_embeddings = (pos_embedding_dim != null),

  local text_encoder_input_dim = embedding_dim  
                                 + (if use_offset_embeddings then 2 * offset_embedding_dim else 0) 
                                 + (if use_ner_embeddings then ner_embedding_dim else 0)
                                 + (if use_pos_embeddings then pos_embedding_dim else 0),

  local classifier_feedforward_input_dim = text_encoder_num_filters * std.length(text_encoder_ngram_filter_sizes),

  local num_classes = if (dataset == "semeval2010_task8") then 19 else 42,

  "dataset_reader": {
    "type": dataset,
    "max_len": max_len,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
      [if use_ner_embeddings then "ner_tokens"]: {
        "type": "ner_tag"
      },
      [if use_pos_embeddings then "pos_tokens"]: {
        "type": "pos_tag"
      },
    },
  },
  
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,

  "model": {
    "type": "basic_relation_classifier",
    "f1_average": "macro",
    "verbose_metrics": false,
    "embedding_dropout": embedding_dropout,
    "encoding_dropout": text_encoder_dropout,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "embedding_dim": embedding_dim,
        "trainable": embedding_trainable,
      },
      [if use_ner_embeddings then "ner_tokens"]: {
        "type": "embedding",
        "embedding_dim": ner_embedding_dim,
        "trainable": true
      },
      [if use_pos_embeddings then "pos_tokens"]: {
        "type": "embedding",
        "embedding_dim": pos_embedding_dim,
        "trainable": true
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
      "num_filters": text_encoder_num_filters,
      "ngram_filter_sizes": text_encoder_ngram_filter_sizes,
    },
    "classifier_feedforward": {
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1,
      "hidden_dims": [num_classes],
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
    "batch_size": 20,
  },

  "vocabulary": {
    "min_count": {
      "tokens": 2,
    },
  },

  "trainer": {
    "num_epochs": num_epochs,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adadelta",
      "rho": 0.9,
      "eps": 1e-6,
      "lr": lr,
    },
  }
}

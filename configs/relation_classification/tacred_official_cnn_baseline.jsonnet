function (
  lr = 0.1, num_epochs = 200,
  embedding_dim = 300,
  offset_type = "relative", offset_embedding_dim = 50,
  text_encoder_num_filters = 100, text_encoder_ngram_filter_sizes = [2, 3, 4, 5],
  max_len = 200) {
  
  local use_offset_embeddings = (offset_embedding_dim != null),

  local combined_offset_embedding_dim = if use_offset_embeddings && (offset_type != "sine") then 2 * offset_embedding_dim else 0,
  local text_encoder_input_dim = combined_offset_embedding_dim + embedding_dim,
  local classifier_feedforward_input_dim = text_encoder_num_filters * std.length(text_encoder_ngram_filter_sizes),

  "dataset_reader": {
    "type": "tacred",
    "max_len": max_len,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true,
      },
    },
  },
  
  "train_data_path": "data/tacred/train.json",
  "validation_data_path": "data/tacred/dev.json",

  "vocabulary": {
    min_count: 2
  }

  "model": {
    "type": "basic_relation_classifier",
    "verbose_metrics": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "data/embeddings/glove.840B.300d.txt",
        "embedding_dim": embedding_dim,
        "trainable": true,
      },
    },
    "offset_embedder_head": {
      "type": offset_type,
      "n_position": max_len,
      "embedding_dim": offset_embedding_dim,
    },
    "offset_embedder_tail": {
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
    "encoding_dropout": 0.5,
    "classifier_feedforward": {
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1,
      "hidden_dims": [42],
      "activations": ["linear"],
      "dropout": [0.0],
    },
    "regularizer": [["text_encoder.conv_layer_*.weight.*", {"type": "l2", "alpha": 1e-3}]]
  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 50,
  },

  "trainer": {
    "num_epochs": num_epochs,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adagrad",
      "lr": lr,
    }
  }
}

function (
  lr = 1e-4, num_epochs = 100,
  word_dropout = 0.04,
  embedding_dim = 300, embedding_trainable = false, embedding_dropout = 0.5,
  ner_embedding_dim = null, pos_embedding_dim = null, dep_embedding_dim = null,
  offset_type = "relative", offset_embedding_dim = 50,
  text_encoder_hidden_dim = 256, text_encoder_projection_dim = 256, text_encoder_feedforward_hidden_dim = 512,
  text_encoder_num_layers = 16, text_encoder_num_heads = 4, text_encoder_dropout = 0.1, text_encoder_resid_dropout = 0.2,
  textn_encoder_attn_dropout = 0.1, text_encoder_pooling = "final",
  dataset = "semeval2010_task8",
  train_data_path = "../relex-data/semeval_2010_task_8_annotated/train.json",
  validation_data_path = "../relex-data/semeval_2010_task_8_annotated/dev.json",
  max_len = 100, run=1) {
  
  local use_offset_embeddings = (offset_embedding_dim != null),
  local use_ner_embeddings = (ner_embedding_dim != null),
  local use_pos_embeddings = (pos_embedding_dim != null),
  local use_dep_embeddings = (dep_embedding_dim != null),

  local text_encoder_input_dim = embedding_dim  
                                 + (if use_offset_embeddings then 2 * offset_embedding_dim else 0) 
                                 + (if use_ner_embeddings then ner_embedding_dim else 0)
                                 + (if use_pos_embeddings then pos_embedding_dim else 0)
                                 + (if use_dep_embeddings then dep_embedding_dim else 0),

  local classifier_feedforward_input_dim = text_encoder_hidden_dim,

  local num_classes = if (dataset == "semeval2010_task8") then 19 else 42,

  "random_seed": 13370 * run,
  "numpy_seed": 1337 * run,
  "pytorch_seed": 133 * run,

  "dataset_reader": {
    "type": "tacred", // dataset,
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
      [if use_dep_embeddings then "dep_labels"]: {
        "type": "dependency_label"
      },
    },
  },
  
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,

  "model": {
    "type": "basic_relation_classifier",
    "f1_average": "micro",
    "ignore_label": "no_relation",
    "verbose_metrics": false,
    "word_dropout": word_dropout,
    "embedding_dropout": embedding_dropout,
    "encoding_dropout": 0.5,
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
      [if use_dep_embeddings then "dep_labels"]: {
        "type": "embedding",
        "embedding_dim": dep_embedding_dim,
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
      "type": "seq2seq_pool",
      "encoder": {
        "type": "stacked_self_attention",
        "input_dim": text_encoder_input_dim,
        "hidden_dim": text_encoder_hidden_dim,
        "projection_dim": text_encoder_projection_dim,
        "feedforward_hidden_dim": text_encoder_feedforward_hidden_dim,
        "num_layers": text_encoder_num_layers,
        "num_attention_heads": text_encoder_num_heads,
        "use_positional_encoding": (use_offset_embeddings == false),
        "dropout_prob": text_encoder_dropout,
        "residual_dropout_prob": text_encoder_resid_dropout,
        "attention_dropout_prob": textn_encoder_attn_dropout,
      },
      "pooling": text_encoder_pooling,
    },
    "classifier_feedforward": {
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1,
      "hidden_dims": [num_classes],
      "activations": ["linear"],
      "dropout": [0.0],
    },
    // "regularizer": [
    //   ["text_encoder.*weight", {"type": "l2", "alpha": 1e-3}],
    // ],
  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 50,
  },

  "vocabulary": {
    "min_count": {
      "tokens": 2,
    },
  },

  "trainer": {
    "num_epochs": num_epochs,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      // "type": "sgd",
      "type": "adam",
      "lr": lr,
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.9,
      "mode": "max",
      "patience": 1
    },
  }
}

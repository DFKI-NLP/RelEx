function (
  lr = 1e-3, num_epochs = 200,
  embedding_dim = 300,
  offset_type = "relative", offset_embedding_dim = 10,
  ner_embedding_dim = 16,
  text_encoder_attn_dropout = 0, text_encoder_num_heads = 8, text_encoder_attention_dim = 1024, text_encoder_values_dim = 512,
  text_encoder_output_projection_dim = 512, text_encoder_pooling = "max",
  max_len = 200) {
  
  local use_offset_embeddings = (offset_embedding_dim != null),
  local use_ner_embeddings = (ner_embedding_dim != null),

  local combined_offset_embedding_dim = if use_offset_embeddings && (offset_type != "sine") then 2 * offset_embedding_dim else 0,
  local text_encoder_input_dim = combined_offset_embedding_dim + embedding_dim + (if use_ner_embeddings then ner_embedding_dim else 0),
  local classifier_feedforward_input_dim = text_encoder_output_projection_dim,

  "dataset_reader": {
    "type": "semeval2010_task8",
    "max_len": max_len,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      [if use_ner_embeddings then "ner_tokens"]: {
        "type": "ner_tag"
      },
    },
  },
  
  "train_data_path": "../relex-data/semeval_2010_task_8/train.jsonl",
  "validation_data_path": "../relex-data/semeval_2010_task_8/dev.jsonl",

  "model": {
    "type": "basic_relation_classifier",
    "verbose_metrics": false,
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
        "embedding_dim": embedding_dim,
        "trainable": false
      },
      [if use_ner_embeddings then "ner_tokens"]: {
        "type": "embedding",
        "embedding_dim": ner_embedding_dim,
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
        "type": "multi_head_self_attention",
        "input_dim": text_encoder_input_dim,
        "num_heads": text_encoder_num_heads,
        "attention_dim": text_encoder_attention_dim,
        "values_dim": text_encoder_values_dim,
        "output_projection_dim": text_encoder_output_projection_dim,
        "attention_dropout_prob": text_encoder_attn_dropout,
      },
      "pooling": text_encoder_pooling,
    },
    "classifier_feedforward": {
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1,
      "hidden_dims": [19],
      "activations": ["linear"],
      "dropout": [0.0],
    },
    // "initializer": [
    //   ["text_encoder._encoder._module.weight.*", "kaiming_uniform"],
    // ],
  },

  "iterator": {
    "type": "bucket",
    "sorting_keys": [["text", "num_tokens"]],
    "batch_size": 32
  },

  "trainer": {
    "num_epochs": num_epochs,
    "patience": 10,
    "cuda_device": 0,
    "num_serialized_models_to_keep": 1,
    // "grad_clipping": 5.0,
    "validation_metric": "+f1-measure-overall",
    "optimizer": {
      "type": "adam",
      "lr": lr
    },
  }
}

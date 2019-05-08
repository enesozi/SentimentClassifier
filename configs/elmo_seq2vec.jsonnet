local batch_size = 32;
local num_epochs = 29;
local patience = 10;
local cuda_device = 0;
local min_tokens = 3;
local do_layer_norm = false;
local dropout = 0.5;

// seq2vec_encoder parameters
local seq_out_dim = 64;
local seq_hidden_dim = 64;
local seq_num_layers = 1;
local seq_proj_dim = 64;
local seq_feed_dim = 64;
local seq_num_atten = 2;
local dropout_prob = 0.1;
local residual_dropout_prob = 0.2;
local attention_dropout_prob = 0.1;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "tokens": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "data/trees/train.txt",
  "validation_data_path": "data/trees/dev.txt",
  "vocabulary": {
        "min_count": {
            "tokens": min_tokens
        }
  },
  "model": {
    "type": "lstm_classifier",

    "word_embeddings": {
      "tokens": {
        "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
        "do_layer_norm": do_layer_norm,
        "dropout": dropout
      }
    },

    "encoder": {
      "type": "seq2vec_encoder",
      "out_dim": seq_out_dim,
      "hidden_dim": seq_hidden_dim,
      "num_layers": seq_num_layers,
      "projection_dim": seq_proj_dim, 
      "feedforward_hidden_dim": seq_feed_dim, 
      "num_attention_heads": seq_num_atten,
      "dropout_prob": dropout_prob,
      "residual_dropout_prob": residual_dropout_prob,
      "attention_dropout_prob": attention_dropout_prob
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": batch_size,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
            "type": "adam"
    },
    "num_epochs": num_epochs,
    "patience": patience,
    "cuda_device": cuda_device
  }
}

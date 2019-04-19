local embedding_dim = 768;
local hidden_dim = 128;
local batch_size = 32;
local num_epochs = 20;
local patience = 5;
local cuda_device = 0;
local min_tokens = 3;
local allow_unmatched_keys = true;

{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
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
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
      },
      "allow_unmatched_keys": allow_unmatched_keys
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "batch_first": "True"
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

// AllenNLP configuration files are written in Jsonnet, a superset of JSON
// with added functionalities such as variables and comments (like this one :).

// For example, you can write variable declrations as follows:
local embedding_dim = 128;
local hidden_dim = 128;
local learning_rate = 1e-4;
local weight_decay = 1e-5;

{
  "dataset_reader": {
    "type": "sst_tokens"
  },
  "train_data_path": "data/trees/train.txt",
  "validation_data_path": "data/trees/dev.txt",
  "vocabulary": {
        "min_count": {
            "tokens": 3
        }
  },
  // In order to use a model in configuration, it must
  //   1) inherit from the Registrable base class, and
  //   2) be decorated by @Model.register("model_name").
  // Also, the class has to be discoverable by the "allennlp" command
  // by specifying '--include-package [import path]'.
  "model": {
    "type": "lstm_classifier",

    // Other keys in the JSON dict correspond to the parameters passed to the constructor.

    // What's going on here -
    // The `word_embeddings` parameter takes an instance of TextFieldEmbedder.
    // In the Python code, you instantiated a BasicTextFieldEmbedder and passed it to
    // `word_embeddings`. However, the default implementation of TextFieldEmbedder is
    // "basic", which is BasicTextFieldEmbedder.
    // That's why you can write parameters to BasicTextFieldEmbedder (dictionary from
    // field names to their embedder) directly here.
    "word_embeddings": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": embedding_dim
      }
    },

    // In Python code, you need to wrap encoders (e.g., torch.nn.LSTM) by PytorchSeq2VecWrapper.
    // Conveniently, "wrapped" version of popular encoder types ("lstm", "gru", ...)
    // are already registered (see https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2vec_encoders/__init__.py)
    // so you can just use them by specifying intuitive names
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "batch_first": "True"
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": {
            "type": "adam",
            "lr": learning_rate,
            "weight_decay": weight_decay
    },
    "num_epochs": 20,
    "patience": 10,
    "cuda_device": 0
  }
}

// You can make predictions for instance(s) using the `allennlp`
// allennlp predict \
//  tmp/model.tar.gz \
//  =(echo '{"tokens": ["This", "is", "the", "best", "movie", "ever", "!"]}')  \
// --include-package examples.sentiment.sst_classifier \
// --predictor sentence_classifier_predictor
import torch
from allennlp.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from overrides import overrides


@Seq2VecEncoder.register("seq2vec_encoder")
class TransformerSeq2VecEncoder(Seq2VecEncoder):

    def __init__(self,
                 out_dim: int,
                 hidden_dim: int,
                 projection_dim: int,
                 feedforward_hidden_dim: int,
                 num_layers: int,
                 num_attention_heads: int,
                 dropout_prob: float,
                 residual_dropout_prob: float,
                 attention_dropout_prob: float) -> None:

        super(TransformerSeq2VecEncoder, self).__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.feedforward_hidden_dim = feedforward_hidden_dim
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.stacked_self_attention_encoder = StackedSelfAttentionEncoder(
            input_dim=256,
            hidden_dim=self.hidden_dim,
            projection_dim=self.projection_dim,
            feedforward_hidden_dim=self.feedforward_hidden_dim,
            num_layers=self.num_layers,
            num_attention_heads=self.num_attention_heads,
            dropout_prob=self.dropout_prob,
            residual_dropout_prob=self.residual_dropout_prob,
            attention_dropout_prob=self.attention_dropout_prob
        )

    @overrides
    def get_output_dim(self) -> int:
        return self.out_dim

    @overrides
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
        out = self.stacked_self_attention_encoder(inputs, mask)
        return out.sum(1)

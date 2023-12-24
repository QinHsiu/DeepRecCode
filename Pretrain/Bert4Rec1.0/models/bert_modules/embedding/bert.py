import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # vocab_size = num_items + 2  embed_size = 256
        # TokenEmbedding(3708, 256, padding_idx=0)
        # (num_item + 2, dim)
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        # max_len = 100  embed_size = 256
        # (batch size, len, dim)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # 推荐中不需要段嵌入
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        # dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 256
        self.embed_size = embed_size

    def forward(self, sequence):
        # 得到对应序列的embedding表示
        x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)
        # 通过一个dropout层
        return self.dropout(x)

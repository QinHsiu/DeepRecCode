import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # MultiHeadedAttention 多头attention
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        # d_model => 4 * d_model => d_model
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        # 输入sublayer
        # 层正则化 + dropout + 残差连接
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # 输出sublayer
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # 不是很理解 传入x和sublayer
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # d_model => 4 * d_model => d_model
        x = self.output_sublayer(x, self.feed_forward)
        # dropout
        return self.dropout(x)

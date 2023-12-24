import torch.nn as nn
from .single import Attention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        # embedding维数能除断head个数
        # 256 / 4 = 64
        assert d_model % h == 0

        # We assume d_v always equals d_k
        # 每个head的维数
        self.d_k = d_model // h
        self.h = h

        # 三层线性层
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        # 输出线性层
        self.output_linear = nn.Linear(d_model, d_model)
        # attention
        self.attention = Attention()

        # dropout层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 获取当前batch size
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 线性映射
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # 应用attention
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # concate所有头的结果
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # 通过一个线性层
        return self.output_linear(x)

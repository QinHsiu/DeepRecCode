from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args, fixed=False):
        super().__init__()

        # seed为0
        fix_random_seed_as(args.model_init_seed)
        # self.init_weights()

        # 100
        max_len = args.bert_max_len
        num_items = args.num_items
        # 2
        n_layers = args.bert_num_blocks
        # 4
        heads = args.bert_num_heads
        vocab_size = num_items + 2
        # 256 隐含层神经元个数  embedding维数
        hidden = args.bert_hidden_units
        self.hidden = hidden
        # 0.1
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        # 得到原始embedding与位置embedding相加之后通过dropout的结果
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        # 多层transformer堆叠
        # n_layers = 2
        # 256 * 4
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        ######fixed model
        if fixed:
            for param in self.parameters():
                param.requires_grad=False

    # bert模型的forward函数
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        # 得到当前序列的embedding表示 (加上位置编码)
        x = self.embedding(x)
        layer_output = []
        layer_output.append(x)
        # running over multiple transformer blocks
        # 叠加transformer模块

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
            layer_output.append(x)

        return x, layer_output

    def init_weights(self):
        pass

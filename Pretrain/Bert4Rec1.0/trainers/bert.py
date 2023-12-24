from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks

import torch.nn as nn
import numpy as np

#######################################MLPENCODER,projection_head
class lossfunction(nn.Module):
    def __init__(self):
        super(lossfunction, self).__init__()


    def forward(self,c_i,h_i,c_j,h_j):
        pass
        # si=0
        # loss=0
        # for i in range(1,b):
        #     for k in range(l):
        #         for m in range(1, b + 1):
        #             for n in range(l):
        #                 si += np.exp(nn.CosineSimilarity(c_i, h_i[m][n]) / t)
        #         si_0 = nn.CosineSimilarity(c_i, h_i[k])
        #         loss_0 = si_0 / (si + si_0)
        #     loss+=loss_0
        # loss=loss/(b*(l+1))
        # return loss




# 继承于AbstractTrainer
class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        # 计算交叉熵损失
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.lossfunction = lossfunction()
    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    # 计算loss的函数
    def calculate_loss(self, batch):
        seqs, labels, negs = batch

        # 输入模型之后的预测结果
        logits = self.model([seqs, negs])  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        # 计算交叉熵损失
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores = self.model(seqs)  # B x T x V 维数
        scores = scores[:, -1, :]  # B x V 取最后一维的结果作为最终预测结果
        # 以列形式 取出所有candidates物品的打分结果
        scores = scores.gather(1, candidates)  # B x C

        # 调用utils文件中的recalls_and_ndcgs_for_ks函数
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        # 返回recall和ndcg的计算结果
        return metrics


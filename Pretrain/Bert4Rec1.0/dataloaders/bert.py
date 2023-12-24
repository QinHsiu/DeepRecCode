from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


# 继承AbstractDataloader类
class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        # 物品映射个数
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        # 训练序列中被mask的物品个数 None？？？
        self.mask_prob = args.bert_mask_prob
        # 物品个数+1
        # 此处应该不需要修改
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        # random bert中没有用到训练的负样本
        code = args.train_negative_sampler_code
        # train_negative_sample_size=100  train_negative_sampling_seed=None
        #
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        # 用于评价的负采样方法
        code = args.test_negative_sampler_code
        # test_negative_sample_size=100  test_negative_sampling_seed=None
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        # 调用get_negative_samples函数 保存生成的负样本 并返回
        self.train_negative_samples = train_negative_sampler.get_negative_samples(sample_type="train")
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    # 加载训练集测试集验证集
    def get_pytorch_dataloaders(self):
        # 分别对训练集验证集测试集进行处理
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        # 返回三个dataloader
        return train_loader, val_loader, test_loader

    # 训练集loader
    def _get_train_loader(self):
        # 调用_get_train_dataset函数
        dataset = self._get_train_dataset()
        # 得到tokens和labels 固定长度为100
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng, self.train_negative_samples)
        return dataset

    # 验证集loader
    def _get_val_loader(self):
        # 调用_get_eval_loader函数
        # 传入val
        return self._get_eval_loader(mode='val')

    # 测试集loader
    def _get_test_loader(self):
        # 调用_get_eval_loader函数
        # 传入test
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        # test和train时可以用不一样的batch size
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        # 调用_get_eval_dataset函数
        dataset = self._get_eval_dataset(mode)
        # 得到测试时的输入序列 所有候选物品 以及 物品对应的label  所有序列固定长度为100
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode=='val':
            answers = self.val
            dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                      self.test_negative_samples)
        else:
            answers = self.test
            dataset = BertTestDataset(self.train, self.val, answers, self.max_len, self.CLOZE_MASK_TOKEN,
                                      self.test_negative_samples)
        #answers = self.val if mode == 'val' else self.test
        #dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    # self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count, self.rng
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng, negative_samples):
        # 用户对应的序列内容
        self.u2seq = u2seq
        # 根据用户id进行排序
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # 取出用户和对应的物品
        user = self.users[index]
        seq = self._getseq(user)
        negs = self.negative_samples[user]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            # 对于每个物品都计算一下概率
            # 'bert_mask_prob': 0.15
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # 根据BERT论文中的方法 为了缓解预训练与微调时的不匹配 (mask token不出现在微调过程中)
                # 80%的概率被mask
                if prob < 0.8:
                    # self.mask_token = numItem + 1 利用物品个数+1的编号进行mask
                    tokens.append(self.mask_token)
                # 0.8-0.9的概率被随机替换成别的单词
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                # 0.9-1的概率保持不变
                else:
                    tokens.append(s)

                # label保存实际物品编号 表示被mask了
                labels.append(s)
            else:
                # token保存当前物品编号 label保存0表示没有被mask
                tokens.append(s)
                labels.append(0)

        # 取100个
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        negs = negs[-self.max_len:]

        # 0
        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        negs = [0] * mask_len + negs
        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.torch.LongTensor(negs)

    def _getseq(self, user):
        return self.u2seq[user]


class BertEvalDataset(data_utils.Dataset):
    # self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # 取出当前对应的用户的内容
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        # 所有候选物品 由一些负样本和正确的样本组成
        candidates = answer + negs
        # 正确答案的标签为1 其余负样本的标签为0
        labels = [1] * len(answer) + [0] * len(negs)

        # 对测试集的位置进行mask
        seq = seq + [self.mask_token]
        # 截取相同的长度
        seq = seq[-self.max_len:]
        # 如果长度不够 则需要padding
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        # 返回测试时的序列输入 所有候选物品 以及 所有物品对应的label
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)

class BertTestDataset(data_utils.Dataset):
    # self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, u2seq, u2val, u2answer, max_len, mask_token, negative_samples):
        self.u2seq = u2seq
        self.u2val = u2val
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        # 取出当前对应的用户的内容
        user = self.users[index]
        seq = self.u2seq[user]
        val = self.u2val[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]

        # 所有候选物品 由一些负样本和正确的样本组成
        candidates = answer + negs
        # 正确答案的标签为1 其余负样本的标签为0
        labels = [1] * len(answer) + [0] * len(negs)

        # 对测试集的位置进行mask
        seq = seq + val + [self.mask_token]
        # 截取相同的长度
        seq = seq[-self.max_len:]
        # 如果长度不够 则需要padding
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq


        # 返回测试时的序列输入 所有候选物品 以及 所有物品对应的label
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
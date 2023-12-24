from .ml_1m import ML1MDataset
from .ml_20m import ML20MDataset

DATASETS = {
    # 调用code函数 返回对应的
    ML1MDataset.code(): ML1MDataset,
    ML20MDataset.code(): ML20MDataset
}


def dataset_factory(args):
    # 选择数据集
    # <class 'datasets.ml_1m.ML1MDataset'>
    dataset = DATASETS[args.dataset_code]
    return dataset(args)

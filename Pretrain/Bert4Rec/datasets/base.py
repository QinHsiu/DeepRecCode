from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.min_rating = args.min_rating  # 4 Only keep ratings greater than equal to this value
        self.min_uc = args.min_uc  # 5 Only keep users with more than min_uc ratings
        self.min_sc = args.min_sc  # 0 Only keep items with more than min_sc ratings
        self.split = args.split  # leave_one_out

        assert self.min_uc >= 2, 'Need at least 2 ratings per user for validation and test'

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        # 调用preprocess函数
        self.preprocess()
        # 调用_get_preprocessed_dataset_path函数
        dataset_path = self._get_preprocessed_dataset_path()
        # 加载文件
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def preprocess(self):
        # 调用_get_preprocessed_dataset_path函数
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        # 下载原始数据
        self.maybe_download_raw_dataset()
        # ['uid', 'sid', 'rating', 'timestamp']
        df = self.load_ratings_df()
        # 取打分高于最低分数的物品
        df = self.make_implicit(df)
        # 过滤
        df = self.filter_triplets(df)
        # 重新对用户物品进行编号
        df, umap, smap = self.densify_index(df)
        # 划分训练集 验证集 测试集
        train, val, test = self.split_df(df, len(umap))
        dataset = {'train': train,
                   'val': val,
                   'test': test,
                   'umap': umap,
                   'smap': smap}
        # 保存以上文件
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    # 判断是否要下载数据集
    def maybe_download_raw_dataset(self):
        folder_path = self._get_rawdata_folder_path()
        if folder_path.is_dir() and\
           all(folder_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
            return
        print("Raw file doesn't exist. Downloading...")
        # true
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, folder_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            folder_path.mkdir(parents=True)
            shutil.move(tmpfile, folder_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    # 将隐式反馈转换为显式反馈
    def make_implicit(self, df):
        print('Turning into implicit ratings')
        # 取打分高于最低分数的物品
        df = df[df['rating'] >= self.min_rating]
        # return df[['uid', 'sid', 'timestamp']]
        return df

    # 过滤
    def filter_triplets(self, df):
        print('Filtering triplets')
        # 物品
        if self.min_sc > 0:
            # 根据物品id划分group
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        # 用户
        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    # 重新编号
    def densify_index(self, df):
        print('Densifying index')
        # 对用户和物品重新编号
        # ===========================================2021_1_9===========================================================
        # umap = {u: i for i, u in enumerate(set(df['uid']))}
        # smap = {s: i for i, s in enumerate(set(df['sid']))}
        umap = {u: i for i, u in enumerate(set(df['uid']))}
        smap = {s: i+1 for i, s in enumerate(set(df['sid']))}
        # self.CLOZE_MASK_TOKEN应该不需要修改
        # ===========================================2021_1_9===========================================================
        # 将新的编号映射到对应的物品上
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        # 返回处理后的df 对应的用户和物品map
        return df, umap, smap

    def split_df(self, df, user_count):
        # 留一法
        # 倒数第一个物品作为测试集 倒数第二个物品作为验证集 前面部分物品作为训练集
        if self.args.split == 'leave_one_out':
            print('Splitting')
            user_group = df.groupby('uid')
            # 按时间戳排序
            user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
            train, val, test = {}, {}, {}
            for user in range(user_count):
                items = user2items[user]
                train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
            return train, val, test
        elif self.args.split == 'holdout':
            print('Splitting')
            np.random.seed(self.args.dataset_split_seed)
            eval_set_size = self.args.eval_set_size

            # Generate user indices
            permuted_index = np.random.permutation(user_count)
            train_user_index = permuted_index[                :-2*eval_set_size]
            val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
            test_user_index  = permuted_index[  -eval_set_size:                ]

            # Split DataFrames
            train_df = df.loc[df['uid'].isin(train_user_index)]
            val_df   = df.loc[df['uid'].isin(val_user_index)]
            test_df  = df.loc[df['uid'].isin(test_user_index)]

            # DataFrame to dict => {uid : list of sid's}
            train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
            return train, val, test
        else:
            raise NotImplementedError

    def _get_rawdata_root_path(self):
        # 'Data'
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.split)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')


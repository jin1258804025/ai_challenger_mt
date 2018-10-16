# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems

from utils import util
util.gpu_config(0)

@registry.register_problem
class MyProblem(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 2**11
    @property
    def max_subtoken_length(self):
        return 10
    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        train_data = open("./rawdata/train_data/data.txt", "r")
        text = train_data.readlines()
        
        sub_train_num = 100000
        text = text[:sub_train_num]
        
        train_data.close()
        
        for sample in text:
            DocID, SenID, EngSen, ChnSen = sample.split('\t')
            en = EngSen.strip()
            zh = ChnSen.strip()
            yield {
                "inputs": en,
                "targets": zh
            }

    def generate_samples_(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        q_r = open("./rawdata/q.txt", "r")
        a_r = open("./rawdata/a.txt", "r")

        comment_list = q_r.readlines()
        tag_list = a_r.readlines()
        q_r.close()
        a_r.close()
        for comment, tag in zip(comment_list, tag_list):
            comment = comment.strip()
            tag = tag.strip()
            # print comment, tag
            yield {
                "inputs": comment,
                "targets": tag
            }

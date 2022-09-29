from time import time
from datetime import timedelta

from fastNLP.io import JsonLoader
from fastNLP.io.data_bundle import DataBundle
from fastNLP import logger

class CoLoExtLoader(JsonLoader):
    def __init__(self, pad_id, ext_num):
        super(CoLoExtLoader, self).__init__()
        self.pad_id = pad_id
        self.ext_num = ext_num

    def _load(self, paths):
        dataset = super(CoLoExtLoader, self)._load(paths)
        return dataset


    def process(self, paths):
        logger.info('Start loading datasets !!! Notice that you should extract labels and tokenize first with our preprocess script')
        start = time()

        # load datasets
        datasets = {}
        for name in paths:
            datasets[name] = self._load(paths[name])
            # set input and target
            datasets[name].apply(lambda x: int(x['article_id']), new_field_name='article_id_int')
            datasets[name].set_input('text_id', 'cls_ids', 'article_id_int')
            datasets[name].set_target('labels')
            # set padding value
            datasets[name].set_pad_val('text_id', self.pad_id)
            datasets[name].set_pad_val('cls_ids', -1)
            datasets[name].set_pad_val('labels', 0)
            datasets[name].set_pad_val('article_id_int', -1)
            if name != "test":
                #####  drop instance with sents num < ext_num in training set
                datasets[name].drop(lambda ins: len(ins['text']) < self.ext_num)
                datasets[name].drop(lambda ins: len(ins['cls_ids']) != len(ins["text"]), inplace=True)

        print('Finished in {}'.format(timedelta(seconds=time() - start)))

        return DataBundle(datasets=datasets)

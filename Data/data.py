
import numpy as np
from data_seg import PortraitSeg
from jittor.dataset.dataset import Dataset
from easydict import EasyDict as edict

class MyDataset(Dataset): 
    def __init__(self, exp_args):
        assert exp_args.task in ['seg'], 'Error!, <task> should in [seg]'
        
        self.exp_args = exp_args
        self.task = exp_args.task
        self.datasetlist = exp_args.datasetlist
        self.data_root = exp_args.data_root # data_root = '/home/dongx12/Data/'
        self.file_root = exp_args.file_root # file_root = '/home/dongx12/PortraitNet/data/select_data/'
        
        self.datasets = {}
        self.imagelist = []
        
        # load dataset
        # if 'supervisely_face_easy' in self.datasetlist:
        #     ImageRoot = self.data_root
        #     AnnoRoot = self.data_root
        #     ImgIds_Train = self.file_root + 'supervisely_face_train_easy.txt'
        #     ImgIds_Test = self.file_root + 'supervisely_face_test_easy.txt'
        #     exp_args.dataset = 'supervisely_face_easy'
        #     self.datasets['supervisely_face_easy'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        if 'EG1800' in self.datasetlist:
            ImageRoot = self.data_root + 'EG1800/Images/'
            AnnoRoot = self.data_root + 'EG1800/Labels/'
            ImgIds_Train = self.file_root + 'train.txt'
            ImgIds_Test = self.file_root + 'test.txt'
            exp_args.dataset = 'eg1800'
            self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
            
        # image list
        for key in self.datasets.keys():
            length = len(self.datasets[key])
            for i in range(length):
                self.imagelist.append([key, i])
        
        # set_attrs must be called to set batch size total len and shuffle like __len__ function in pytorch
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle) # bs , total_len, shuffle

    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]
        
        input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
        return input_ori.astype(np.float32), input.astype(np.float32), \
                    output_edge.astype(np.int64), output_mask.astype(np.int64)
           



exp_args = edict()
exp_args.tasks = 'seg'
exp_args.datasetlist = ['EG1800']
exp_args.data_root = "./"
exp_args.file_root = "./EG1800/datalist"

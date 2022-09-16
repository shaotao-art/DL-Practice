from distutils.log import INFO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from cv_preprocess import PreProcessor
import torchvision

class TrainDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        self.pre_processor = PreProcessor()

    def __getitem__(self, index):
        return self._apply_preprocess(index, self.pre_processor)
    
    def __len__(self):
        return len(self.datas)

    def _apply_preprocess(self, index, pre_processor):
        x, y = self.datas[index]
        x = pre_processor(x)
        return x, y

class TestDataset(Dataset):
    def __init__(self) -> None:
        self.datas = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        self.pre_processor = PreProcessor()

    def __getitem__(self, index):
        return self._apply_preprocess(index, self.pre_processor)

    def __len__(self):
        return len(self.datas)

    def _apply_preprocess(self, index, pre_processor):
        x, y = self.datas[index]
        x = pre_processor(x)
        return x, y
    

def get_train_loader(configer):
    dataset = TrainDataset()
    b_s = configer.params['b_s']
    dataloader = DataLoader(dataset,
                            batch_size=b_s,
                            shuffle=True,
                            num_workers=configer.params['num_workers'])
    info_str = '\nGetting Train DataLoader\n'
    info_str += f'\ttraining set num samples: {len(dataset)}\n'
    info_str += f'\ttraining set batch size: {b_s}\n'
    info_str += f'\ttraining set len dataloder: {len(dataloader)}\n'
    print(info_str)
    return dataloader

def get_test_loader(configer):
    dataset = TestDataset()
    b_s = configer.params['b_s']
    dataloader = DataLoader(dataset,
                            batch_size=b_s,
                            shuffle=True,
                            num_workers=configer.params['num_workers'])
    info_str = '\nGetting Test DataLoader\n'
    info_str += f'\tTesting set num samples: {len(dataset)}\n'
    info_str += f'\tTesting set batch size: {b_s}\n'
    info_str += f'\ttesting set len dataloder: {len(dataloader)}\n'
    print(info_str)
    return dataloader
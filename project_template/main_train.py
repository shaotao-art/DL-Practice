from unittest import getTestCaseNames
from configer import Configer
from data import get_train_loader
from data import get_test_loader
from logger import Logger
from model import NNet
from trainer import Trainer
from utilers import Utiler
import os
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

utiler = Utiler()
utiler.apply()
print(utiler)

configer = Configer()
device = configer.params['device']
logger = Logger()
logger.log(configer)
print(configer)
logger.log(utiler)


model = NNet()
logger.log(model)
print(model)

optimizer = optim.Adam(model.parameters(), lr=configer.params['l_r'])
criterion = nn.CrossEntropyLoss()
train_dataloader = get_train_loader(configer)
valid_dataloader = get_test_loader(configer) 

if os.path.exists(configer.params['model_save_path']):
    start_epoch = utiler.load_ckp(model, optimizer, device)
else:
    start_epoch = 0
    model.to(device)

trainer = Trainer(configer.params['num_epoch'])
trainer.run(train_dataloader, valid_dataloader, model, optimizer, criterion, logger, start_epoch, utiler, device)








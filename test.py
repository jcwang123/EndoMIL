import torch
import torchvision.models as models
from dataloader import EndoBag
from model import DepResLoss
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score
import torch.utils.data as data_utils
import pickle
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader

# from AFSfMLearner.test_simple_copy import test_simple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# balanced-class entropy loss
import torch
from balanced_loss import Loss

from tqdm import tqdm

import time
import psutil

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    torch.cuda.set_device(0)

    saved_model_path = '/data1/ryqiu/Zhongshan/EndoMIL/checkpoints'
    model_names = ['WS','CH','RAC','ZZ','JPY']
    
    columns = ['Task', 'Test Accuracy', 'Kappa', 'Macro Precision', 'Macro Recall', 'Macro F1']
    results_df = pd.DataFrame(columns=columns)
    matrix_folder = saved_model_path
    os.makedirs(matrix_folder, exist_ok=True)


    for model_name in model_names:

        parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        if args.cuda:
            print('\nGPU is ON!')

        
        loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        model_path = os.path.join(saved_model_path, model_name+'_best.pth') 
        loaded_model_state_dict = torch.load(model_path, map_location='cuda:0')

        model =  DepResLoss(model_name)
        model.load_state_dict(loaded_model_state_dict, strict=False)
        print(' model is loaded. ')
  
        if args.cuda:
            model = model.cuda()

        model.eval()

        with open('/data1/ryqiu/Zhongshan/EndoMIL/pesudo_dep.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        label_path = '/data1/ryqiu/Zhongshan/EndoMIL/data/labels/testlabel.xlsx'
        root ='/data1/ryqiu/Zhongshan/EndoMIL/data' 

        test_loader = data_utils.DataLoader(EndoBag(target_number=model_name, 
                                                        root=root, 
                                                        label_path = label_path, 
                                                        train=False),
                                                batch_size=1,
                                                shuffle=False, **loader_kwargs)


        test_loss, test_error = 0., 0.
        y_pred, y_true, y_score = [], [], []
        positive_bag2, positive_bag1, neg_bag = 0., 0., 0.

        scoring = {} 


        with torch.no_grad(): 
            for idx, (id, data, label) in enumerate(tqdm(test_loader, desc='testset', unit='patient')):
                if id[0] in data_dict:
                    dep = torch.tensor(data_dict[id[0]])
                else:
                    print(f"cannot find Dep for {id}")
            
                if label == 2: 
                    positive_bag2 += 1
                elif label ==1:
                    positive_bag1 += 1
                else:
                    neg_bag += 1

                if args.cuda:
                    data, label, dep = data.cuda(), label.cuda(), dep.cuda()
                data, label, dep = Variable(data), Variable(label), Variable(dep)

                output, A, M, X, H = model(data, dep) 

                Y_hat = torch.argmax(output, dim=1)
                logits = nn.Softmax(dim=1)(output)[0][1]
                error = 1. - Y_hat.eq(label).cpu().float().mean().data.item()

                y_score.append(logits.item())
                y_pred.append(Y_hat.item())
                y_true.append(label.item())
            

                test_error += error
                scoring[id[0]] = (Y_hat.float().item(), label.item())


        with open('/data1/ryqiu/Zhongshan/EndoMIL/outputs/{}.pkl'.format(model_name), 'wb') as f:
            pickle.dump(scoring, f)


        test_error /= len(test_loader)
        test_loss /= len(test_loader)

        print('\nTest Set size: {},Test Set, Loss: {:.8f}, Test acc: {:.4f}'.format(len(test_loader),test_loss, 1 - test_error))
        print('Task: {} , Test acc: {:.4f}'.format(model_name, 1 - test_error))

        kappa = cohen_kappa_score(y_true, y_pred)
        print('kappa={:.4f}'.format(kappa))

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        print('Macro Precision={:.4f}, Macro Recall={:.4f}, Macro F1={:.4f}'.format(macro_precision,macro_recall,macro_f1))
        

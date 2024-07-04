"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random
import pandas as pd
from PIL import Image
# from model import Attention,  GatedAttention

import os
import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

# from AFSfMLearner.test_simple_copy import test_simple
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset

def merge_labels(label_vectors):
    merged_label = [1 if any(label[i] == 1 for label in label_vectors) else 0 for i in range(10)]
    return merged_label



class EndoBag(data_utils.Dataset):
    def __init__(self, target_number=None, root=None, label_path = None, train=False):
            self.target_number = target_number
            self.root = root
            self.train = train 
            self.label_path = label_path

            # 获取所有文件夹名字
            folder_names = sorted(os.listdir(root))
            
            self.labels, exist_name = self.read_excel_and_extract_data(label_path, folder_names, target_number)

            # 存储所有子列表的列表
            all_image_paths = []

            # 遍历每个文件夹
            for folder_name in exist_name:
                folder_path = os.path.join(root, str(folder_name))
                image_paths = []
                
                # 遍历文件夹中的图片文件
                for filename in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, filename)
                    image_paths.append(img_path)
                
                all_image_paths.append(image_paths)
            
            # new_name_list = [name.split('-')[1] for name in  folder_names]
            # new_name_list = folder_names
            self.patient_bags_list = all_image_paths        

    def read_excel_and_extract_data(self, excel_file, new_name_list, target_number):
    # 读取Excel文件
        df = pd.read_excel(excel_file)

        result_array = []
        result =[]
        exist_name = []
        # 遍历新列表
        for name in (new_name_list):
            # 在Excel第一列中寻找匹配的名称, new_name_list的长度是209，即文件夹大小
            matching_row = df[df.iloc[:, 0].astype(str).str.contains(name, case=False, na=False)]

            if not matching_row.empty:
                # 提取匹配单元格后五列的数据
                # data_vector = matching_row.iloc[0, 1:6].values.tolist() # split1
                data_vector = matching_row.iloc[0, 1:8].values.tolist() #risk split
                result_array.append(data_vector)
                exist_name.append(name)
        if target_number == 'WS':
            t = 0
        elif target_number == 'CH':
            t = 1
        elif target_number == 'RAC':
            t = 2
        elif target_number == 'ZZ':
            t = 3
        else:
            t=4

        for i in range(0, len(result_array)):
            # result.append(result_array[i][0]) #萎缩
            # result.append(result_array[i][1]) #肠化
            # result.append(result_array[i][2]) #RAC

            # result.append(result_array[i][3]) #肿胀
            # result.append(result_array[i][4]) #鸡皮样

            ## risk model
            # result.append(result_array[i][6]) #risk, low=0, high=1
            # result.append(result_array[i]) #test label

            result.append(result_array[i][t]) 

        return result, exist_name
    
    def __len__(self):
       
            return len(self.patient_bags_list)
 
    def __getitem__(self, index):
        label = self.labels[index]
        bag_path = self.patient_bags_list[index]

        train_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        test_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        onebag = True
        img_order = []
        if self.train:
            for instance_path in bag_path:
            
                image = Image.open(instance_path).convert("RGB")
                image = train_transform(image)
                if onebag is True:
                    onebag = image[np.newaxis, :]
                    img_order.append(instance_path.split('/')[-1])
                else:
                    onebag = np.concatenate((onebag, image[np.newaxis, :]), axis=0)
                    img_order.append(instance_path.split('/')[-1])

        else:
            for instance_path in bag_path:
            
                image = Image.open(instance_path).convert("RGB")
                image = test_transform(image)
                if onebag is True:
                    onebag = image[np.newaxis, :]
                    img_order.append(instance_path.split('/')[-1])
                else:
                    onebag = np.concatenate((onebag, image[np.newaxis, :]), axis=0)
                    img_order.append(instance_path.split('/')[-1])
                    
        # print(bag_path)

        patient_id = instance_path.split('/')[-2]
        
        # patient_id = instance_path.split('/')[-1][:8]
        # if patient_id in ['94651500','95345630','95377934','95381416','95386156','95450683']:
        #     print(bag_path)

        return patient_id, onebag, label


if __name__ == "__main__":
    print['']
   
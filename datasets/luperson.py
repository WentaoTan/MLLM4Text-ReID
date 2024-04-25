import os.path as op
import random
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset

import os
import json
from prettytable import PrettyTable
import collections
import numpy as np

class LuPerson_PEDES(BaseDataset):
    dataset_dir = 'LUPerson_images'
    def __init__(self, root='', verbose=True):
        super(LuPerson_PEDES, self).__init__()
        self.dataset_dir = '/data0/wentao/data/LuPerson-T/LUPerson_images'
        self.image_dir = op.join(self.dataset_dir, 'LUPerson-MLLM')
        self.caption_dir = self.dataset_dir
        self.train_img_paths = []
        self.train_cap_paths = []

        self.test_img_paths = []
        self.test_cap_paths = []

        for filename in os.listdir(self.image_dir): # part1234
            image_path = os.path.join(self.image_dir, filename)
            if filename.endswith('.jpg'):
                self.train_img_paths.append(image_path)
        for filename in os.listdir(self.caption_dir):
            caption_path = os.path.join(self.caption_dir, filename)
            if filename.endswith('.json'):
                self.train_cap_paths.append(caption_path)
        
        train_cap_dict = self._merged_multi_json_file(self.train_cap_paths)
        test_cap_dict = self._merged_json_file(self.test_cap_paths)

        self.train, self.train_id_container, self.part_dataset, num_caption,self.fpath2part_cap,self.fpaht2sim = self._get_dataset(self.train_img_paths, train_cap_dict)
        self.test = self._get_test_dataset(self.test_img_paths, test_cap_dict)
        
        self.logger.info("=> LuPerson-MLLM Images and Captions are loaded")
        self.logger.info("LuPerson-MLLM Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(['train', len(set(self.train_id_container)),len(self.train), num_caption])
        table.add_row(['test', len(self.test["image_pids"]),len(self.test["image_pids"]), len(self.test["image_pids"])])
        self.logger.info('\n' + str(table))
        

    def _merged_json_file(self, json_path_list):
        merged_dict = {}

        # 逐个读取JSON文件并合并到字典中
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                merged_dict.update(data)
        return merged_dict
    
    def _merged_multi_json_file(self, json_path_list):
        merged_dict = collections.defaultdict(list)
        json_path_list = [
                          "./caption/Ts-qwen.json",
                          "./caption/Td-qwen.json",
                          "./caption/Ts-shikra.json",
                          "./caption/Td-shikra.json",
                          ]
        for file_path in json_path_list:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                print(file_path, len(data))
                for k,v in data.items():
                    img_name = k.split('/')[-1]
                    merged_dict[img_name].append(v)
        return merged_dict

    def _get_test_dataset(self, test_img_paths, cap_dict):
        dataset = {}
        img_paths = []
        captions = []
        image_pids = []
        caption_pids = []
        for i in range(len(test_img_paths)):
            pid = i
            img_path = test_img_paths[i]
            img_paths.append(img_path)
            image_pids.append(pid)
            path2cap = '/'.join(img_path.split('/')[-1])
            caption = cap_dict[path2cap][0]
            captions.append(caption)
            caption_pids.append(pid)
        dataset = {
            "image_pids": image_pids,
            "img_paths": img_paths,
            "caption_pids": caption_pids,
            "captions": captions
        }
        return dataset
    
    def _get_dataset(self, img_paths, cap_dict):
        safe_dict = collections.defaultdict(list)
        with open('./caption/Ts-shikra.json', 'r') as json_file:
            data = json.load(json_file)
            for k,v in data.items():
                img_name = k.split('/')[-1]
                safe_dict[img_name].append(v)
        
        with open('./caption/Ts-qwen.json', 'r') as json_file:
            data = json.load(json_file)
            for k,v in data.items():
                img_name = k.split('/')[-1]
                safe_dict[img_name].append(v)
        pid_container = set()
        img_paths = sorted(img_paths)

        dataset = []
        part_dataset = []
        idx_count = 0
        pid_count = 0
        num_caption = 0

        fpath2part_cap = {}
        fpaht2sim = {}
        for i in range(len(img_paths)):
            img_path = img_paths[i]
            
            path2cap = img_path.split('/')[-1]
            caption = cap_dict[path2cap]

            # if len(caption) != 4:
            #     continue
            fpath2part_cap[img_path] = {}
            fpaht2sim[img_path] = {}
            pid = pid_count
            image_id = idx_count
            pid_container.add(pid)
            for cap in caption:
                if 'description]' in cap or '<' in cap: 
                    try:
                        cap = random.choice(safe_dict[path2cap])
                    except:
                        pass
                part2sim = 77 * [1- 0.15]
                part2sim = np.array(part2sim)
                dataset.append([pid,idx_count,img_path, cap, part2sim])
                num_caption += 1
                idx_count += 1
            pid_count += 1
        assert idx_count == len(dataset)

        return dataset, pid_container, part_dataset,num_caption,fpath2part_cap,fpaht2sim
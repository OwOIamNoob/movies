
import os

import torch 
from torch.utils.data import Dataset
from torch.nn.functional import pad
from torchvision.transforms import Compose
from torchvision import transforms

import numpy as np
import cv2
from collections import defaultdict
import random
import json
from PIL import Image 
import math
from typing import Optional, List, Dict

from src.data.components.vocab import Vocab

class MLDataset(Dataset):
    def __init__(self, 
                 data_dir:str,
                 title_vocab: Vocab,
                 genre_vocab: Vocab, 
                 img_dir:str,
                 data_type:str):
        super().__init__()
        self.title_vocab = title_vocab
        self.genre_vocab = genre_vocab
        self.img_dir = img_dir
        self.prefix = "{}.jpg"
        self.data = dict()
        data_type += ".json"
        self.ids = []
        path = os.path.join(data_dir,data_type)
        if not os.path.isfile(path):
            raise FileNotFoundError("Where da heck is " + path)
        self.load(path)
    
    def load(self, path):
        file = open(path,"r")
        self.data = json.load(file)
        self.ids = list(self.data.keys())
        print(len(self.ids))
        for movieid in self.ids:
            self.data[movieid]["title"] = self.title_vocab.tokenize(self.data[movieid]["title"])
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        movieid = self.ids[index]
        info = self.data[movieid]
        if not info['image']:
            image = None
        else:
            image = Image.open(os.path.join(self.img_dir, self.prefix.format(movieid))).convert("RGB")
        
        return {"movieid": movieid,
                "title": self.title_vocab.to_index(info['title']),
                "genre": self.genre_vocab.to_index(info['genre']),
                "ratings": np.array(info['ratings']),
                "image": image
                }

class MLTransformedDataset(Dataset):
    def __init__(self,
                 dataset:MLDataset,
                 pad_id: int = 0,
                 transforms: Optional[Compose] = Compose([transforms.ToTensor()]),
                 rating_transforms: Optional[Compose] = Compose([torch.FloatTensor()]),
                 rating_img_size: int = 64):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.rating_transforms = rating_transforms
        self.pad_id = pad_id
        self.rating_img_size = rating_img_size

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        if len(sample['ratings']) == 0:
            rating_image_size = 1
            sample['ratings'] = np.zeros((1, 4), dtype=np.uint8)
        else:
            rating_image_size = int(math.ceil(math.sqrt(sample['ratings'].shape[0])))
        
        sample['ratings'] = np.pad(sample['ratings'], 
                                   ((rating_image_size ** 2 - sample['ratings'].shape[0], 0), (0, 0)), 
                                   'constant', 
                                   constant_values=(self.pad_id, )).reshape((rating_image_size, rating_image_size, 4)).astype(np.uint8)
        # print(sample['ratings'].to_numpy())
        sample['ratings'] = cv2.resize(sample['ratings'], 
                                       (self.rating_img_size, self.rating_img_size),
                                        interpolation=cv2.INTER_NEAREST).transpose((2, 0, 1))
        sample['ratings'] = self.rating_transforms(sample['ratings'])
        
        if sample['image'] is not None:
            sample['image'] = self.transforms(sample['image'])

        
        return sample
        
        
        
        
class Collator:
    def __init__(self, 
                 max_seq_len:int, 
                 target_vocab:Vocab,
                 pad_id:int,
                 rating_img_size: int,
                 img_size:int):
        self.target_vocab = target_vocab
        self.max_seq_len = max_seq_len
        self.max_label_len = len(target_vocab)
        self.seq_pad_id = pad_id
        self.label_pad_id = self.target_vocab.vocab['<PAD>']
        self.rating_img_size = rating_img_size
        self.img_size = img_size
    
    def __call__(self, batch):
        movieids = []
        titles = []
        genre_probs = []
        ratings = []
        images = []
        
        for item in batch:
            movieids.append(int(item['movieid']))
            title_length = item['title'].shape[0]
            title = torch.LongTensor(np.pad(item['title'], 
                                            (0, self.max_seq_len - title_length), 
                                            'constant', 
                                            constant_values=(self.seq_pad_id, )))
            titles.append(title)
        
            genre_probs.append(torch.FloatTensor(self.target_vocab.to_prob(item['genre'])))
            
            if item['image'] is None:
                images.append(torch.zeros((3, self.img_size, self.img_size)))
            else:
                images.append(item['image'])
            
            ratings.append(item['ratings'])
            
        return {"movieids": torch.LongTensor(movieids),
                "titles": torch.stack(titles),
                "label": torch.stack(genre_probs),
                "ratings": torch.stack(ratings),
                "images": torch.stack(images)}
    

# import pyrootutils
# pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

# title_vocab = Vocab("/work/hpc/potato/movies/data/movies/dataset/words.txt")
# genre_vocab = Vocab("/work/hpc/potato/movies/data/movies/dataset/genres.txt")

# dataset = MLDataset(data_dir="data/movies/dataset/",
#                     img_dir="data/movies/dataset/ml1m-images/",
#                     title_vocab=title_vocab,
#                     genre_vocab=genre_vocab,
#                     data_type="test")

# data = dataset[200]
# print(data['ratings'])
# transformed_dataset = MLTransformedDataset(dataset=dataset,
#                                            pad_id=0.,
#                                            transforms=Compose([transforms.Resize((256, 256)),
#                                                                transforms.RandomAffine(degrees=(-10, 10),
#                                                                                        translate=(0.1, 0.1),
#                                                                                        interpolation=transforms.InterpolationMode.NEAREST),
#                                                                transforms.ToTensor()
#                                                                ]),
#                                            rating_transforms=Compose([torch.FloatTensor,
#                                                                       transforms.Normalize(mean=[0.5, 2.5, 10, 2.5],
#                                                                                            std=[0.5, 2.5, 10, 2.5])]),
#                                            )

# data = transformed_dataset[200]
# print(data['ratings'].permute(1, 2, 0))
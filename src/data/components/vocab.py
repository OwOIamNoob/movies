from nltk import wordpunct_tokenize
import os 
import numpy as np
import re
import torch 

class Vocab:
    def __init__(self,
                 path:str=None):
        self.vocab = dict()
        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.index = dict()
        self.update = True
        if path is not None:
            self.load_vocab(path)
            self.update = False
        
    
    def load_vocab(self, path):
        f = open(path, "r")
        for line in f:
            if line.strip() not in self.vocab:
                self.vocab[line.strip()] = len(self.vocab)
        
    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        tokens = wordpunct_tokenize(text)
        tokens = tokens[:-1]
        
        if self.update:
            for token in tokens: 
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        
        return tokens

    def to_index(self, tokens):
        return np.array([self.vocab[word] for word in tokens])
    
    def to_prob(self, indexes):
        ref = np.zeros((len(self), ), dtype=float)
        ref[indexes] = 1.
        return ref
    
    def __len__(self) -> int:
        return len(self.vocab)

    def decode(self, inp, threshold):
        if len(self.index) == 0:
            self.index = dict(zip(self.vocab.values(), self.vocab.keys()))
        output = []
        inp = inp.cpu().numpy()
        if inp.shape[0] == len(self):
            inp = np.where(inp > threshold)[0]
        output = [self.index[pos] for pos in inp]
        
        return output
            
    
    
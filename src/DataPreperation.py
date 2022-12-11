#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import skimage.io as io
import random
import json
import pickle

import math
import numpy as np
from collections import Counter
import nltk
nltk.download('punkt')
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler
from pycocotools.coco import COCO
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# # Vocabulary Definitions

# In[2]:


class WordVocabulary:
    def __init__(self,
        vocab_threshold,
        vocab_file='./word_vocab.pkl',
        start_word='<start>',
        end_word='<end>',
        unk_word="<unk>",
        annotations_file='../data/annotations/captions_train2017.json',
        vocab_from_file=False,
        verbose=False):
        """Initialize the word vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
          verbose: If true, print out iterations. 
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.word2idx = dict()
        self.idx2word = dict()
        self.verbose = verbose
        self.get_vocab()
    
    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                print(os.getcwd())
                print(self.vocab_file)
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
                
            print('Word Vocabulary successfully loaded from {} file'.format(self.vocab_file))
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
            
            print('Word Vocabulary succesfully created at {}'.format(self.vocab_file))
            
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        idx = 0
        coco_captions = COCO(self.annotations_file)
        counter = Counter()
        ids = coco_captions.anns.keys()
        
        def add_word(word, idx):
            """Add the given word token to vocab"""
            if not word in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx = idx + 1
                
            return idx
                
        idx = add_word(self.start_word, idx)
        idx = add_word(self.end_word, idx)
        idx = add_word(self.unk_word, idx)
        
        if self.verbose:
            print(f'Current idx: {idx}')
            print('Current word2idx:', self.word2idx)
            
        for i, caption_id in enumerate(ids):
            caption = str(coco_captions.anns[caption_id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if self.verbose and i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))
            
        words = []
        for word, count in counter.items():
            if count >= self.vocab_threshold:
                words.append(word)
                
        for word in words:
            idx = add_word(word, idx)

    def __getitem__(self, word):
        if not word in self.word2idx:  
            return self.word2idx[self.unk_word]
        
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)


# In[3]:


class CharVocabulary:
    def __init__(self,
        vocab_threshold,
        vocab_file='./char_vocab.pkl',
        unk_char='<unk>',
        annotations_file='../data/annotations/captions_train2017.json',
        vocab_from_file=False,
        verbose=False):
        """Initialize the char vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          unk_char: Special word denoting unknown chars.
          annotations_file: Path for train annotation file.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
          verbose: If true, print out iterations. 
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.unk_char = unk_char
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.char2idx = dict()
        self.idx2char = dict()
        self.verbose = verbose
        self.get_vocab()
    
    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) and self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.char2idx = vocab.char2idx
                self.idx2char = vocab.idx2char
                
            print('Char Vocabulary successfully loaded from {} file'.format(self.vocab_file))
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
            
            print('Char Vocabulary succesfully created at {}'.format(self.vocab_file))
            
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        idx = 0
        coco_captions = COCO(self.annotations_file)
        counter = Counter()
        ids = coco_captions.anns.keys()
        
        def add_char(word, idx):
            """Add the given char  token to vocab"""
            for char in word:
                if not char in self.char2idx:
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char
                    idx = idx + 1

                return idx
                
        self.char2idx[self.unk_char] = idx
        self.idx2char[idx] = self.unk_char
        idx = idx + 1
        
        if self.verbose:
            print(f'Current idx: {idx}')
            print('Current char2idx:', self.char2idx)
            
        for i, caption_id in enumerate(ids):
            caption = str(coco_captions.anns[caption_id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if self.verbose and i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))
            
        words = []
        for word, count in counter.items():
            if count >= self.vocab_threshold:
                words.append(word)
                
        for word in words:
            idx = add_char(word, idx)

    def __getitem__(self, char):
        if not char in self.char2idx:  
            return self.char2idx[self.unk_char]
        
        return self.char2idx[char]
    
    def __len__(self):
        return len(self.char2idx)


# # Data Loader

# In[11]:


#TODO: Change transform to optional argument
class CoCoDataset(Dataset):
    def __init__(self, transform, mode, 
                 batch_size, vocab_threshold, vocab_file, 
                 start_word, end_word, unk_word, 
                 annotations_file, vocab_from_file, img_folder, 
                 use_word_vocab=True):
        
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        # if statement for word or char vocab
        self.use_word_vocab = use_word_vocab
        if use_word_vocab:
            self.vocab = WordVocabulary(vocab_threshold, vocab_file, start_word,
              end_word, unk_word, annotations_file, vocab_from_file, verbose=False)
        else:
            self.vocab = CharVocabulary(vocab_threshold, vocab_file, unk_word, annotations_file,
                                        vocab_from_file, verbose=False)
            
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = []
            #TODO: try changing this to not require np.arange?
            for i in tqdm(np.arange(len(self.ids))):
                caption = str(self.coco.anns[self.ids[i]]['caption']).lower()
                tokens = nltk.tokenize.word_tokenize(caption)
                all_tokens.append(tokens)
                
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            with open(annotations_file, 'r') as f:
                test_info = json.loads(f.read())
                self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
      # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            if self.use_word_vocab:
                caption.append(self.vocab[self.vocab.start_word])
                
            caption.extend([self.vocab[token] for token in tokens])
            if self.use_word_vocab:
                caption.append(self.vocab[self.vocab.end_word])
                
            caption = torch.Tensor(caption).long()

          # Convert image to tensor and pre-process using transform
            with Image.open(os.path.join(self.img_folder, path)) as image:
                image = image.convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                                           
                return image, caption

      # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            with Image.open(os.path.join(self.img_folder, path)) as image:
                image = image.convert('RGB')
                orig_image = np.array(image)
                if self.transform is not None:
                    image = self.transform(image)

              # return original image and pre-processed image tensor
                return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)


# In[12]:


#TODO: Pass in imag_folder_path and annotations_file_path
def get_loader(transform,
               img_folder = '../data/images/',
               annotations_file = '../data/annotations/',
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./word_vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               use_word_vocab=True):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
      use_word_vocab: If True we use WordVocabulary, if False, we use CharVocabulary
    """
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if vocab_from_file == False: 
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."
        # Based on mode (train, val, test), obtain img_folder and annotations_file.

    if mode == 'train':
        if vocab_from_file == True:
            assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."

        img_folder = os.path.join(img_folder, 'train2017')
        annotations_file = os.path.join(annotations_file, 'captions_train2017.json')
    elif mode == 'validation':
        assert batch_size == 1, "Please change batch_size to 1 if testing your model." #TODO: LOOK AT THIS????
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        img_folder = os.path.join(img_folder, 'val2017')
        annotations_file = os.path.join(annotations_file, 'captions_val2017.json')

    elif mode == 'test':
        assert batch_size == 1, "Please change batch_size to 1 if testing your model." #TODO: LOOK AT THIS????
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        img_folder = os.path.join(img_folder, 'test2017')
        annotations_file = os.path.join(annotations_file, 'image_info_test2017.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder,
                          use_word_vocab=use_word_vocab)

    
    if mode == 'train':
      # Randomly sample a caption length, and sample indices with that length.
      indices = dataset.get_train_indices()
      # Create and assign a batch sampler to retrieve a batch with the sampled indices.
      initial_sampler = SubsetRandomSampler(indices=indices)
      # data loader for COCO dataset.
      data_loader = DataLoader(dataset=dataset,
                                       num_workers=num_workers,
                                       batch_sampler=BatchSampler(sampler=initial_sampler,
                                                                                  batch_size=dataset.batch_size,
                                                                                  drop_last=False))
    else:
        data_loader = DataLoader(dataset=dataset,
                                         batch_size=dataset.batch_size,
                                         shuffle=True,
                                         num_workers=num_workers)

    return data_loader
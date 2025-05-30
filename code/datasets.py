from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data, gpuId):
    imgs, captions, captions_lens, class_ids, keys, raw_captions, sent_vector, word_vector, image_vector, image_region_vector = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).to(gpuId))
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).to(gpuId)
        sorted_cap_lens = Variable(sorted_cap_lens).to(gpuId)
        sent_vector = sent_vector.to(gpuId)
        word_vector = word_vector.to(gpuId)
        image_vector = image_vector.to(gpuId)
        image_region_vector = image_region_vector.to(gpuId)
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    sent_vector = sent_vector.squeeze()
    word_vector = word_vector.squeeze()
    image_vector = image_vector.squeeze()
    image_region_vector = image_region_vector.squeeze()

    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, raw_captions, sent_vector, word_vector, image_vector, image_region_vector]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


def load_tensor(filepath):
    return torch.load(filepath)


def pad_tensor_to_dim(tensor, dim, target_size):
    current_size = tensor.size(dim)

    if current_size == target_size:
        return tensor
    elif current_size < target_size:
        pad_amount = target_size - current_size
        pad = [0] * (2 * tensor.dim())
        pad_index = 2 * (tensor.dim() - dim - 1)
        pad[pad_index] = pad_amount
        return torch.nn.functional.pad(tensor, pad, "constant", 0)
    else:
        return torch.narrow(tensor, dim, 0, target_size)


class TextDatasetCSV(data.Dataset):
    def __init__(self, csv_path, base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = 1

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        # Load CSV file
        self.data = pd.read_csv(csv_path, sep=';')
        # assert 'image_path' in self.data.columns and 'caption' in self.data.columns, \
        #     "CSV must have 'image_path' and 'caption' columns."

        self.image_paths = self.data['image_path'].tolist()
        self.raw_captions = self.data['caption'].tolist()
        self.sent_vector_paths = self.data['sentence_vector_path'].tolist()
        self.word_vector_paths = self.data['word_vector_path'].tolist()
        self.image_vector_paths = self.data['image_vector_path'].tolist()
        self.image_region_vector_paths = self.data['image_region_vector_path'].tolist()

        # Process all captions into tokenized word indices
        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.build_dictionary(self.raw_captions)
        self.number_example = len(self.image_paths)

        self.base_dir = ""

    def build_dictionary(self, captions_text):
        word_counts = defaultdict(float)
        tokenizer = RegexpTokenizer(r'\w+')

        all_captions = []
        for sent in captions_text:
            tokens = tokenizer.tokenize(sent.lower())
            tokens = [t.encode('ascii', 'ignore').decode('ascii') for t in tokens if t]
            all_captions.append(tokens)
            for word in tokens:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {0: '<end>'}
        wordtoix = {'<end>': 0}
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        captions_indices = []
        for tokens in all_captions:
            caption = [wordtoix[t] for t in tokens if t in wordtoix]
            captions_indices.append(caption)

        return captions_indices, ixtoword, wordtoix, len(ixtoword)

    def get_caption(self, index):
        sent_caption = np.asarray(self.captions[index]).astype('int64')

        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)

        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        img_path = os.path.join(self.base_dir, self.image_paths[index])
        sent_vector_path = os.path.join(self.base_dir, self.sent_vector_paths[index])
        word_vector_path = os.path.join(self.base_dir, self.word_vector_paths[index])
        image_vector_path = os.path.join(self.base_dir, self.image_vector_paths[index])
        image_region_vector_path = os.path.join(self.base_dir, self.image_region_vector_paths[index])
        cls_id = index  # Dummy class ID
        key = os.path.splitext(os.path.basename(img_path))[0]

        imgs = get_imgs(img_path, self.imsize,
                        bbox=None, transform=self.transform, normalize=self.norm)

        sent_vector = load_tensor(sent_vector_path)
        word_vector = load_tensor(word_vector_path)
        word_vector = pad_tensor_to_dim(word_vector, 2, 18)
        image_vector = load_tensor(image_vector_path)
        image_region_vector = load_tensor(image_region_vector_path)

        caps, cap_len = self.get_caption(index)
        raw_caption = self.raw_captions[index]  # Add raw caption text

        return imgs, caps, cap_len, cls_id, key, raw_caption, sent_vector, word_vector, image_vector, image_region_vector

    def __len__(self):
        return len(self.image_paths)

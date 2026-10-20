# Similarity-Ranking-of-Wafer-Defect-Maps
## Abstract
This dataset contains an anonymized subset of proprietary wafer defect maps collected from an industrial semiconductor manufacturing process. The dataset has been prepared to support the reproducibility of the research presented in "Label-Efficient Similarity Ranking of Wafer Defect Maps using Domain-motivated Contrastive Learning." Due to confidentiality agreements associated with the industrial collaboration, the original production dataset cannot be publicly released. Therefore, this dataset includes representative anonymized wafer maps that preserve the structural characteristics of real manufacturing defects while removing all information that could reveal proprietary process details.

Unlike publicly available wafer map datasets such as WM811K and MixedWM38, this dataset is not intended for wafer defect pattern classification. Instead, it is designed for wafer map similarity estimation and retrieval, where similarity is determined by expert process engineers based on both visual characteristics and process-related domain knowledge. The dataset consists of representative target wafer maps and their corresponding candidate wafer maps used for similarity evaluation.

The released dataset contains the anonymized labeled wafer map sets used in this study, including those for model fine-tuning and performance evaluation. The complete industrial dataset used for pre-training cannot be publicly released due to confidentiality restrictions. This dataset is provided to support independent verification of the reported experimental results while preserving the confidentiality of the proprietary industrial data.

## Instructions:
```
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, torch, random
import pickle
import cv2
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class bin_dataloaders(Dataset):
    def __init__(self, opt, epoch):
        self.epoch = epoch
        self.mode = opt.mode
        self.img_size = opt.img_size
        self.labeled_img = np.load('./imgs.npy') # ndarray: (200, 3, 32, 32)
        h, w = self.labeled_img[0].shape[1:]
        self.img_all = np.array(self.labeled_img).reshape(-1,h,w)
        self.consensus_th = 2
        self.num_pool = 30
        with open('./posidx.pkl', 'rb') as f:
            self.train_df = pickle.load(f)  # DataFrame: (166, 5)
            
        self.transforms_bin = transforms.Compose([
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.05, 0.05), shear=(3,3))], p=.3),
            transforms.GaussianBlur((5,5), sigma=2.0),
            transforms.ToTensor()
        ])

        self.channel = 3
        self.batch_size = opt.bs   # even number
        self.p_top_k = int(opt.n_pos_neg / 4) - 1

        self.idx_sampling = opt.idx_sampling
        self.seed = opt.seed
        self.save_path = opt.save_path

        self.window_h = (3, 10)    # (min, max)
        self.window_w = (3, 10)
        self.n_roi = (3, 5)

    def resize_img(self, img, sz=128):
        if sz < 128:
            resized_im = cv2.resize(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (sz, sz),
                                    interpolation=cv2.INTER_AREA)
        else:
            resized_im = cv2.resize(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), (sz, sz),
                                    interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(resized_im, cv2.COLOR_BGR2GRAY)
        return img / 255

    def compute_pixel_similarity(self, imgA, candi, sigma=2.0):
        """BIoM: blurred intersection over maximum (Eq. 5 in the paper)."""
        ksize = 5
        imgB = np.array([self.img_all[j, :, :] for j in candi])

        blurred_A = cv2.GaussianBlur((imgA * 255).astype(np.uint8), (ksize, ksize), sigma) / 255.0
        blurred_B = np.stack([
            cv2.GaussianBlur((b * 255).astype(np.uint8), (ksize, ksize), sigma) / 255.0
            for b in imgB
        ])

        intersect = np.minimum(blurred_A[np.newaxis], blurred_B).sum(axis=(1, 2))
        sum_A = blurred_A.sum()
        sum_B = blurred_B.sum(axis=(1, 2))
        score = intersect / (np.maximum(sum_A, sum_B) + 1e-6)

        return score

    def search_neg_img(self, target, n_neg):
        n_neg_pool = 2000
        score_th_neg = 0.2

        candi = np.random.choice(len(self.img_all), n_neg_pool, replace=False)
        score_array = self.compute_pixel_similarity(target, candi)
        neg_idx = np.delete(candi, np.where(score_array > score_th_neg))[:n_neg]
        neg_img = [self.img_all[k,:,:] for k in neg_idx]

        return neg_img

    def __getitem__(self, idx):
        img_idx = self.train_df.loc[idx].target_idx
        n_pos_sample = self.batch_size // 2  - 1 # num. of positive sample
        n_neg_sample = self.batch_size - n_pos_sample -1

        pos_idx = self.train_df.positive_idx.loc[idx][self.consensus_th - 1]
        if len(pos_idx) < n_pos_sample:
            pos_sup_idx = random.choices(pos_idx, k=n_pos_sample - len(pos_idx))
            _pos_idx = pos_idx + pos_sup_idx
        else:
            _pos_idx = pos_idx[:n_pos_sample]
        neg_idx = np.arange(self.num_pool)[np.isin(np.arange(self.num_pool), pos_idx, invert=True)]

        #### if the num. of neg_idx is less than a certain num., we search the entire dataset for neg_idx
        if len(neg_idx) < self.batch_size // 4:
            imgA = self.labeled_img[img_idx][0] # target
            _neg_sup_img = self.search_neg_img(imgA, n_neg_sample - len(neg_idx))
            neg_sup_img = [self.resize_img(_neg_sup_img[k], sz=self.img_size) for k in range (len(_neg_sup_img))]
            _neg_img = [self.resize_img(self.labeled_img[img_idx][k+1], sz=self.img_size) for k in neg_idx]
            neg_img = _neg_img + neg_sup_img
        else:
            _neg_idx = random.choices(neg_idx, k=n_neg_sample)
            neg_img = [self.resize_img(self.labeled_img[img_idx][k+1], sz=self.img_size) for k in _neg_idx]

        target_img = self.resize_img(self.labeled_img[img_idx][0], sz=self.img_size)
        pos_img = [self.resize_img(self.labeled_img[img_idx][k+1], sz=self.img_size) for k in _pos_idx]

        input_imgs = (np.concatenate([[target_img], pos_img, neg_img], axis=0) * 255).astype("uint8")  # [batch size, h, w] (P,...P, N,...N), np.array

        input_tensor = []
        angle = np.random.randint(180)
        for i in range (len(input_imgs)):
            rotated_input = transforms.functional.rotate(Image.fromarray(input_imgs[i]), angle)
            _input_tensor = self.transforms_bin(rotated_input)
            input_tensor.append(torch.cat([_input_tensor, _input_tensor, _input_tensor], axis=0))

        return torch.stack(input_tensor, dim=0)

    def __len__(self):
        return len(self.train_df)
```

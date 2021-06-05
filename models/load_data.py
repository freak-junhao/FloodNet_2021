import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import numpy as np
import os.path as opth
import json
import cv2
from PIL import Image

from models.data_utils import load_image, load_image_feat, proc_ques, pre_emb_load


class VqaDataset(Dataset):
    def __init__(self, config):
        self.config = config

        # load image path
        self.image_dict = {}
        # image_path_list = glob(config['image_path'] + '*.JPG')
        image_path_list = glob(config['image_path'] + '*')
        for img_path in image_path_list:
            _, image_index = opth.split(img_path)
            self.image_dict[image_index] = img_path

        # load questions
        with open(config['question_path'], 'r') as ans_file:
            if config['run_mode'] == 'train':
                # ans_file = open(config['question_path'], 'r')
                questions_info = json.load(ans_file)
            else:
                # ans_file = open(config['question_path'], 'r')
                questions_info = json.load(ans_file)
                for info in questions_info:
                    questions_info[info]['Ground_Truth'] = 'None'

        self.question_lst = questions_info

        # create questions and answers list
        # self.question_lst = []
        # self.question_type_lst = []
        # self.answer_lst = []
        # self.image_lst = []
        #
        # for info in questions_info:
        #     self.question_lst.append(questions_info[info]['Question'])
        #     self.question_type_lst.append(questions_info[info]['Question_Type'])
        #     self.answer_lst.append(questions_info[info]['Ground_Truth'])
        #     self.image_lst.append(questions_info[info]['Image_ID'])

        # Define run data size
        self.data_size = self.question_lst.__len__()
        print('== Dataset size:', self.data_size)

        # {image_path} -> {image}
        _size = self.config['image_size']
        if not config['use_npz']:
            self.id_to_image = load_image(self.image_dict, _size, config['data_workers'])
        else:
            self.id_to_image_feat = load_image_feat(self.image_dict)

        # load questions and answers token
        with open(config['token_path'], 'r') as token_file:
            self.question_token, self.answer_token, self.ans_ix = json.load(token_file)
        self.token_size = self.question_token.__len__()
        self.ans_size = self.answer_token.__len__()

        # load pretrain embedding weight
        self.pre_emb = pre_emb_load(self.question_token)

    def __getitem__(self, idx):
        ques_ans = self.question_lst.get(str(idx))
        image_id = ques_ans.get('Image_ID')
        # _size = self.config['image_size']

        # get image
        if not self.config['use_npz']:
            image_data = self.id_to_image.get(image_id)

            trans = transforms.Compose([
                # transforms.Resize((_size, _size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.config['mean'], self.config['std'])
            ])
            image_data = trans(image_data)

        else:
            image_data = self.id_to_image_feat.get(image_id)
            image_data = torch.from_numpy(image_data)

        # get question
        ques = ques_ans.get('Question')
        ques_ix = proc_ques(ques, self.question_token, self.config['max_token'])
        ques_ix = torch.from_numpy(ques_ix)

        ans = str(ques_ans.get('Ground_Truth'))
        # ans_ix = torch.LongTensor([self.answer_token.get(ans)])

        # label to one-hot
        ans_ix = self.answer_token.get(ans)
        one_hot = np.zeros(self.ans_size, np.float32)
        one_hot[ans_ix] = 1
        ans_ix = torch.from_numpy(one_hot)

        return image_data, ques_ix, ans_ix

    def __len__(self):
        return self.data_size


class TransDataset(Dataset):
    def __init__(self, config):
        self.config = config

        # load image path
        self.image_dict = {}
        # image_path_list = glob(config['image_path'] + '*.JPG')
        image_path_list = glob(config['image_path'] + '*')
        for img_path in image_path_list:
            _, image_index = opth.split(img_path)
            self.image_dict[image_index] = img_path

        # load questions
        with open(config['question_path'], 'r') as ans_file:
            if config['run_mode'] == 'train':
                # ans_file = open(config['question_path'], 'r')
                questions_info = json.load(ans_file)
            else:
                # ans_file = open(config['question_path'], 'r')
                questions_info = json.load(ans_file)
                for info in questions_info:
                    questions_info[info]['Ground_Truth'] = 0

        self.question_lst = questions_info

        # Define run data size
        self.data_size = self.question_lst.__len__()
        print('== Dataset size:', self.data_size)

        # {image_path} -> {image}
        _size = self.config['image_size']
        if not config['use_npz']:
            self.id_to_image = load_image(self.image_dict, _size, config['data_workers'])
        else:
            self.id_to_image_feat = load_image_feat(self.image_dict)

        # # load questions and answers token
        # with open(config['token_path'], 'r') as token_file:
        #     self.question_token, self.answer_token, self.ans_ix = json.load(token_file)
        # self.token_size = self.question_token.__len__()
        # self.ans_size = self.answer_token.__len__()
        #
        # # load pretrain embedding weight
        # self.pre_emb = pre_emb_load(self.question_token)

    def __getitem__(self, idx):
        ques_ans = self.question_lst.get(str(idx))
        image_id = ques_ans.get('Image_ID')
        # _size = self.config['image_size']

        # get image
        if not self.config['use_npz']:
            image_data = self.id_to_image.get(image_id)
            trans = transforms.Compose([
                # transforms.Resize((_size, _size)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.config['mean'], self.config['std'])
            ])
            image_data = trans(image_data)

        else:
            image_data = self.id_to_image_feat.get(image_id)
            image_data = torch.from_numpy(image_data)

        # label process
        ques = ques_ans.get('Question')
        if 'non flooded' in ques:
            class_label = np.array([1, 0, 0], dtype=np.float32)
        elif 'flooded' in ques:
            class_label = np.array([0, 1, 0], dtype=np.float32)
        else:
            class_label = np.array([0, 0, 1], dtype=np.float32)
        class_label = torch.from_numpy(class_label)

        num_label = np.array(ques_ans.get('Ground_Truth'), dtype=np.float64)
        num_label = torch.LongTensor(num_label)

        return image_data, num_label, class_label

    def __len__(self):
        return self.data_size

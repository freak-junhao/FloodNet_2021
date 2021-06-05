from PIL import Image, ImageFile
import cv2
import numpy as np
from tqdm import tqdm
import re, en_vectors_web_lg
from multiprocessing import Pool

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_image(image_path, _size):
    id_to_image = {}
    for image_info in tqdm(image_path):
        path = image_path.get(image_info)
        # image = cv2.imread(path)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = Image.open(path)
        # resize
        image = image.resize((_size, _size), Image.ANTIALIAS)

        # image_arr = np.array(image)
        id_to_image[image_info] = image
    return id_to_image


def load_image(image_path, _size, pids_num):
    # split dict
    new_dict = []
    image_path_lst = list(image_path.items())
    for i in range(pids_num):
        if i == pids_num - 1:
            new_dict.append(
                dict(image_path_lst[len(image_path) // pids_num * i:])
            )
        else:
            new_dict.append(
                dict(
                    image_path_lst[len(image_path) // pids_num * i:len(image_path) // pids_num * (i + 1)]
                )
            )

    # multiprocessing mode
    image_pool = Pool(pids_num)
    multi_results = []
    for i in range(pids_num):
        multi_results.append(image_pool.apply_async(get_image, args=(new_dict[i], _size)))

    r = []
    for result in multi_results:
        r.append(result.get())
    image_pool.close()
    image_pool.join()

    # combine results
    id_to_image = {}
    for i in range(pids_num):
        id_to_image.update(r[i])

    return id_to_image


def load_image_feat(image_path):
    return {}


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def pre_emb_load(token):
    pretrained_emb = []
    spacy_tool = en_vectors_web_lg.load()
    # pretrained_emb.append(spacy_tool('PAD').vector)
    # pretrained_emb.append(spacy_tool('UNK').vector)

    for word in token:
        pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)
    return pretrained_emb

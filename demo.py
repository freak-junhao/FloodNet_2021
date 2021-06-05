import cv2
from PIL import Image
import numpy as np
import json
from copy import deepcopy


def make_token():
    ques_token = ['be', 'image', 'buildings', 'flooded', 'entire', 'given', 'How', 'overall', 'road', 'many', 'seen',
                  'condition', 'What', 'Is', 'of', 'non', 'are', 'the', 'this', 'can', 'in', 'is', 'UNK']
    q = {}
    for i in range(len(ques_token)):
        q[ques_token[i]] = i

    ans_ix = ['flooded,non flooded', 'flooded', 'non flooded', 'Yes', 'No']
    ans = {}
    for i in range(55):
        if i < 5:
            ans[ans_ix[i]] = i
        else:
            ans[str(i - 4)] = i

    ix = {}
    for i in ans:
        ix[ans.get(i)] = i

    with open('qu_token.json', 'w') as f:
        json.dump([q, ans, ix], f)

    with open('./results/answers/answer.txt', 'r') as f:
        ans = f.readlines()
        ans = [i.strip() for i in ans]

    with open('./results/answers/lr0001.txt', 'r') as f:
        count = f.readlines()
        count = [i.strip() for i in count]

    with open('results/questions/Valid_count_raw.json') as f:
        dataset = json.load(f)

    order = 0
    for i in dataset:
        index = int(i)
        ans[index] = count[order]
        order += 1

    with open('./results/answers/ans_count.txt', 'w') as f:
        for i in ans:
            f.write(i + '\n')


def add_json(path, data, count):
    with open(path, 'r') as f:
        add = json.load(f)

    for i in add:
        data[str(count)] = add.get(i)
        count += 1

    return data, count


def combine_json():
    with open('results/questions/Train_count.json', 'r') as f:
        raw_data = json.load(f)

    path = [
        'results/questions/Train_count_90.json',
        'results/questions/Train_count_180.json',
        'results/questions/Train_count_270.json',
        'results/questions/Train_count_LRF.json',
        'results/questions/Train_count_TBF.json'
    ]

    count = raw_data.__len__()
    for i in path:
        raw_data, count = add_json(i, raw_data, count)

    with open('results/questions/Train_count_large.json', 'w') as f:
        json.dump(raw_data, f)


def enlarge_questions():
    with open('results/questions/Training Question.json', 'r') as f:
        raw_qu = json.load(f)

    new_qu = deepcopy(raw_qu)
    # new_index = ['90X', '180X', '270X', 'LRF', 'TBF']
    new_index = ['90X', '180X', '270X']

    count = raw_qu.__len__()
    for i in raw_qu:
        info = deepcopy(raw_qu.get(i))
        image_id = info.get("Image_ID")

        for j in new_index:
            new_id = j + image_id
            info["Image_ID"] = new_id
            new_qu[str(count)] = deepcopy(info)
            count += 1

    with open('results/questions/Training Question_large.json', 'w') as f:
        json.dump(new_qu, f)


def extract_count():
    with open('/data/njh/data/Questions/Test_Question.json', 'r') as f:
        raw_qu = json.load(f)

    new_qu = {}
    for i in raw_qu:
        ques_type = raw_qu.get(i)
        if ques_type['Question_Type'] == 'Simple_Counting' or ques_type['Question_Type'] == 'Complex_Counting':
            new_qu[i] = ques_type

    with open('results/questions/Test_count_raw.json', 'w') as f:
        json.dump(new_qu, f)

    count = 0
    sort_qu = {}
    for i in new_qu:
        info = new_qu.get(i)
        sort_qu[str(count)] = info
        count += 1

    with open('results/questions/Test_count.json', 'w') as f:
        json.dump(sort_qu, f)


def combine_answer():
    with open('/data/njh/pythonProject/results/answers/_mcan_efb4_0507.txt', 'r') as f:
        answer = f.readlines()

    with open('/data/njh/pythonProject/results/answers/_se_count_0511.txt', 'r') as f:
        num = f.readlines()

    with open('/data/njh/pythonProject/results/questions/Test_count_raw.json', 'r') as f:
        raw_qu = json.load(f)

    count = 0
    for i in raw_qu:
        answer[int(i)] = num[count]
        count += 1

    with open('/data/njh/pythonProject/results/answers/_answer.txt', 'w') as f:
        for i in answer:
            f.write(i)


if __name__ == '__main__':
    combine_answer()
    print('Done.')

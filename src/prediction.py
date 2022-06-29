import torch
import numpy as np
import pandas as pd
import heapq

from models import AllergyClassifierModel
from system import AllergyClassifier

def index_list(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

def hoge():
    import cv2 # not recommended
    img_path = "../img/sample_food.jpeg"
    img = cv2.imread(img_path)
    return img

def main():
    model = AllergyClassifierModel(
        weight_file='../food-101/meta/weights.csv'
    )
    classifier = AllergyClassifier(model)

    ckpt = torch.load("../logs/epoch=0-step=1263-v1.ckpt")
    classifier.load_state_dict(ckpt['state_dict'])
    classifier.eval()

    # データを入れる部分
    # --------------
    from datasets import FoodDataModule
    dataset = FoodDataModule(
        data_dir='../food-101/images/',
        ann_dir='../food-101/meta/',
        class_file='../food-101/meta/classes.txt',
        weight_file='../food-101/meta/weights.csv',
        batch_size=16
    )
    dataset.setup()
    img, label = dataset.train_dataloader().__iter__().next()[0][0], dataset.train_dataloader().__iter__().next()[1][0]
    # img = hoge()
    # ----------------

    with torch.no_grad():
        # output = classifier(img.unsqueeze(0))
        # output_f = classifier(img.unsqueeze(0))
        output_a = classifier(img.unsqueeze(0))
    # print(output.argmax(), label)
    # print(output)


    table_path = "../food-101/meta/table.csv"
    # table_path = "../table.csv"
    table = pd.read_csv(table_path)
    table_column = list(table.columns)[1:]
    table_index = list(table["table"])
    # print(table_columun)

    # output_f = classifier.model(img.unsqueeze(0))
    # output_f = np.random.rand(101)
    # output_a = np.random.rand(27)
    output_a = output_a.tolist()[0]
    # output_f = output_f.tolist()
    print(output_a)



    possible_foods_list = []
    possible_allergen_list = [] 
    n = 3
    temp = 0

    # # if output is only food label
    # for value in (heapq.nlargest(n, output_f)):
    #     possible_foods_list.append([list(output_f).index(value), table_column[list(output_f).index(value)] ,value])

    # if output is only allergen label
    for value in (heapq.nlargest(n, output_a)):
        if value == temp:
            break
        else:
            for ind in index_list(output_a, value):
                # print(ind)
                possible_allergen_list.append([ind, table_column[ind] ,value])
        temp = value

    # # if output is both food and allergen label
    # for value_f in (heapq.nlargest(n, output_f)):
    #     possible_foods_list.append([list(output_f).index(value_f), table_column[list(output_f).index(value_f)] ,value_f])
    # for value_a in (heapq.nlargest(n, output_a)):
    #     possible_allergen_list.append([list(output_a).index(value_a), table_columun[list(output_a).index(value_a)] ,value_a])

    # print(possible_foods_list)
    print(possible_allergen_list)
    
    return


if __name__ == '__main__':
    main()
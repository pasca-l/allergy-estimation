import torch
import numpy as np
import pandas as pd
import heapq
import cv2
from torchvision import transforms

from datasets import FoodImageTransform
from models import AllergyClassifierModel
from system import AllergyClassifier


class Predictor():
    def __init__(self):
        self.weight_file = "../food-101/meta/weights.csv"
        self.model = AllergyClassifierModel(
            weight_file=self.weight_file
        )
        self.classifier = AllergyClassifier(self.model)
        self.ckpt = torch.load("../logs/epoch=0-step=1263-v1.ckpt")
        self.classifier.load_state_dict(self.ckpt['state_dict'])
        self.classifier.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        self.table = pd.read_csv(self.weight_file)
        self.allergen_list = np.array(self.table.columns)[1:]
        self.food_list = np.array(self.table["table"])
        self.weights = torch.tensor(np.loadtxt(self.weight_file, delimiter=',', skiprows=1, 
                usecols=range(1, 28), dtype='float32'))

    def index_list(self, l, x):
        return [i for i, _x in enumerate(l) if _x == x]

    def load_sample_img(self):
        # img_path = "../img/sample_food.jpeg"
        img_path = "../img/36641.jpg"
        img = cv2.imread(img_path)
        return img

    def list_sort(self, output, hoge_list):
        sorted_output = np.sort(output)[::-1]

        sorted_list = hoge_list[np.argsort(output)][::-1]
        possible_list = []
        for ind in range(len(output)):
            possible_list.append([np.argsort(output)[ind], sorted_list[ind], sorted_output[ind]])
        return possible_list


    def predict(self, img, output="a", debug=False):
        if debug:
            img = self.load_sample_img()
        img = self.transform(img)

        with torch.no_grad():
            if output == "a":
                output_a = self.classifier(img.unsqueeze(0))
                
            elif output == "f":
                output_f = self.classifier.model.forward_demo(img.unsqueeze(0))
                output_a = torch.mm(output_f, torch.tensor(self.weights))
            else:
                print("output should be \"a\" or \"f\"")
                return

        # n = 3

        if output == "a":
            output_a = output_a.tolist()[0]
            possible_allergen_list = self.list_sort(output_a, self.allergen_list)
            print(possible_allergen_list)
            return possible_allergen_list

        elif output == "f":
            output_f = output_f.tolist()[0]
            output_a = output_a.tolist()[0]
            possible_foods_list = self.list_sort(output_f, self.food_list)
            possible_allergen_list = self.list_sort(output_a, self.allergen_list)
            print(possible_foods_list[:5])
            print(possible_allergen_list[:5])
            return possible_foods_list, possible_allergen_list

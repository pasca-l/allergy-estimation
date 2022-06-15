import pytorch_lightning as pl
import cv2
from src.models import AllergyClassifierModel
from src.system import AllergyClassifier
# from app_ui import hoge

# temp
def hoge():
    img_path = "../img/sample_food.jpeg"
    img = cv2.imread(img_path)
    return img


def main():
    model = AllergyClassifierModel()
    classifier = AllergyClassifier(model)
    ckpt = "../ckpt/epoch=0-step=1263.ckpt"
    img= hoge()
    
    # write code of prediction like "classifier.predict(ckpt_path = ckpt)"


if __name__ == "__main__":
    main()
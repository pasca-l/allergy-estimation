import cv2
import numpy as np
import matplotlib.pyplot as plt

from prediction import Predictor


def main():
    img_path = "../img/36641.jpg"
    img = cv2.imread(img_path)

    weight_file_path = '../meta/weights.csv'
    ckpt_file_path = '../logs/epoch=00-val_loss=4.62.ckpt'

    p = Predictor(
        weight_file=weight_file_path, 
        ckpt_file=ckpt_file_path
    )

    def put_text(frame, text, position, color, scale):
        cv2.putText(
            frame,
            text=text,
            org=position,
            color=color,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            thickness=2,
            lineType=cv2.LINE_4
        )

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            food_name, food_prob, allergy_name, allergy_prob = p.predict(frame)

            put_text(frame, food_name[0], (30, 40), (0, 0, 255), 1)
            for i in range(27):
                result = f"{allergy_name[i]:<10} : {allergy_prob[i]}"
                put_text(frame, result, (30, 80 + 30 * i), (0, 255, 0), 0.7)

            cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("w"):
            cv2.imwrite("../img/image.png", frame)


if __name__ == '__main__':
    main()
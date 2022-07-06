import os
import cv2
import numpy as np

from prediction import Predictor


def main():
    weight_file_path = '../meta/weights.csv'
    ckpt_file_path = '../logs/trained_model.ckpt'

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
    bg_flag = True
    while True:
        ret, frame = cap.read()
        if ret:
            food_name, food_prob, allergy_name, allergy_prob = p.predict(frame)

            if bg_flag:
                mask = frame.copy()
                cv2.rectangle(mask, (0,0), (450,1200), (255,0,0), thickness=-1)
                frame = cv2.addWeighted(mask, alpha:=0.4, frame, 1-alpha, 0)

            for i in range(5):
                result = f"{food_prob[i]*100:.1f}% {food_name[i]:<15}"
                put_text(frame, result, (30,40+40*i), (0,0,255), 1)

            for i in range(len(allergy_prob)):
                put_text(frame, f"{allergy_name[i]}", 
                         (30,250+30*i), (0,255,0), 0.7)
                put_text(frame, f": {allergy_prob[i]*100:.1f}%",
                         (180,250+30*i), (0,255,0), 0.7)

            cv2.imshow("Allergen Estimation", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("w"):
            os.makedirs("../screenshots/", exist_ok=True)
            cv2.imwrite("../screenshots/image.png", frame)
        if key == ord("f"):
            bg_flag = not bg_flag


if __name__ == '__main__':
    main()
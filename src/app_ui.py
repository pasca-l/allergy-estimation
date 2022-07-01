import cv2
import numpy as np
import matplotlib.pyplot as plt

from app_prediction import Predictor


def main():
    p = Predictor()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        mask = np.ones(frame.shape, dtype = "uint8") * 10
        frame += mask

        result = p.predict(frame)
        cv2.putText(frame,
            # text=f'possible_foods_dict{rand}',
            text = f'text{result}',
            org=(150, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("w"):
            cv2.imwrite("./image.png", frame)


if __name__ == '__main__':
    main()
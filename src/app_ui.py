import cv2
import numpy as np
import matplotlib.pyplot as plt

from app_prediction import Predictor


def main():
    img_path = "../img/36641.jpg"
    img = cv2.imread(img_path)

    p = Predictor(
        weight_file='../meta/weights.csv', 
        ckpt_file='../logs/imagecls_epoch=02-val_loss=4.62.ckpt'
    )
    _, a, b = p.predict(img)
    print(a + b.astype('str'))
    return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            # result = p.predict(frame)
            cv2.putText(frame,
                # text=f'possible_foods_dict{rand}',
                text = f'text',#{result}',
                org=(150, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)

            cv2.rectangle(frame, (50, 10), (125, 60), (255, 0, 0), thickness=-1)

            cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("w"):
            cv2.imwrite("./image.png", frame)


if __name__ == '__main__':
    main()
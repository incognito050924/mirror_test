import tensorflow as tf
import cv2
import numpy as np
from services.analysis import CascadeDetector
from services.image_processor import resize_image
from ml.predictor import predict_emotion
print('Importing Completed')


def predict(input_img):
    # gray_face = CascadeDetector().detect_face(input_img, use_gray=True, visible=False)
    # face = resize_image(gray_face, (299, 299, 1))
    # face = np.reshape(face, (299, 299, 1))
    # cv2.imwrite('sample.jpg', face)
    input_img = np.reshape(input_img, (299, 299, 1))
    face = np.expand_dims(input_img, 0)
    print(face.shape, face.dtype)

    emotion, score = predict_emotion(face, text_label=True)

    return emotion, score


img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
print(predict(img))

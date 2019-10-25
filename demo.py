"""
aioz.aiar.truongle - May 06, 2019
head pose detection with FSANET
"""
import cv2
import os
from src.fsanet_wrapper import FsanetWrapper
from src.config import Config
from mtcnn.mtcnn import MTCNN
from termcolor import colored
import numpy as np
import math


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def draw_axis(img, param, tdx=None, tdy=None, size=80):
    yaw = param[0]
    pitch = param[1]
    roll = param[2]
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (math.sin(yaw)) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def draw_res(input_img, boxes, params, ratio):
    img_h, img_w, _ = np.shape(input_img)
    if len(boxes) > 0:
        for i, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb
            w = xmax - xmin
            h = ymax - ymin
            xw1 = max(int(xmin - ratio * w), 0)
            yw1 = max(int(ymin - ratio * h), 0)
            xw2 = min(int(xmax + ratio * w), img_w - 1)
            yw2 = min(int(ymax + ratio * h), img_h - 1)
            img = draw_axis(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], params[i])
            input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img

    return input_img


def take_faces(detected, input_img, threshold, im_size, ratio):
    boxes = []
    if len(detected) > 0:
        for i, d in enumerate(detected):
            if d['confidence'] > threshold:
                xmin, ymin, w, h = d['box']
                xmax, ymax = xmin + w, ymin + h
                boxes.append([xmin, ymin, xmax, ymax])

    faces = np.empty((len(boxes), im_size, im_size, 3))
    img_h, img_w, _ = np.shape(input_img)

    if len(boxes) > 0:
        for i, bb in enumerate(boxes):
            xmin, ymin, xmax, ymax = bb
            w = xmax - xmin
            h = ymax - ymin
            xw1 = max(int(xmin - ratio * w), 0)
            yw1 = max(int(ymin - ratio * h), 0)
            xw2 = min(int(xmax + ratio * w), img_w - 1)
            yw2 = min(int(ymax + ratio * h), img_h - 1)
            faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (im_size, im_size))
            faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return faces, boxes


def main():
    detector = MTCNN()
    config = Config()
    fsanet = FsanetWrapper(graph=config.graph_fsanet)

    input_img = cv2.imread("images/test.jpg")
    detected = detector.detect_faces(input_img)
    faces, boxes = take_faces(detected, input_img, config.threshold, config.image_size, config.ratio)
    # Detect head pose
    params = fsanet.predict(images=faces)

    # Write result
    img = draw_res(input_img, boxes, params, config.ratio)
    save_path = "images/result.jpg"
    cv2.imwrite(save_path, img)
    print(colored("[INFO] write result at: {}".format(save_path), "green", attrs=['bold']))


if __name__ == '__main__':
    main()

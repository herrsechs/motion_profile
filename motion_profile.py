#encoding=utf-8
import cv2 as cv
from moviepy.editor import VideoFileClip
import numpy as np
import math
import os
import time


class MPGenerator:
    def __init__(self, videoPth, saveImgFolder='', duration=None):
        """
        @para videoPth path of video
        @saveImgFolder img save path
        """

        self.video = VideoFileClip(videoPth)
        if duration is not None:
            self.video = self.video.subclip(duration[0], duration[1])
        self._saveImgFolder = saveImgFolder

    def generate(self, y1, y2, fileName, mean=True):
        """
        this generate two files mainly, an image for viewing convieniently
        @para y1 lower-bound
                采样的下边界
        @para y2 upper-bound
                采样的上边界
        @para fileName name of image to be saved eg."o1.jpg"
                生成运动轮廓图的保存文件名
        @para mean  boolean, compute mean value of belt or not
                布尔值，是否计算采样带的均值
        """
        w = self.video.size[0]
        frame_count = self.video.duration * self.video.fps
        if mean:
            blank_img = np.zeros((int(frame_count)+1, w, 3), np.uint8)

            idx = 0
            for frame in self.video.iter_frames():
                blank_img[idx, :, 0] = np.mean(frame[y1:y2, :, 0], axis=0)
                blank_img[idx, :, 1] = np.mean(frame[y1:y2, :, 1], axis=0)
                blank_img[idx, :, 2] = np.mean(frame[y1:y2, :, 2], axis=0)
                idx += 1
        else:
            belt_width = y2 - y1
            blank_img = np.zeros(((int(frame_count) + 1) * belt_width, w, 3), np.uint8)
            idx = 0
            for frame in self.video.iter_frames():
                blank_img[idx:idx+belt_width, :, :] = frame[y1:y2, :, :]
                idx += belt_width

        cv.imwrite(os.path.join(self._saveImgFolder, fileName), blank_img)


VIDEO_PATH = r'PICT9620.MP4'
OUTPUT_FILE_NAME = 'PICT9620_mean.jpg'
# 采样带上边界
UPPER_BOUND = 520
# 采样带下边界
LOWER_BOUND = 510
# 是否对采样带取均值
MEAN = True

mpGen = MPGenerator(VIDEO_PATH)
print("****** Pocessing " + OUTPUT_FILE_NAME + " ******")
mpGen.generate(LOWER_BOUND, UPPER_BOUND, OUTPUT_FILE_NAME, mean=MEAN)
print("****** " + OUTPUT_FILE_NAME + " complete! ******")

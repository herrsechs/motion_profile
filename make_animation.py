import cv2
import numpy as np

def make_short_animation():
    VIDEO_PATH = 'E:/video/1790/1790.avi'
    ANIMATION_PATH = 'E:/video/1790/1790_motion.avi'

    near_motion = cv2.imread('E:/video/1790frame/near.png')
    mid_motion = cv2.imread('E:/video/1790frame/mid.png')
    far_motion = cv2.imread('E:/video/1790frame/far.png')

    mid_motion = cv2.resize(mid_motion, (near_motion.shape[1], near_motion.shape[0]))
    far_motion = cv2.resize(far_motion, (near_motion.shape[1], near_motion.shape[0]))

    video = cv2.VideoCapture(VIDEO_PATH)
    frameCount = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    m_height_per_frame = int(near_motion.shape[0] / frameCount)

    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    success, frame = video.read()
    frame_number = 1

    v_writer = cv2.VideoWriter(ANIMATION_PATH,
                               -1,
                               fps, (near_motion.shape[1] * 2, frame.shape[0]))

    while success:
        frame = frame[:, :near_motion.shape[1], :]
        motion_part = np.zeros(frame.shape, dtype=np.uint8)
        motion_part[:(frame_number*m_height_per_frame), :, :] = \
            far_motion[:(frame_number*m_height_per_frame), :, :]
        motion_part[int(0.33*motion_part.shape[0]):int(0.33*motion_part.shape[0])+frame_number*m_height_per_frame, :, :] = \
            mid_motion[:(frame_number*m_height_per_frame), :, :]
        motion_part[int(0.67 * motion_part.shape[0]):int(0.67 * motion_part.shape[0])+frame_number * m_height_per_frame,
                    :, :] = near_motion[:(frame_number * m_height_per_frame), :, :]

        motion_frame = np.hstack((frame, motion_part))

        # cv2.imshow('win', motion_part)
        # cv2.waitKey(0)
        v_writer.write(motion_frame)
        success, frame = video.read()
        frame_number += 1


def make_long_animation():
    VIDEO_PATH = 'E:/video/1790/target_clip.avi'
    ANIMATION_PATH = 'E:/video/1790/target_motion.avi'

    motion = cv2.imread('E:/video/1790frame/far.png')

    video = cv2.VideoCapture(VIDEO_PATH)
    frameCount = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    m_height_per_frame = int(motion.shape[0] / frameCount)

    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    success, frame = video.read()
    frame_number = 1

    v_writer = cv2.VideoWriter(ANIMATION_PATH,
                               -1,
                               fps, (motion.shape[1] * 2, frame.shape[0]))

    while success:
        motion_part = np.zeros(frame.shape, dtype=np.uint8)
        if frame_number * m_height_per_frame < size[1]:
            motion_part[:(frame_number * m_height_per_frame), :, :] = \
                motion[:(frame_number * m_height_per_frame), :, :]
        else:
            motion_part[:, :, :] = \
                motion[(frame_number * m_height_per_frame-size[1]):(frame_number * m_height_per_frame), :, :]
        motion_frame = np.hstack((frame, motion_part))

        # cv2.imshow('win', motion_part)
        # cv2.waitKey(0)
        v_writer.write(motion_frame)
        success, frame = video.read()
        frame_number += 1


make_long_animation()

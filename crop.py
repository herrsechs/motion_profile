import os
import cv2
TOTAL_MOTION_PATH = r'E:\video\total_motion_profile\total'
OUTPUT_PATH = r'D:\journal_paper\research_part_C\data\2_6'
TOTAL_TIME = 12.0


def crop_motion_profile(start_time, end_time):
    for f in os.listdir(TOTAL_MOTION_PATH):
        img = cv2.imread(os.path.join(TOTAL_MOTION_PATH, f))
        h = img.shape[0]
        h1 = int((start_time / TOTAL_TIME) * h)
        h2 = int((end_time / TOTAL_TIME) * h)
        cv2.imwrite(os.path.join(OUTPUT_PATH, f), img[h1:h2, :])


if __name__ == '__main__':
    crop_motion_profile(2, 6)

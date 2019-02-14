from moviepy.editor import VideoFileClip
from darknet import detect
from darknet import load_net_custom, load_meta
from skimage import draw, io
import numpy as np
import os

configPath = './cfg/yolov3.cfg'
weightPath = 'yolov3.weights'
metaPath = './cfg/coco.data'


class YMGenerator:
    def __init__(self, videoPth, saveImgFolder='', duration=None):
        self.video = VideoFileClip(videoPth)
        if duration is not None:
            self.video = self.video.subclip(duration[0], duration[1])
        self._saveImgFolder = saveImgFolder

        # Load yolo
        self.netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
        self.metaMain = load_meta(metaPath.encode("ascii"))
        altNames = None
        if altNames is None:
            # In Python 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def generate(self, y1, y2, thresh=0.25):
        """
        @para y1 lower-bound
                采样的下边界
        @para y2 upper-bound
                采样的上边界
        :return:
        """
        blank_img = None
        low_bound = y1
        up_bound = y2

        time_idx = 0
        for frame in self.video.iter_frames():
            if blank_img is None:
                blank_img = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)

            res = detect(self.netMain, self.metaMain, frame, thresh, altNames=self.altNames)
            print('In time: %i, %i objects detected' % (time_idx, len(res)))
            for obj in res:
                tag, prob, pos = obj[0], obj[1], obj[2]

                # Paint the trajectory of objects which are in the sampling area of motion profile
                center_x, center_y, w, h = pos[0], pos[1], pos[2], pos[3]
                w = w * 0.3
                try:
                    if low_bound < center_y < up_bound and prob > 0.5 and time_idx < blank_img.shape[0]:
                        x1 = center_x - 0.5 * w if center_x - 0.5 * w > 0 else 0
                        x2 = center_x + 0.5 * w if center_x + 0.5 * w < blank_img.shape[1] else blank_img.shape[1]
                        rr, cc = draw.line(time_idx, int(x1), time_idx, int(x2))
                        if tag == 'car' or tag == 'truck' or tag == 'bus':
                            blank_img[rr, cc, 0] = 255
                        elif tag == 'person':
                            blank_img[rr, cc, 1] = 255
                except IndexError as e:
                    print(e)

            time_idx += 1
        return blank_img


if __name__ == '__main__':
    EVENT_ID_FOLDER = r'E:\journal_paper\Applied_Science\data\VTTI\vtti_event_id'
    VIDEO_FOLDER = r'E:\journal_paper\Applied_Science\data\VTTI\VehicleID_296344_DriverID_22207'
    OUTPUT_FOLDER = r'E:\journal_paper\Applied_Science\data\VTTI\yolo_motion'
    for event_f in os.listdir(EVENT_ID_FOLDER):
        video_name = event_f.split('.')[0]
        video_path = os.path.join(VIDEO_FOLDER, video_name + '_Front.mp4')

        # Read time id from event id file
        ef = open(os.path.join(EVENT_ID_FOLDER, event_f), 'r')
        time_ids = ef.readlines()
        ef.close()
        # Generate yolo motion image for each event
        for time_id in time_ids:
            time_id = int(time_id) / 10
            output_path = os.path.join(OUTPUT_FOLDER, video_name + '_' + str(time_id) + '.jpg')
            if os.path.exists(output_path):
                print('%s has been created.' % output_path)
                continue
            gen = YMGenerator(video_path, duration=[time_id-8, time_id+4])
            img = gen.generate(120, 360)

            io.imsave(output_path, img)
            print('===================%s saved!===================' % output_path)
            del gen

import argparse
import sys
import time
from threading import Thread
from queue import Queue
import pickle

from tqdm import tqdm
import cv2

import utils

from Modules.imager import ImageReader
from Modules.windows_client import WindowsClient

parser = argparse.ArgumentParser(description='Debug')
parser.add_argument('--rgb_path', type=str, default=r"I:\0walk/RGB",
                        help='path to store rgb images')
parser.add_argument('--dp_path', type=str, default=r"I:\0walk/depth",
                        help='path to store depth images')
parser.add_argument('--cam_json', type=str, default=r"/home/dongjai/catkin_ws/src/act_recognizer/src/mediandensity.json",
                        help='the camera setting json file path')
parser.add_argument('--width', type=int, default=640,
                        help='image width')
parser.add_argument('--height', type=int, default=480,
                        help='image height')
parser.add_argument('--fps', type=int, default=30,
                        help='camera fps')
parser.add_argument('--sp', default=True, action='store_true', #多线程系统显存不足
                    help='Use single process')
parser.add_argument('--d435', type=bool, default=False,
                        help='Using the D435 cam')
parser.add_argument('--host', type=str, default='10.60.2.65',#10.60.2.47 10.60.2.65
                        help='server_host')
parser.add_argument('--port', type=int, default=10086,#60001 10086
                        help='server_port')

args = parser.parse_args()
# args, unknown = parser.parse_known_args()#NOTE : using in ROS environment
# label_dict = {0: "walk", 1: "stop", 2: "come", 3: "phone", 4: "open", 5: "stand"}
label_list = ["walk", "stop", "come", "phone", "open", "stand"]

#Start the Image Reader Thread
image_reader = ImageReader(args)
image_reader.start()

my_windows_client = WindowsClient(args)
my_windows_client.start()

if args.d435:
    print('Starting  D435, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
else:
    data_len = image_reader.rgb_len
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)#终端输出

try:
    for i in im_names_desc:
        # (rgb_nums, rgb_frame) = camera_reader.read_rgb()
        (rgb_name, rgb_nums, rgb_frame) = image_reader.read_rgb()

        # (dp_nums, dp_frame) = camera_reader.read_dp()
        (dp_name, dp_nums, dp_frame) = image_reader.read_dp()

        my_windows_client.put_data(rgb_name, rgb_nums, rgb_frame, dp_name, dp_nums, dp_frame)

        (bboxes, pred_data) = my_windows_client.read_data()
        # print("pred data: ", pred_data)

        image = utils.draw_bbox(rgb_frame, bboxes, pred=pred_data, classes=label_list)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", rgb_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    image_reader.stop()
    my_windows_client.stop()
except KeyboardInterrupt:
    image_reader.stop()
    my_windows_client.stop()

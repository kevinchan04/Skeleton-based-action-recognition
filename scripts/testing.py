import argparse
import sys
import time
from threading import Thread
from queue import Queue

from tqdm import tqdm
import cv2
# import rospy

sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer/src/')
import core.utils as utils
from Modules.camera import CamReader
from Modules.imager import ImageReader
from Modules.detection import DetectionLoader
from Modules.poser import  PoseLoader
from Modules.midprocessor import MidProcessor
from Modules.recognizer import ActionRecognizer

def getTime(time1=0):
    if not time1:
        return time.time()
    else:
        interval = time.time() - time1
        return time.time(), interval

def loop():
    n = 0
    while True:
        yield n
        n += 1

parser = argparse.ArgumentParser(description='Debug')
parser.add_argument('--rgb_path', type=str, default=r"/home/dongjai/action data set/dataset/multiperson/2come/RGB",
                        help='path to store rgb images')
parser.add_argument('--dp_path', type=str, default=r"/home/dongjai/action data set/dataset/multiperson/2come/depth",
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
parser.add_argument('--d435', type=bool, default=True,
                        help='Using the D435 cam')
parser.add_argument('--yolo_gpu_usages', type=float, default=0.4,
                    help='restrict the gpu memory')
parser.add_argument('--nn_pix', type=int, default=3,
                    help='extend pixel for depth image')
parser.add_argument('--kps_usage_num', type=int, default=5,
                    help='the number of kps be used in recognizor')

args = parser.parse_args()
# label_dict = {0: "walk", 1: "stop", 2: "come", 3: "phone", 4: "open", 5: "stand"}
label_list = ["walk", "stop", "come", "phone", "open", "stand"]
walk_t, stand_t, come_t, stop_t, open_t, phone_t = 0,0,0,0,0,0
last_action_name, action_stable_time, stable_action_name = None, None, None

mid_proc_time, mlp_pred_time = 0, 0
det_loader_put_time, det_time = 0, 0
pose_loader_put_time, det_openpose_time = 0, 0

det_loader = DetectionLoader(args)
det_loader.start()

camera_reader = CamReader(args)
camera_reader.start()

if args.d435:
    print('Starting  D435, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
else:
    data_len = image_reader.rgb_len
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)#终端输出

try:
    for i in im_names_desc:
        (rgb_name, rgb_nums, rgb_frame) = camera_reader.read_rgb()
        (dp_name, dp_nums, dp_frame) = camera_reader.read_dp()

        det_loader.put_image(rgb_name, rgb_nums, rgb_frame)
        (rgb_name, det_nums, bboxes) = det_loader.read()

        image = rgb_frame
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    camera_reader.stop()
    det_loader.stop()
except KeyboardInterrupt:
    # print("walk: {}, stand: {}, stop: {}, come:{}, open:{}, phone:{}".format(walk_t, stand_t, stop_t, come_t, open_t, phone_t)) 
    # image_reader.stop()
    camera_reader.stop()
    det_loader.stop()
    # pose_loader.stop()
    # mid_processor.stop()
    # act_recognizer.stop()
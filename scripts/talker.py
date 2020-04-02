#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python

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
from Modules.act_talker import Talker
# from Modules.myserver import myServer

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
parser.add_argument('--d435', type=bool, default=False,
                        help='Using the D435 cam')
parser.add_argument('--yolo_gpu_usages', type=float, default=0.4,
                    help='restrict the gpu memory')
parser.add_argument('--nn_pix', type=int, default=3,
                    help='extend pixel for depth image')
parser.add_argument('--kps_usage_num', type=int, default=5,
                    help='the number of kps be used in recognizor')
parser.add_argument('--kp_num', type=list, default=[0, 4, 7, 9, 12],
                    help='the kps be used in recognizor')
parser.add_argument('--yolo_weights', type=str, default=r"/home/dongjai/catkin_ws/src/act_recognizer/src/checkpoints/YOLO/yolov3.weights",
                    help='yolov3 weights') 
parser.add_argument('--mlp_weights', type=str, default=r"/home/dongjai/catkin_ws/src/act_recognizer/src/checkpoints/new_mlp20_35_15.h5",
                    help='mlp weights') 
parser.add_argument('--host', type=str, default='10.60.2.65',
                    help='server_host')
parser.add_argument('--port', type=int, default=10086,
                    help='server_port')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()#NOTE : using in ROS environment
# label_dict = {0: "walk", 1: "stop", 2: "come", 3: "phone", 4: "open", 5: "stand"}
label_list = ["walk", "stop", "come", "phone", "open", "stand"]
#For data statstic
walk_t, stand_t, come_t, stop_t, open_t, phone_t = 0,0,0,0,0,0
last_action_name, action_stable_time, stable_action_name = None, None, None

mid_proc_time, mlp_pred_time = 0, 0
det_loader_put_time, det_time = 0, 0
pose_loader_put_time, det_openpose_time = 0, 0

#Start the Server to receive data from client
# proc_server = myServer(args)
# proc_server.start()

#Start the Camera Thread
# camera_reader = CamReader(args)
# camera_reader.start()

#Start the Image Reader Thread
image_reader = ImageReader(args)
image_reader.start()

#Start the YOLO Detection Thread
det_loader = DetectionLoader(args)
det_loader.start()

#Start the Openpose Estimation Thread
pose_loader = PoseLoader(args)
pose_loader.start()

#Start the Middle Data Processing Thread
mid_processor = MidProcessor(args)
mid_processor.start()

#Start the MLP Action Recognition Thread
act_recognizer = ActionRecognizer(args)
act_recognizer.start()

#Start the ROS Talker Thread to publish recognition data
ros_talker = Talker(args)
ros_talker.start()

if args.d435:
    print('Starting  D435, press Ctrl + C to terminate...')
    sys.stdout.flush()
    im_names_desc = tqdm(loop())
else:
    # data_len = 2000
    data_len = image_reader.rgb_len
    im_names_desc = tqdm(range(data_len), dynamic_ncols=True)#终端输出

try:
    for i in im_names_desc:
        pred_data = None
        start_time = getTime()

        # (rgb_name, rgb_nums, rgb_frame, dp_name, dp_nums, dp_frame) = proc_server.read_data()

        # (rgb_nums, rgb_frame) = camera_reader.read_rgb()
        (rgb_name, rgb_nums, rgb_frame) = image_reader.read_rgb()
        # ckpt_time, rgb_read_time = getTime(start_time)

        # (dp_nums, dp_frame) = camera_reader.read_dp()
        (dp_name, dp_nums, dp_frame) = image_reader.read_dp()
        ckpt_time, img_read_time = getTime(start_time)

        det_loader.put_image(rgb_nums, rgb_frame)
        ckpt_time, det_loader_put_time = getTime(ckpt_time)

        (det_nums, bboxes) = det_loader.read()
        ckpt_time, det_time = getTime(ckpt_time)

        pose_loader.put_image(rgb_nums, rgb_frame)
        ckpt_time, pose_loader_put_time = getTime(ckpt_time)

        (kp_nums, kps_list) = pose_loader.read()
        ckpt_time, det_openpose_time = getTime(ckpt_time)
        
        if kps_list and bboxes:# if detect some human in camera
            mid_processor.put_data(kp_nums, kps_list, dp_nums, dp_frame, det_nums, bboxes)
            ckpt_time, mid_proc_put_time = getTime(ckpt_time)

            (kp_nums, output) = mid_processor.read()
            ckpt_time, mid_proc_time = getTime(ckpt_time)
            # output = True
            if output:
                # print(output)
                act_recognizer.put_data(kp_nums, output)
                (out_nums, pred_data) = act_recognizer.read()
                ckpt_time, mlp_pred_time = getTime(ckpt_time)
                # cv2.imwrite("/home/dongjai/catkin_ws/src/act_recognizer/store/random/{}.png".format(rgb_nums), image)
                print(pred_data)
                #Using ROS talker to publish the information.
                # ros_talker.put_data(pred_data)
                for i in pred_data:
                    action_t = i[-1]
                    if action_t == 0:
                        walk_t +=1
                    if action_t == 5:
                        stand_t += 1
                    if action_t == 1:
                        stop_t +=1
                    if action_t == 3:
                        phone_t +=1
                    if action_t == 2:
                        come_t +=1
                    if action_t == 4:
                        open_t +=1   
            else:
                pred_data = None
        # proc_server.get_data(pred_data)
        image = utils.draw_bbox(rgb_frame, bboxes, pred=pred_data, classes=label_list)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", image)
        # cv2.imwrite("/home/dongjai/catkin_ws/src/act_recognizer/store/2come/{}.png".format(rgb_nums), image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            # im_names_desc.set_description(
            #         'img time: {img:.4f} | det put: {put:.4f} | det: {det:.4f} | opt: {opt:.4f} | mid: {mid:.4f} | mlp: {mlp:.4f}'.format(
            #         img=img_read_time, put=det_loader_put_time, det=det_time, opt=det_openpose_time, mid=mid_proc_time, mlp=mlp_pred_time)
            # )
    print("walk: {}, stand: {}, stop: {}, come:{}, open:{}, phone:{}".format(walk_t, stand_t, stop_t, come_t, open_t, phone_t))       
    # proc_server.stop() 
    image_reader.stop()
    # camera_reader.stop()
    det_loader.stop()
    pose_loader.stop()
    mid_processor.stop()
    act_recognizer.stop()
    ros_talker.stop()
except KeyboardInterrupt:
    print("walk: {}, stand: {}, stop: {}, come:{}, open:{}, phone:{}".format(walk_t, stand_t, stop_t, come_t, open_t, phone_t)) 
    # proc_server.stop() 
    image_reader.stop()
    # camera_reader.stop()
    det_loader.stop()
    pose_loader.stop()
    mid_processor.stop()
    act_recognizer.stop()
    ros_talker.stop()






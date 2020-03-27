#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
from sys import platform
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np

sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer/src/')
import core.utils as utils
import core.operation as operation

sys.path.append('/home/dongjai/openpose/build/python/')
from openpose import pyopenpose as op

class PoseLoader():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.kp_num = opt.kp_num
        self._params = dict()
        self._params["model_folder"] = "/home/dongjai/openpose/models/"
        self._params["model_pose"] = "COCO"
        self._params["face"] = False
        self._params["hand"] = False
        self.pose_estimator = self._load_estimator()

        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)#input queue
            self.pose_queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.pose_queue = mp.Queue(maxsize=queueSize)
        
    def _load_estimator(self):
        print("Loading Openpose......")
        self._opWrapper = op.WrapperPython()
        self._opWrapper.configure(self._params)
        self._opWrapper.start()

        datum = op.Datum()
        print("Successfully Loading Openpose")
        return datum
        
    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p

    def start(self):
        self.estimator_worker = self.start_worker(self.estimator_process)
        return self

    def stop(self):
        self.estimator_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.pose_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def estimator_process(self, **kargcs):
        while True:
            (nums, frame) = self.wait_and_get(self.image_queue)
            self.pose_estimator.cvInputData = frame
            self._opWrapper.emplaceAndPop([self.pose_estimator])
            if not operation.is_get_keypoints(self.pose_estimator):
                print("No keypoints")
                self.wait_and_put(self.pose_queue, (nums, None))
            else:
                kps_list = operation.get_keypoints_xy(datum=self.pose_estimator, kp_num=self.kp_num) 
                self.wait_and_put(self.pose_queue, (nums, kps_list))

    def put_image(self, nums, frame):
        self.wait_and_put(self.image_queue, (nums, frame))

    def read(self):
        return self.wait_and_get(self.pose_queue)

#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np

sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer/src/')
import tensorflow as tf

from core.yolov3 import YOLOv3, decode
import core.utils as utils
import core.operation as operation

class DetectionLoader():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self._num_classes = 80
        self._input_size = 416
        self._gpu_usages = opt.yolo_gpu_usages#0.4
        self.detector = self._load_detector()

        # initialize the queue used to store data
        """
        image_queue: the buffer storing pre-processed images for object detection
        det_queue: the buffer storing human detection results
        pose_queue: the buffer storing post-processed cropped human image for pose estimation
        """
        if opt.sp:
            self._stopped = False
            self.image_queue = Queue(maxsize=queueSize)#input queue
            self.det_queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.image_queue = mp.Queue(maxsize=queueSize)
            self.det_queue = mp.Queue(maxsize=queueSize)

    def _load_detector(self):
        print("Loading YOLO Model......")
        config_tf = tf.compat.v1.ConfigProto()
        config_tf.gpu_options.per_process_gpu_memory_fraction = self._gpu_usages
        tf.compat.v1.Session(config=config_tf)
        input_layer  = tf.keras.layers.Input([self._input_size, self._input_size, 3])
        feature_maps = YOLOv3(input_layer)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)
        detector = tf.keras.Model(input_layer, bbox_tensors)
        utils.load_weights(detector, self.opt.yolo_weights)
        # detector.summary()
        print("Successfully Loading YOLO Model")
        return detector

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target)#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p

    def start(self):
        self.detection_worker = self.start_worker(self.detection_process)
        return self

    def stop(self):
        self.detection_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.image_queue)
        self.clear(self.det_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def detection_process(self , **kargcs):
        while True:
            (nums, frame) = self.wait_and_get(self.image_queue)
            frame_size = frame.shape[:2]
            image_data = utils.image_preporcess(np.copy(frame), [self._input_size, self._input_size])#裁剪图片成416x416
            image_data = image_data[np.newaxis, ...].astype(np.float32)#转换成32位浮点数格式

            pred_bbox = self.detector.predict(image_data)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self._input_size, 0.3)#对预测出来的bouding box作后处理
            bboxes = utils.nms(bboxes, 0.45, method='nms')#非极大抑制nms选择概括性最高的一个框

            only_one_bboxes = operation.detect_only_one(bboxes, IDi=0)#只输出一个类别
            self.wait_and_put(self.det_queue, (nums, only_one_bboxes))

    def put_image(self, nums, frame):
        self.wait_and_put(self.image_queue, (nums, frame))

    def read(self):
        return self.wait_and_get(self.det_queue)

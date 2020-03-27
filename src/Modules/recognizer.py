#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
from sys import platform
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np
import tensorflow as tf

sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer/src/')
import core.utils as utils
import core.operation as operation

class ActionRecognizer():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.kps_size = opt.kps_usage_num

        self.act_recognizer = self._load_recognizer()

        if opt.sp:
            self._stopped = False
            self.data_queue = Queue(maxsize=queueSize)#input queue
            self.act_queue = Queue(maxsize=queueSize)
        else:
            self._stopped = mp.Value('b', False)
            self.data_queue = mp.Queue(maxsize=queueSize)
            self.act_queue = mp.Queue(maxsize=queueSize)
    
    def _load_recognizer(self):
        print("Loading Recognizer......")
        mlp_inputs = tf.keras.Input(shape=(3, self.kps_size, 1))
        mlp_outputs = operation.my_mlp(mlp_inputs)
        mlp_model = tf.keras.Model(inputs=mlp_inputs, outputs=mlp_outputs)
        mlp_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = tf.keras.losses.sparse_categorical_crossentropy,
            metrics = [tf.keras.metrics.sparse_categorical_accuracy]
        )
        mlp_model.load_weights("/home/dongjai/catkin_ws/src/act_recognizer/src/checkpoints/new_mlp20_35_15.h5")
        print("Successfully Loading Recognizer")
        return mlp_model

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p

    def start(self):
        self.recognizer_worker = self.start_worker(self.recognizer_process)
        return self

    def stop(self):
        self.recognizer_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.data_queue)
        self.clear(self.act_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def recognizer_process(self, **kargcs):
        while True:
            (nums, data) = self.wait_and_get(self.data_queue)
            len_data = len(data)
            # print(len_data)
            _xyz_data = []
            # action_name = []
            for _t in data:
                _xyz_data.append(_t[1].tolist())
            _q = np.asarray(_xyz_data)
            action_pred = self.act_recognizer.predict(_q, batch_size=len_data)
            # print("pred: ",action_pred)
            action_name = operation.get_action_name(label=action_pred)#需要测试多人的情况
                # action_name.append(operation.get_action_name(label=action_pred))
            # print("name: ",action_name)
            for i, _t in enumerate(data):
                _t[1] = action_name[i]
            self.wait_and_put(self.act_queue, (nums, data))

    def put_data(self, nums, data):
        self.wait_and_put(self.data_queue, (nums, data))
    
    def read(self):
        return self.wait_and_get(self.act_queue)
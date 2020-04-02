#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import time
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np

class ImageReader():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.rgb_path = opt.rgb_path
        self.dp_path = opt.dp_path

        for root, dirs, files in os.walk(self.rgb_path):
            self.rgb_len = len(files)
        for root, dirs, files in os.walk(self.dp_path):
            self.dp_len = len(files)
        assert self.rgb_len == self.dp_len, "RGB和深度图数量不相等"

        if opt.sp:#单线程
            self.rgb_queue = Queue(maxsize=queueSize)
            self.dp_queue = Queue(maxsize=queueSize)

    def start_worker(self, target, path):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={"num": 0, "path": path})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p
    
    def start(self):
        self.rgb_image_worker = self.start_worker(self.rgb_reader_process, path=self.rgb_path)
        self.dp_image_worker = self.start_worker(self.dp_reader_process, path=self.dp_path)
        return self
    
    def stop(self):
        self.rgb_image_worker.join()
        self.dp_image_worker.join()
        self.clear_queues()
    
    def clear_queues(self):
        self.clear(self.rgb_queue)
        self.clear(self.dp_queue)
    
    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def rgb_reader_process(self, **kargcs):
        while kargcs["num"] < self.rgb_len:
            rgb_name = "color_" + str(kargcs["num"]) + ".png"
            rgb_file = kargcs["path"] + "/" + rgb_name
            # print(rgb_file)
            frame = cv2.imread(rgb_file, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.wait_and_put(self.rgb_queue, (rgb_name, kargcs["num"], frame))
            kargcs["num"] += 1
    
    def dp_reader_process(self, **kargcs):
        while kargcs["num"] < self.dp_len:
            dp_name = "depth_8bit_" + str(kargcs["num"]) + ".png"
            dp_file = kargcs["path"] + "/" + dp_name
            frame = cv2.imread(dp_file, -1)
            self.wait_and_put(self.dp_queue, (dp_name, kargcs["num"], frame))
            kargcs["num"] += 1
    
    def read_rgb(self):
        return self.wait_and_get(self.rgb_queue)
    
    def read_dp(self):
        return self.wait_and_get(self.dp_queue)
    


#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
import time
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np

from core import camera as cam
import pyrealsense2 as rs
import json

class CamReader():
    def __init__(self, opt, queueSize=10):
        self.opt = opt
        self.json_file_path = opt.cam_json
        self.width = opt.width
        self.height = opt.height
        self.fps = opt.fps

        if opt.sp:#单线程
            self.rgb_queue = Queue(maxsize=queueSize)
            self.dp_queue = Queue(maxsize=queueSize)
    
    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p

    def start(self):
        self.cam_worker = self.start_worker(self.cam_reader_process)
        return self

    def stop(self):
        self.cam_worker.join()
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
    
    def cam_reader_process(self, **kargcs):
        print("Loading Intel Realsense D435......")
        try:
            print(self.json_file_path)
            cam.try_to_advanced_mode(self.json_file_path)
        except Exception as e:
            print(e)
            pass
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        # Start streaming
        pipeline.start(config)
        cam.wait4cam_init(pipeline, WaitTimes=10) #等待相机初始化调节自动曝光
        #align
        align_to = rs.stream.color
        align = rs.align(align_to)
        print("Camera Starting")
        frame_num = 0
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)# Align the depth frame to color frame
            aligned_depth_frame = aligned_frames.get_depth_frame()# Get aligned depth and color frames
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth_frame or not color_frame:#只有一张rgb和depth不对应的话就舍去
                continue
            # dp_frame = np.asanyarray(aligned_depth_frame.get_data()) #16bit 
            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.wait_and_put(self.rgb_queue, (frame_num, frame))
            self.wait_and_put(self.dp_queue, (frame_num, aligned_depth_frame))
            frame_num += 1

    def read_rgb(self):
        return self.wait_and_get(self.rgb_queue)
    
    def read_dp(self):
        return self.wait_and_get(self.dp_queue)
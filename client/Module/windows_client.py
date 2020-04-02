import os
import sys
import time
from threading import Thread
from queue import Queue
import pickle

import cv2

class WindowsClient():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.host = opt.host
        self.port = opt.port
        from multiprocessing.connection import Client
        self.client = Client((self.host, self.port))

        if opt.sp:
            self.frame_queue = Queue(maxsize=queueSize)
            self.output_queue = Queue(maxsize=queueSize)
        
    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target)#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p
    
    def start(self):
        self.client_worker = self.start_worker(self.client_worker_process)
        self.client_recv_worker = self.start_worker(self.client_recv_process)
        return self
    
    def stop(self):
        self.client_worker.join()
        self.client_recv_worker.join()
        self.clear_queues()
    
    def clear_queues(self):
        self.clear(self.frame_queue)
        self.clear(self.output_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def client_worker_process(self, **kargcs):
        while True:
            (rgb_name, rgb_No, rgb_frame, dp_name, dps_No, dp_frame) = self.wait_and_get(self.frame_queue)
            data_string = pickle.dumps((rgb_name, rgb_No, rgb_frame, dp_name, dps_No, dp_frame))
            # byte_size = sys.getsizeof(data_string)
            # print(byte_size)
            self.client.send(data_string)
            # print('Send', type(data_string))
            # time.sleep(0.02)

            # pred_data_byte = self.client.recv()
            # # (rgb_name, rgb_No, rgb_frame, dp_name, dps_No, dp_frame) = pickle.loads(pred_data_byte)
            # (bboxes, pred_data) = pickle.loads(pred_data_byte)
            # # print("Recv", pred_data)
            # self.wait_and_put(self.output_queue, (bboxes, pred_data))

    def client_recv_process(self):
        while True:
            pred_data_byte = self.client.recv()
            # (rgb_name, rgb_No, rgb_frame, dp_name, dps_No, dp_frame) = pickle.loads(pred_data_byte)
            (bboxes, pred_data) = pickle.loads(pred_data_byte)
            # print("Recv", pred_data)
            self.wait_and_put(self.output_queue, (bboxes, pred_data))

    def read_data(self):
        return self.wait_and_get(self.output_queue)
    
    def put_data(self, rgb_name, rgb_No, rgb_frame, dp_name, dp_No, dp_frame):
        self.wait_and_put(self.frame_queue, (rgb_name, rgb_No, rgb_frame, dp_name, dp_No, dp_frame))


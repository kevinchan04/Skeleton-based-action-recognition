#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
from threading import Thread
from queue import Queue
import pickle


class myServer():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.host = opt.host
        self.port = opt.port
        from multiprocessing.connection import Listener
        server_sock = Listener((self.host, self.port))

        self.conn = server_sock.accept()
        print('Server Listening')

        if opt.sp:
            self.data_queue = Queue(maxsize=queueSize)
            self.outptu_queue = Queue(maxsize=queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p
    
    def start(self):
        self.server_worker = self.start_worker(self.frame_reader_process)
        self.data_send_worker = self.start_worker(self.data_send_process)
        return self

    def stop(self):
        self.server_worker.join()
        self.data_send_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.data_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def frame_reader_process(self, **kargcs):
        while True:
            data_bytes = self.conn.recv()
            data = pickle.loads(data_bytes)
            # print('Received:', type(data))
            self.wait_and_put(self.data_queue, data)

    def data_send_process(self):
        while True:
            (bboxes, pred_data) = self.wait_and_get(self.outptu_queue)
            pred_data_bytes = pickle.dumps((bboxes, pred_data))
            self.conn.send(pred_data_bytes)
    
    def read_data(self):
        return self.wait_and_get(self.data_queue)
    
    def get_data(self, bboxes, pred_data):
        self.wait_and_put(self.outptu_queue, (bboxes, pred_data))
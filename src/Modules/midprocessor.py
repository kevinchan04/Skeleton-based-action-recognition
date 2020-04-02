#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
import time
from sys import platform
from threading import Thread
from queue import Queue

import cv2
import scipy.misc
import numpy as np

sys.path.append('..')
import core.utils as utils
import core.operation as operation

class MidProcessor():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self.nn_pix = opt.nn_pix

        if opt.sp:#单线程
            self.data_queue = Queue(maxsize=queueSize)
            self.fuse_queue = Queue(maxsize=queueSize)

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p
    
    def start(self):
        self.fuse_worker = self.start_worker(self.fuse_process)
        return self
    
    def stop(self):
        self.fuse_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.data_queue)
        self.clear(self.fuse_queue)
    
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def fuse_process(self, **kargcs):
        while True:
            (kp_nums, kps_list, dp_nums, dp_frame, det_nums, bboxes) = self.wait_and_get(self.data_queue)
            # print(kp_nums, dp_nums, det_nums)
            assert (kp_nums == dp_nums and dp_nums == det_nums), "队列数据顺序不匹配"
            # print("kp list: ",kps_list)
            # print("bboxes: ",bboxes)
            _flag = self._filter_error(kps_list, bboxes)
            if not _flag or len(kps_list) != len(bboxes):#change in Friday night
                print("error")
                self.wait_and_put(self.fuse_queue, ((kp_nums, None)))
            else:
                output = []
                for kp_list in kps_list:
                    # file_txt = open(r"/home/dongjai/action data set/dataset/5_stand.txt", 'a')#根据不同的类别修改txt的文件命名前缀
                    #[x, y, z]获得5个点映射到原图frame的3维坐标(x, y, z)，和附近的几个点
                    if self.opt.d435:
                        dp_nparray = operation.get_keypoints_xyz(kp_list=kp_list, nn_pix=self.nn_pix, live=True, live_frame=dp_frame)
                    else:
                        dp_nparray = operation.get_keypoints_xyz(kp_list=kp_list, depth_frame=dp_frame, nn_pix=self.nn_pix)
                    #求平均值替代中心点深度值
                    dp_list = operation.get_mean_dep_val_xyz(kp_list=kp_list, dp_nparray=dp_nparray)
                    #坐标归一化
                    dp_array_out = operation.norm_keypoints_xyz(dp_list, norm_p_index=0, rgb_size=[640, 480])
                    #坐标平移，让坐标值都变成非负数
                    dp_array_no_neg = operation.no_negative(array=dp_array_out)
                    #将小于0.001的数字认为是0
                    dp_array_no_neg_zero = operation.filter_zero(dp_array_no_neg, threadhold=0.001)
                    # file_txt.write(str(dp_array_no_neg))
                    # file_txt.write('\n')
                    # file_txt.close()
                    dp_array_no_neg_zero = np.expand_dims(dp_array_no_neg_zero, axis=-1) 
                    # dp_array_no_neg_zero = np.expand_dims(dp_array_no_neg_zero, axis=0) 
                    # print(dp_array_no_neg_zero)

                    output.append([dp_list[0], dp_array_no_neg_zero])#kp_list[0] means the nose joint of body
                self.wait_and_put(self.fuse_queue, (kp_nums, output))
    
    def _filter_error(self, kps_list, bboxes):
        kps_len = len(kps_list)
        bboxes_len = len(bboxes)
        if not kps_len == bboxes_len:
            print("目标检测和姿态估计人数结果不匹配")
            return False
        kps_cet, bboxes_cet = [], []
        kps_array = np.asarray(kps_list)
        for i in bboxes:
            x_min, y_min = i[0], i[1]
            x_max, y_max = i[2], i[3]
            _j = 0
            for j in kps_array:
                # print("J:",j)
                for p in j:
                    # print("P: ",p)
                    if p[0] >= x_min and p[0] <= x_max:
                        if p[1] >= y_min and p[1] <= y_max:
                            _j += 1
            # print("-j: ", _j)
            if _j >= 10:#too many kps in one bbox, error occur in openpose
                print("too many kps in one bbox, error occur in openpose")
                return False
                break
        return True

    def filter_IoU_error(self, kps_list, bboxes):
        '''
        bboxes:[x_min,y_min,x_max,y_max,prob,cls]
        '''
        kps_len = len(kps_list)
        bboxes_len = len(bboxes)
        assert kps_len == bboxes_len, "目标检测和姿态估计人数结果不匹配"
        for i in kps_list:
            for j in i:
                if j[0] == 0 or j[1] == 0:
                    i.remove(j)
        kps_box = []
        kps_array = np.asarray(kps_list)
        for i in kps_array:
            x_min, y_min = np.amin(i, 0)
            x_max, y_max = np.amax(i, 0)
            # x_cet = x_min + int((x_max - x_min) / 2)
            # y_cet = y_min + int((y_max - y_min) / 2)
            kps_box.append([x_min, y_min, x_max, y_max])
        print("kps_box: ",kps_box)
        _t = 0
        for i in kps_box:
            for j in bboxes:
                print(j[:4])
                _IoU = self.IOU(i, j[:4])
                print(_IoU)
                if _IoU > 0.4:
                    _t += 1
        if _t != len(kps_box):
            print("出现错误, 跳过一帧")
        # else
        pass

    def IOU(self, box1, box2):
        '''
        :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
        :param box2:[x1,y1,x2,y2]
        :return: iou_ratio--交并比
        '''
        width1 = abs(box1[2] - box1[0])
        height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
        width2 = abs(box2[2] - box2[0])
        height2 = abs(box2[1] - box2[3])
        x_max = max(box1[0],box1[2],box2[0],box2[2])
        y_max = max(box1[1],box1[3],box2[1],box2[3])
        x_min = min(box1[0],box1[2],box2[0],box2[2])
        y_min = min(box1[1],box1[3],box2[1],box2[3])
        iou_width = x_min + width1 + width2 - x_max
        iou_height = y_min + height1 + height2 - y_max
        if iou_width <= 0 or iou_height <= 0:
            iou_ratio = 0
        else:
            iou_area = iou_width * iou_height # 交集的面积
            box1_area = width1 * height1
            box2_area = width2 * height2
            iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
        return iou_ratio

    def put_data(self, kp_nums, kp_list, dp_nums, dp_frame, det_nums, bboxes):
        self.wait_and_put(self.data_queue, (kp_nums, kp_list, dp_nums, dp_frame, det_nums, bboxes))
    
    def read(self):
        return self.wait_and_get(self.fuse_queue)

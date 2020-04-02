#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
from threading import Thread
from queue import Queue

import rospy
from std_msgs.msg import String
sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer')
from act_recognizer.msg import Person
from act_recognizer.msg import ActElem
# from act_recognizer.msg import Persons

import cv2
import scipy.misc
import numpy as np

class Talker():
    def __init__(self, opt, queueSize=128):
        self.opt = opt
        self._talker, self._talk_rate  = self._init_talker()

        if opt.sp:
            self.data_queue = Queue(maxsize=queueSize)
            self.talk_queue = Queue(maxsize=queueSize)
        else:
            self.data_queue = mp.Queue(maxsize=queueSize)
            self.talk_queue = mp.Queue(maxsize=queueSize)
        
    def _init_talker(self):
        print("Initializing ROS Talker......")
        pub = rospy.Publisher('chatter', Person, queue_size=10)
        rospy.init_node('puber_xyzcls', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        print("Successfully Initialized ROS Talker")
        return pub, rate
    
    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=(), kwargs={})#target是該線程需要完成的方法函數
        # else:
        p.start()
        return p

    def start(self):
        self.talker_worker = self.start_worker(self.talker_process)
        return self

    def stop(self):
        self.talker_worker.join()
        self.clear_queues()

    def clear_queues(self):
        self.clear(self.talk_queue)

    def clear(self, queue):
        while not queue.empty():
            queue.get()
    
    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()
    
    def talker_process(self, **kargcs):
        while not rospy.is_shutdown():
            pred_data_list = self.wait_and_get(self.data_queue)
            person_msg = self.persondata_to_msg(pred_data_list)

            # hello_str = "hello world %s" % rospy.get_time()
            # rospy.loginfo(person_msg)
            self._talker.publish(person_msg)
            self._talk_rate.sleep()
    
    def persondata_to_msg(self, pred_data_list):
        # persons = Persons()
        person = Person()
        for i in pred_data_list:
            act_elem = ActElem()
            act_elem.x = i[0][0]
            act_elem.y = i[0][1]
            act_elem.z = i[0][2]
            act_elem.act_cls = i[-1]
            person.xyz_cls.append(act_elem)
            # persons.persons.append(person)
            # person.xyz_cls.clear()
        return person

    def put_data(self, pred_data_list):
        self.wait_and_put(self.data_queue, (pred_data_list))


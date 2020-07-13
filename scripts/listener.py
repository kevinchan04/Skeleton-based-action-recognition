#!/home/dongjai/anaconda3/envs/tensorflow2/bin/python
import os
import sys
import rospy
from std_msgs.msg import String
sys.path.append('/home/dongjai/catkin_ws/src/act_recognizer')
from act_recognizer.msg import Persons
from act_recognizer.msg import Person
from act_recognizer.msg import ActElem

def callback(data):
    rospy.logwarn(data.xyz_cls)
    #print(data.xyz_cls)

def listener():
    rospy.init_node('suber_xyzcls', anonymous=True)
    rospy.Subscriber('chatter', Person, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()



    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

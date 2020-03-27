#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
import core.operation as operation
import core.camera as cam

#####################################################################
######                        Import openpose  &&  Openpose initialization                       ######
import os
import sys
from sys import platform
import argparse

sys.path.append('/home/dongjai/openpose/build/python')
from openpose import pyopenpose as op

##       setting up openpose flags      ##
params = dict()
params["model_folder"] = "/home/dongjai/openpose/models/"
params["model_pose"] = "COCO"
params["face"] = False
params["hand"] = False

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Initialize Openpose
datum = op.Datum()
#####################################################################

# video_path      = "./docs/office2.mp4"
# img_path = "./docs/color_270.png"
# dp_path = "./docs/depth_8bit_270.png"
#make sure the RGB image and the depth image in the same size
alpha = 0.03 # depth frame is 8bit. if depth frame is 16bit, alpha = 0.0
# video_path      = 0
num_classes     = 80
input_size      = 416

input_layer  = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
# model.summary()

# vid = cv2.VideoCapture(video_path)
# cam.wait4cam_init(vid, WaitTimes=10) #等待相机初始化调节自动曝光
# vid = cv2.imread(img_path, -1)
# dp_vid = cv2.imread(dp_path, -1)
files_path = r"/home/dongjai/action data set/dataset"
dirs = os.listdir(files_path)
for files in dirs:
    img_file_path = r"/home/dongjai/action data set/dataset/" + files
    file_txt = open(img_file_path + "/" + files + ".txt", 'a')#根据不同的类别修改txt的文件命名前缀
    img_num = 0
    print("Action: ", files)
    while True:
        prev_time = time.time()
        if img_num == 2000: #读完所有照片，退出程序
            file_txt.close()
            break
        ########################################
        #####                           Read image                     ######
        ########################################
        print("Picture: ", img_num)
        img_path = img_file_path+"/RGB/color_"+str(img_num)+".png"
        dp_path = img_file_path+"/depth/depth_8bit_"+str(img_num)+".png"
        vid = cv2.imread(img_path, -1)
        dp_vid = cv2.imread(dp_path, -1)
        img_num = img_num + 1
        ########################################
        #####          YOLOv3 human detection         ######
        ########################################
        # return_value, frame = vid.read()
        frame = vid
        return_value = 1
        dp_frame = dp_vid
        # print("rgbshape:", frame.shape)
        # print("dpshape:", dp_frame.shape)
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])#裁剪图片成416x416
        image_data = image_data[np.newaxis, ...].astype(np.float32)#转换成32位浮点数格式

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)#对预测出来的bouding box作后处理
        bboxes = utils.nms(bboxes, 0.45, method='nms')#非极大抑制nms选择概括性最高的一个框
        only_one_bboxes = operation.detect_only_one(bboxes, IDi=0)#只输出一个类别
        prob_top_n_bboxes = operation.select_and_pick(only_one_bboxes, index=4, order=0, top_n=3)#index=4表示筛选的依据是分类概论
        top_n_bboxes_dp = operation.get_centerpoints_xyz(prob_top_n_bboxes, depth_frame=dp_frame)
        # print("top n bboxes dp : ", top_n_bboxes_dp)
        #############NOTE: 实际运行时，gap=的数值可以通过rs.get distance函数获取，这样就可以跳过在训练模型阶段抓取的img 8bit大小限制了深度只有8500的问题
        top_n_bboxes_in_gap = operation.select_in_gap(top_n_bboxes_dp, gap=[500, 8000], alpha=alpha)#过滤距离范围，gap单位mm
        the_one_bbox = operation.select_and_pick(top_n_bboxes_in_gap, index=-1, order=1, top_n=1)#深度值越靠近越低，所以用升序
        if len(the_one_bbox) == 0:
            print("error no detection")
            continue
        #确定操作者,选择yolo检测框中央坐标对应的深度值,认为处于距离范围（x～y）的最前者是操作者
        ########################################
        #####                        YOLOv3 End                       ######
        ########################################

        ########################################
        #####                          切割人像                            ######
        ########################################
        img_cut_set, left_top_list = operation.cut_bbox(frame, the_one_bbox, ext_pix_x=0, ext_pix_y=0)#获得每一帧图像上所有bbox内的分割图像，组成一个list图片集
        # img_cut_set = img_set[0:len(img_set)]#单独独立的图片
        if len(left_top_list) != len(img_cut_set):
            print("error")
        ########################################
        #####                    切割人像  End                        ######
        ########################################

        ########################################
        #####                        Openpose                           ######
        ########################################
        for i in range(len(img_cut_set)):#每一帧切割出来的单人图像一张输入openpose中
            #openpose阶段
            datum.cvInputData = img_cut_set[i]
            opWrapper.emplaceAndPop([datum])#openpose input image->BGR ############### ATTENTION 可以结合起来改善运行效率？

            # cv2.namedWindow("cut_one", cv2.WINDOW_AUTOSIZE)
            # cv2.imshow("cut_one", cut_out)
            cv2.namedWindow("cut_backbone_result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("cut_backbone_result", datum.cvOutputData)

            #坐标映射到depth frame阶段
            kp_num = [0, 8, 9, 10, 11, 14, 15, 16, 17]
            kp_list = operation.get_keypoints_xy(datum=datum, kp_num=kp_num) #[x, y]获得5个点的2维坐标(x, y)
            # print("kp list: ",kp_list)
            if operation.filter_kp_list(kp_list, norm_p_index=0):#如果检测的人体骨骼缺少了鼻子，则认为这个骨架是背对相机
                continue
            # dp_nparray.shape = (5, 7, 7, 3)
            nn_pix = 3
            dp_nparray = operation.get_keypoints_xyz(kp_list=kp_list, depth_frame=dp_frame, left_top=left_top_list[i], nn_pix=nn_pix)#[x, y, z]获得5个点映射到原图frame的3维坐标(x, y, z)，和附近的几个点
            #求平均值替代中心点深度值
            dp_list = operation.get_mean_dep_val_xyz(kp_list=kp_list, dp_nparray=dp_nparray)
            #坐标归一化
            dp_array_out = operation.norm_keypoints_xyz(dp_list, norm_p_index=0, cut_img=img_cut_set[i])
            #坐标平移，让坐标值都变成非负数
            dp_array_no_neg = operation.no_negative(array=dp_array_out)
            #将关节点数据存储到txt文件中#骨骼点深度信息处理阶段, 获得最具代表性的深度信息. 五个点的深度信息生成一个output向量
            #将小于0.001的数字认为是0
            dp_array_no_neg_zero = operation.filter_zero(dp_array_no_neg, threadhold=0.001)
            # print("dp nparray: ", dp_array_out)
            # cv2.imshow("nparray", dp_array_no_neg)

            file_txt.write(str(dp_array_no_neg_zero))
            file_txt.write('\n')
        ########################################
        #####                    Openpose End                      ######
        ########################################

        ########################################
        #####                              MLP                                 ######
        ########################################

            #神经网络动作识别阶段

        #可视化阶段
        ##########################
        ###                   原图                     ###
        ##########################
        curr_time = time.time()
        exec_time = curr_time - prev_time

        image = utils.draw_bbox(frame, only_one_bboxes)#画bouding box
        result = np.asarray(image)#好像没用
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            file_txt.close()
            break




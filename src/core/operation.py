import cv2
import numpy as np
import copy
import tensorflow as tf

def list_sortby(bboxes_one, index, order): 
    '''
    sort list. list likes [ [1,2,3,4], [5,6,7,8], [9,1,2,3]].  
    index refers to the inner [] number index. 
    index=4 means it will sort this list according 4 8 3
    ranked by descending order=0, increasing order=1
    '''
    if order == 0:
        tmp_len = len(bboxes_one)
        for i in range(tmp_len):#排序
            for j in range(tmp_len):
                tmp_t = bboxes_one[i][index]
                tmp_k = bboxes_one[j][index]
                if tmp_t > tmp_k:
                    tmp = bboxes_one[i]
                    bboxes_one[i] = bboxes_one[j]
                    bboxes_one[j] = tmp
    if order == 1:
        tmp_len = len(bboxes_one)
        for i in range(tmp_len):#排序
            for j in range(tmp_len):
                tmp_t = bboxes_one[i][index]
                tmp_k = bboxes_one[j][index]
                if tmp_t < tmp_k:
                    tmp = bboxes_one[i]
                    bboxes_one[i] = bboxes_one[j]
                    bboxes_one[j] = tmp
    return bboxes_one

def detect_only_one(bboxes, IDi):
    '''
    Select one specific class to output according the class ID.
    '''
    bboxes_one = []
    # bboxes_one_out = []
    bboxes_num = len(bboxes)
    if bboxes_num > 0:
        for j in range(bboxes_num):
            if int(bboxes[j][5]) == IDi:#只过滤等于ID的标签，person==0, car==2
                bboxes_one.append(bboxes[j])
    return bboxes_one

def pick_top_n(bboxes, top_n):
    '''
    It will select top n elements in bboxes(a list) as output.\\
    If the list of bboxes was sorted by increasing order, it will select lower elements.
    '''
    tmp_len = len(bboxes)
    if tmp_len < top_n :
        print("only have : ", tmp_len)
        top_n = tmp_len
    return bboxes[:top_n]
    

def select_and_pick(bboxes, index, order, top_n, gap=False):
    '''
    sorted according to index, 
    and ranked by descending order=0, increasing order=1 
    then select top_n elements as output
    '''
    tmp = pick_top_n(list_sortby(bboxes, index, order), top_n)
    if gap:
        tmp_out = []
        for i in range(len(tmp)):
            t = tmp[i][index]
            # print("prob", t)
            if t >= gap[0] and t < gap[1]:
                tmp_out.append(tmp[i])
    else:
        tmp_out = tmp.copy()
    return tmp_out

def select_in_gap(bboxes, gap, alpha, live=False):
    '''
    If depth frame is 16bit frame for each pixel, alpha should be equal to 0.
    If it is 8bit frame, alpha = 0.3
    '''
    i = len(bboxes)
    t = 0
    while t != i :
        if live:
            dist_16bit = bboxes[t][5] * 1000
        else:
            if alpha == 0:
                dist_16bit = bboxes[t][5]
            else:
                dist_16bit = bboxes[t][5] / alpha
        if dist_16bit >= float(gap[0]) and dist_16bit <= float(gap[1]):
            t = t + 1
        else:
            del bboxes[t]
            i = i - 1
    return bboxes
    

def cut_bbox(image, bboxes, ext_pix_x, ext_pix_y):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, depth] format coordinates.
    OUTPUT :
    left_top_list is a list that containing each single img left top point x,y values for frame(original size image),
    img_set is in BGR style
    """
    # img_blank = np.zeros(shape=image.shape, dtype=image.dtype)
    img_set = []
    left_top_list = []

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        if (coor[0] - ext_pix_x) < 0 : 
            x1 = 0 
        else:   
            x1 = coor[0] - ext_pix_x
        if (coor[1] - ext_pix_y) < 0 : 
            y1 = 0
        else: 
            y1 = coor[1] - ext_pix_y
        if (coor[2] + ext_pix_x) > image.shape[1]:
            x2 = image.shape[1]
        else:    
            x2 = coor[2] + ext_pix_x
        if (coor[3] + ext_pix_y) > image.shape[0]:
            y2 = image.shape[0]
        else:    
            y2 = coor[3] + ext_pix_y
        c1, c2 = (x1, y1), (x2, y2) # (x, y)
        img_one_blank = np.zeros(shape=(y2 - y1, x2 - x1, 3), dtype=image.dtype)
        img_one_blank[:, :, 0] = image[y1:y2, x1:x2, 2] #image->RGB  img_one_blank->BGR
        img_one_blank[:, :, 1] = image[y1:y2, x1:x2, 1] 
        img_one_blank[:, :, 2] = image[y1:y2, x1:x2, 0] 
        img_set.append(img_one_blank)
        left_top_list.append([x1, y1])
        # cv2.namedWindow("cut_result", cv2.WINDOW_AUTOSIZE)
        # cut_result = cv2.cvtColor(img_one_blank, cv2.COLOR_RGB2BGR)
        # cv2.imshow("cut_result", cut_result)
    # img_set.append(img_blank)#返回的图像和原图frame尺寸一致，只有bbox内的图像，但各自不是独立的
    return img_set, left_top_list

def get_centerpoints_xyz(bboxes, depth_frame, live=False, live_frame=None):
    '''
    bboxes [x1, y1, x2, y2, prob, cls]
    Change the cls into depth value
    '''
    bboxes_dp = copy.deepcopy(bboxes)
    for i in range(len(bboxes_dp)):
        x1 = bboxes_dp[i][0]
        y1 = bboxes_dp[i][1]
        x2 = bboxes_dp[i][2]
        y2 = bboxes_dp[i][3]
        c1 = int(( x2 - x1 ) / 2)
        c2 = int(( y2 - y1 ) / 2)
        if live:
            dp_val = live_frame.get_distance(c1, c2)
        else:
            dp_val = depth_frame[c2, c1]#depth_frame[c2, c1]
        bboxes_dp[i][-1] = dp_val
    return bboxes_dp

def is_get_keypoints(datum):
    '''
    Judge the keypoints set is empty or not.
    '''
    try:
        a = len(datum.poseKeypoints)
        # print("len keypoinst", a)
    except:
        return False
    return True

def get_keypoints_xy(datum, kp_num):
    '''
    Get keypoint (x, y) and put them into a list as return.
    '''
    kps_list = []
    for j in datum.poseKeypoints:
        kp_list = []
        for i in kp_num:
            x = j[i][0]
            y = j[i][1]
            kp_list.append([x, y]) #[x, y]
        kps_list.append(kp_list)
    return kps_list

def get_keypoints_xyz(kp_list, depth_frame=None, nn_pix=3, live=False, live_frame=None): #dp_frame.shape = (y, x) 单通道,   nn_num 中心点近邻像素数量
    '''
    nn_pix is neighbor pixel of the central pixel\\
    live is a symbol of livecam or not\\
    live_frame is depth frame flow grasp from camera
    '''
    # testkp_x, testkp_y = [326, 280, 365, 302, 343], [127, 269, 264, 335, 335]
    dp_nparray = np.ndarray(shape=(len(kp_list), nn_pix, nn_pix, 3)) #这里的3存的是一个点的xyz信息
    for i in range(len(kp_list)):
        kp_x, kp_y = kp_list[i][0], kp_list[i][1]
        gap = int((nn_pix - 1)/2)
        kp_lt_x, kp_lt_y = int(kp_x - gap), int(kp_y - gap) #left top cornor point
        #纠正方格块，如果方格块有部分已经超出图像边缘，则把左上角坐标移动回到图像边缘
        if kp_lt_x < 0:
            kp_lt_x = 0
        if kp_lt_y < 0:
            kp_lt_y = 0
        if live:
            dp_frame_x = live_frame.get_width()
            dp_frame_y = live_frame.get_height()
        else:
            dp_frame_x = depth_frame.shape[1]
            dp_frame_y = depth_frame.shape[0]
        #纠正方格块，如果方格块有部分已经超出图像边缘,防止方块右边超出边缘
        kp_most_x = kp_lt_x + (nn_pix - 1)
        if kp_most_x >= dp_frame_x:
            x_diff = kp_most_x - dp_frame_x
            kp_lt_x = kp_lt_x - x_diff - 1
        kp_most_y = kp_lt_y + (nn_pix - 1)
        if kp_most_y >= dp_frame_y:
            y_diff = kp_most_y - dp_frame_y
            kp_lt_y = kp_lt_y - y_diff - 1

        for t in range(nn_pix):
            for p in range(nn_pix):
                # if kp_lt_y + t < 0 or kp_lt_y + t >= depth_frame.shape[0] or kp_lt_x + p < 0 or kp_lt_x + p >= depth_frame.shape[1]:  #判断加上nn pix的时候是否会超出范围，超出范围的取0
                #     dp_nparray[i][t][p] = [kp_lt_x + p, kp_lt_y + t, 0]
                # else:
                if live:
                    dp_frame_val = live_frame.get_distance(kp_lt_x + p, kp_lt_y + t)
                else:
                    dp_frame_val = depth_frame[kp_lt_y + t, kp_lt_x + p]
                dp_nparray[i][t][p] = [kp_lt_x + p, kp_lt_y + t, dp_frame_val]
            # kp_lt_y = kp_lt_y + 1
    return dp_nparray

def filter_kp_list(kp_list, norm_p_index=0):
    if kp_list[norm_p_index][0] == 0 or kp_list[norm_p_index][1] == 0:
        return True
    return False

def get_mean_dep_val_xyz(kp_list, dp_nparray):
    dp_list = []
    for i in range(len(dp_nparray)):
        len_size = len(dp_nparray[i][0])
        dp_val = 0
        # print("len_size:",len_size)
        for j in range(len_size):
            for t in range(len_size):
                # print("nparray",dp_nparray[i][j][t])
                dp_val = dp_val + dp_nparray[i][j][t][-1]
        dp_mean = dp_val / (len_size**2)
        dp_list.append([kp_list[i][0], kp_list[i][1], dp_mean])
        # print("mean",[kp_list[i][0], kp_list[i][1], dp_mean])
    return dp_list

def norm_keypoints_xyz(dp_list, norm_p_index, rgb_size):
    '''
    (x, y) is used to normalizate.
    OUTPUT will be reshape as (3,5)
    '''
    x_max, y_max = rgb_size[0], rgb_size[1]
    dp_nparray = np.asarray(dp_list)
    dp_rs_array = dp_nparray.T
    z_max = dp_rs_array[2].max()
    z_min = dp_rs_array[2].min()
    dp_rs_array[0] = dp_rs_array[0] / x_max#最大最小归一化。x坐标/图像长度width，最小位置为0，直接省略掉了
    dp_rs_array[1] = dp_rs_array[1] / y_max
    dp_rs_array[2] = (dp_rs_array[2] - z_min) / (z_max - z_min)
    # 距离摄像机远近不同，切割出来的人像大小会有所不同，但是五个骨骼点的深度差是不变的，骨骼点深度差只与人的动作有关(在realsense返回的深度值准确的前提下)
    # z 深度是需要归一化的，需要变成 [0, 1] ，这样的情况下才能去掉量纲，和另外两个维度可以一起在网络中训练
    zero_diff = [[dp_rs_array[0][norm_p_index]], [dp_rs_array[1][norm_p_index]], [dp_rs_array[2][norm_p_index]]]
    # print("zero diff : ",zero_diff)
    dp_array = np.subtract(dp_rs_array, zero_diff)
    # print("dp array :\n ", dp_array)
    return dp_array

def no_negative(array):
    min_diff = [[array[0].min()], [array[1].min()], [array[2].min()]]
    array_out = array - min_diff
    return array_out

def filter_zero(listin, threadhold):
    list_shape = listin.shape
    for i in range(list_shape[0]):
        for j in range(list_shape[1]):
            if listin[i][j] < threadhold:
                listin[i][j] = 0
    return listin

def get_action_name(label):
    # label_dict = {0: "walk", 1: "stop", 2: "come", 3: "phone", 4: "open", 5: "stand"}
    output = []
    for _y in label:
        y_list = _y.tolist()
        label_pred = y_list.index(max(y_list))
        output.append(label_pred)
    # print(label_pred)
    return output

def my_mlp(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(units=20, activation=tf.nn.relu, use_bias=True)(x)
    x = tf.keras.layers.Dense(units=35, activation=tf.nn.relu, use_bias=True)(x)
    x = tf.keras.layers.Dense(units=15, activation=tf.nn.relu, use_bias=True)(x)
    x = tf.keras.layers.Dense(units=6)(x)
    # x = tf.keras.layers.Dense(units=20, activation=tf.nn.relu, use_bias=True)(x)
    # x = tf.keras.layers.Dense(units=10, activation=tf.nn.relu, use_bias=True)(x)
    # x = tf.keras.layers.Dense(units=6)(x)
    # x = tf.keras.layers.Dense(units=20, activation=tf.nn.relu, use_bias=True)(x)#mlp3
    # x = tf.keras.layers.Dense(units=35, activation=tf.nn.relu, use_bias=True)(x)
    # x = tf.keras.layers.Dense(units=15, activation=tf.nn.relu, use_bias=True)(x)
    # x = tf.keras.layers.Dense(units=6)(x)
    outputs = tf.keras.layers.Softmax()(x)
    return outputs


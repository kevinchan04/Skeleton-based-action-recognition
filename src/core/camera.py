import cv2
import random
import colorsys
import numpy as np
import copy
from core.config import cfg
import pyrealsense2 as rs
import json

def wait4cam_init(rspipeline, WaitTimes):
    for i in range(WaitTimes): 
        tmp_frame = rspipeline.wait_for_frames()

def find_device_that_supports_advanced_mode():
    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07","0B3A"]
    ctx = rs.context()
    ds5_dev = rs.device()
    devices = ctx.query_devices()
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
    raise Exception("No device that supports advanced mode was found")

def try_to_advanced_mode(json_file_path):
    dev = find_device_that_supports_advanced_mode()
    advnc_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # Loop until we successfully enable advanced mode
    while not advnc_mode.is_enabled():
        print("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")
    
    # Load json setting file
    jsonload = json.load(open(json_file_path))
    json_string = str(jsonload).replace("'", '\"')
    advnc_mode.load_json(json_string)
    return dev, advnc_mode

def get_depth_distance(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        # print the distance of depth for pixel of mouse
        print("point: " + str(x) + ", " + str(y))
        print("pixel: ", aligned_depth_8bit[y, x])
        print("dis: " + str(aligned_depth_frame.get_distance(x, y)))
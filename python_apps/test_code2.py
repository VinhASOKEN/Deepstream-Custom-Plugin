import os
import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
import ctypes
from ctypes import *
import time
import sys
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
from gi.repository import GLib, Gst, GstRtspServer
import cv2

fps_streams = {}

DEFAULT_PROCESSING_WIDTH = 1920
DEFAULT_PROCESSING_HEIGHT = 1080
OUTPUT_STREAM_RTSP = 1

PROCESSING_WIDTH = -1
PROCESSING_HEIGHT = -1

INPUT_URL = ""
dict_cache_box = {}
dict_cache_check_id = {}
dict_cache_check_frame_for_id = {}
res_done_id = []

p_down_right = [1500, 0]
p_down_left = [430, 0]
p_up = [0, 700]

def bbox_parser(obj_meta_info):
    """
        Reformat
        Input: muxed_size: image size from mux
               base_size: original image size from video
    """
    global PROCESSING_WIDTH
    global PROCESSING_HEIGHT
    left   = obj_meta_info.rect_params.left
    top    = obj_meta_info.rect_params.top
    width  = obj_meta_info.rect_params.width
    height = obj_meta_info.rect_params.height

    return [
        max(0, int(left)), 
        max(0, int(top)), 
        min(int(left + width), DEFAULT_PROCESSING_WIDTH-1), 
        min(int(top + height), DEFAULT_PROCESSING_HEIGHT-1)
    ]

def is_turning(lst_box):
    global p_down_right, p_down_left, p_up

    count_pass_left = 0
    count_pass_right = 0
    count_pass_up = 0

    for box in lst_box:
        x, y = box[0], box[1]

        if y > p_up[1] and x > p_down_right[0] and count_pass_up == 0:
            count_pass_right += 1
        
        if y > p_up[1] and x < p_down_left[0] and count_pass_up == 0:
            count_pass_left += 1

        if x < p_down_right[0] and x > p_down_left[0] and y < p_up[1] and (count_pass_left > 0 or count_pass_right > 0):
            count_pass_up += 1

    print(count_pass_left, count_pass_right, count_pass_up)
    if (count_pass_left > 1 and count_pass_up > 1) or (count_pass_right > 1 and count_pass_up > 1):
        return True

    return False

def check_2_dict(dict_id, dict_frame_for_id):
    res_done_key = []
    for key in dict_id:
        if dict_frame_for_id[key] - dict_id[key] > 30:
            res_done_key.append(key)

    return res_done_key

output_dir = '/Vinh_Deepstream/extract_frames_custom_clean_danang_capture_0'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0

def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global count
    global dict_cache_box, dict_cache_check_id, dict_cache_check_frame_for_id, res_done_id
    
    gst_buffer = info.get_buffer()

    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Get frames
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_copy = np.array(n_frame, copy=False, order='C')
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id == 4:
                x1, y1, x2, y2 = bbox_parser(obj_meta)
                obj_id = obj_meta.object_id
                if obj_id not in dict_cache_box:
                    dict_cache_box[obj_id] = []
                else:
                    dict_cache_box[obj_id].append([x1, y1, x2, y2, frame_copy])

                if obj_id not in dict_cache_check_id:
                    dict_cache_check_id[obj_id] = 1
                    dict_cache_check_frame_for_id[obj_id] = 1
                else:
                    dict_cache_check_id[obj_id] += 1

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break


        frame_number = frame_meta.frame_num
        source_id    = frame_meta.source_id 
        fps_streams["stream{0}".format(source_id)].get_fps()

        try:
            l_frame = l_frame.next
            for key in dict_cache_check_frame_for_id:
                dict_cache_check_frame_for_id[key] += 1

            res_done_key = check_2_dict(dict_cache_check_id, dict_cache_check_frame_for_id)
            for obj_id in res_done_key:
                if obj_id not in res_done_id:
                    res_check = is_turning(dict_cache_box[obj_id])
                    if res_check:
                        print("FOUND ONE !")
                        out_dir = os.path.join(output_dir, str(obj_id))
                        os.makedirs(out_dir, exist_ok=True)

                        for box in dict_cache_box[obj_id]:
                            x1, y1, x2, y2, frame_copy = box
                            object_img = frame_copy[y1:y2, x1:x2]
                            count += 1
                            output_path = os.path.join(out_dir, f"object_{count}.jpg")
                            cv2.imwrite(output_path, object_img)
                    # else:
                    #     print(len(dict_cache_check_id))
                    #     print(len(dict_cache_check_frame_for_id))
                    #     print(len(dict_cache_box))
                    res_done_id.append(obj_id)
                
                dict_cache_check_id.pop(obj_id)
                dict_cache_check_frame_for_id.pop(obj_id)
                dict_cache_box.pop(obj_id)

        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    global PROCESSING_WIDTH, PROCESSING_HEIGHT, DEFAULT_PROCESSING_WIDTH, DEFAULT_PROCESSING_HEIGHT
    global INPUT_URL


    INPUT_URL = '/Vinh_Deepstream/cam_danang/capture_0.mp4'
    # Parse input
    is_live = False
    if INPUT_URL[:4] == 'rtsp' or INPUT_URL[:4] == 'rtmp':
        lst_camera_url = [INPUT_URL]
        is_live = True
    else:
        lst_camera_url = ['file:{}'.format(INPUT_URL)]
        is_live = False
    fps_streams["stream0"] = GETFPS(0)
    
    # lst_camera_url = ['rtsp://admin:bgg082017@113.160.219.173:554/cam/realmonitor?channel=1&subtype=0']
  
    GObject.threads_init()
    Gst.init(None)

    # Create Pipeline element that will form a connection of other elements
    cmd = 'uridecodebin uri=%s name=demux1 ! queue ! nvvideoconvert name=nvvideoconver1 ! video/x-raw(memory:NVMM), format=RGBA ! mux1.sink_0 nvstreammux sync-inputs=1 name=mux1 \
    ! queue ! nvinfer name=pgie ! nvtracker name=tracker ! nvvideoconvert ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h265enc name=vencoder ! h265parse ! mpegtsmux name=avmux ! rtpmp2tpay \
    ! queue ! udpsink name=vsink demux1. ! queue ! audioconvert ! audioresample ! voaacenc ! audio/mpeg,mpegversion=4,stream-format=raw ! queue ! avmux. ' % (lst_camera_url[0])

    print('  --> CMD:', cmd)
    pipeline = Gst.parse_launch(
        cmd
    )
    

    ## Customize each element
    # Video convert
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    nvvideoconver1 = pipeline.get_by_name("nvvideoconver1")
    nvvideoconver1.set_property("nvbuf-memory-type", mem_type)

    # Streamux
    streammux = pipeline.get_by_name("mux1")
    streammux.set_property('width', DEFAULT_PROCESSING_WIDTH)
    streammux.set_property('height', DEFAULT_PROCESSING_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property("live-source", 1)
    streammux.set_property("sync-inputs", "true")
    streammux.set_property("nvbuf-memory-type", mem_type)

    #Tracker
    tracker = pipeline.get_by_name("tracker")
    config = configparser.ConfigParser()
    config.read('/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_tracker.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    # PGIE
    pgie = pipeline.get_by_name("pgie")
    pgie.set_property('config-file-path', "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_infer_yolov4_tiny.txt")

    # Video-encoder
    streaming_x264enc = pipeline.get_by_name("vencoder")
    streaming_x264enc.set_property("bitrate", 4000000)
    streaming_x264enc.set_property("gpu-id", 0)

    # UDPSink
    updsink_port_num = 5401
    streaming_sink = pipeline.get_by_name("vsink")
    streaming_sink.set_property("host", "10.9.3.239")
    streaming_sink.set_property("port", updsink_port_num)
    streaming_sink.set_property("sync", 0)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    sgie_src_pad = tracker.get_static_pad("src")
    if not sgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
    
    # Start streaming
    if OUTPUT_STREAM_RTSP:
        rtsp_port_num = 5004
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % rtsp_port_num
        server.attach(None)
        factory = GstRtspServer.RTSPMediaFactory.new()

        # Work but jerky
        factory.set_launch(
            "( "
          "udpsrc port=%d "
          "caps = \"application/x-rtp,media=video,clock-rate=90000,encoding-name=%s,payload=96\" ! "
          "rtpmp2tdepay ! rtpmp2tpay name=pay0 " ")" % (updsink_port_num, 'MP2T')
        )
        

        factory.set_shared(True)
        server.get_mount_points().add_factory("/ds-test", factory)

        print(
            "\n *** DeepStream: Launched RTSP Streaming at rtsp://10.9.3.239:%d/ds-test ***\n\n"
            % rtsp_port_num
        )


    # List the sources

    print("Starting pipeline") 
    pipeline.set_state(Gst.State.PLAYING)
    loop.run()

    # cleanup
    print("Exiting app")
    pipeline.set_state(Gst.State.NULL)

    del pipeline
    del loop
    return True

if __name__ == '__main__':
    main()

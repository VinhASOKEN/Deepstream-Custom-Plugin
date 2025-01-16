import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds
from gi.repository import GLib, Gst, GstRtspServer
import time

fps_streams = {}

DEFAULT_PROCESSING_WIDTH = 1920
DEFAULT_PROCESSING_HEIGHT = 1080
OUTPUT_STREAM_RTSP = 1

PROCESSING_WIDTH = -1
PROCESSING_HEIGHT = -1

INPUT_URL = ""

def tiler_sink_pad_buffer_probe(pad, info, u_data):
    """
        Tiler sink pad
            - LPD and LPR
    """

    tik = time.time()
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

        frame_number = frame_meta.frame_num
        source_id    = frame_meta.source_id 
        fps_streams["stream{0}".format(source_id)].get_fps()

        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    tok = time.time()
    fps = 1/(tok - tik)

    return Gst.PadProbeReturn.OK


def main():
    global PROCESSING_WIDTH, PROCESSING_HEIGHT, DEFAULT_PROCESSING_WIDTH, DEFAULT_PROCESSING_HEIGHT
    global INPUT_URL


    # INPUT_URL = '/Vinh_Deepstream/deepstream_python_apps/apps/a.mp4'
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
    lst_camera_url = ['rtsp://admin:bgg082017@14.241.37.250:554/cam/realmonitor?channel=1&subtype=0']
  
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)


    # cmd = 'uridecodebin uridecodebin::source::latency=2000 uridecodebin::source::drop-on-latency=true uri=%s name=demux1 \
    # ! queue ! mux1.sink_0 nvstreammux sync-inputs=1 name=mux1 ! queue ! nvinferlpr name=pgie ! nvinferlpr name=sgie_lpd ! nvinferlpr name=sgie_lpr \
    # ! nvtracker name=tracker ! icdslpr name=icdslpr \
    # ! queue ! nvmultistreamtiler name=tiler ! nvdsosd name=osd ! nvvideoconvert name=nvvideoconver2 \
    # ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h264enc name=vencoder ! h264parse ! rtph264pay ! queue ! udpsink name=vsink ' % (lst_camera_url[0])
    
    cmd = 'uridecodebin uridecodebin::source::latency=2000 uridecodebin::source::drop-on-latency=true uri=%s name=demux1 \
    ! queue ! mux1.sink_0 nvstreammux sync-inputs=1 name=mux1 ! queue ! nvinfer name=pgie ! nvtracker name=tracker \
    ! queue ! nvmultistreamtiler name=tiler ! nvdsosd name=osd ! nvvideoconvert name=nvvideoconver2 \
    ! video/x-raw(memory:NVMM), format=I420 ! nvv4l2h264enc name=vencoder ! h264parse ! rtph264pay ! queue ! udpsink name=vsink ' % (lst_camera_url[0])

    print('  --> CMD:', cmd)
    pipeline = Gst.parse_launch(cmd)


    ## Customize each element
    # Streamux
    streammux = pipeline.get_by_name("mux1")
    streammux.set_property('width', DEFAULT_PROCESSING_WIDTH)
    streammux.set_property('height', DEFAULT_PROCESSING_HEIGHT)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    
    streammux.set_property("sync-inputs", "true")

    if is_live:
        streammux.set_property("live-source", 1)
    # streammux.set_property("nvbuf-memory-type", mem_type)

    # Tracker
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
    # pgie.set_property('config-file-path', "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_infer_yolov4_tiny.txt")
    pgie.set_property('config-file-path', "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_infer_yolov8_detect_7class.txt")

    # # SGIE LICENSE PLATE DETECTION
    # sgie_lpd = pipeline.get_by_name("sgie_lpd")
    # sgie_lpd.set_property('config-file-path', "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_infer_lpd.txt")

    # # SGIE LICENSE PLATE RECOGNITION
    # sgie_lpr = pipeline.get_by_name("sgie_lpr")
    # sgie_lpr.set_property('config-file-path', "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream_custom/config_infer_lpr.txt")
    
    # icdslpr = pipeline.get_by_name("icdslpr")
    # icdslpr.set_property("config-file", "/Vinh_Deepstream/deepstream_python_apps/apps/deepstream-nvdsanalytics/config_nvdsanalytics_custom.txt")

    sgie_src_pad = tracker.get_static_pad("src")
    if not sgie_src_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # Tiler
    tiler = pipeline.get_by_name("tiler")
    tiler.set_property("rows", 1)
    tiler.set_property("columns", 1)
    tiler.set_property("width", DEFAULT_PROCESSING_WIDTH)
    tiler.set_property("height", DEFAULT_PROCESSING_HEIGHT)

    # Onscreen display
    nvosd = pipeline.get_by_name("osd")
    nvosd.set_property('process-mode', 1)
    nvosd.set_property('display-text', 1)

    # Video-encoder
    streaming_x264enc = pipeline.get_by_name("vencoder")
    streaming_x264enc.set_property("bitrate", 4000000)
    streaming_x264enc.set_property("gpu-id", 0)

    # UDPSink
    # updsink_port_num = 5600
    updsink_port_num = 5601
    streaming_sink = pipeline.get_by_name("vsink")
    streaming_sink.set_property("host", "10.9.3.239")
    streaming_sink.set_property("port", updsink_port_num)
    if not is_live:
        streaming_sink.set_property("sync", 1)
    else:
        streaming_sink.set_property("sync", 0)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    
    # Start streaming
    if OUTPUT_STREAM_RTSP:
        rtsp_port_num = 5066
        # rtsp_port_num = 5069
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % rtsp_port_num
        server.attach(None)
        factory = GstRtspServer.RTSPMediaFactory.new()

        factory.set_launch(
            '( udpsrc name=pay0 auto-multicast=0 port=%d buffer-size=524288 do-timestamp=true caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 " )'
            % (updsink_port_num, 'H264')
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
    tik = time.time()
    main()
    tok = time.time()
    res = (tok - tik)
    print("ADD PLUGIN: ", res)

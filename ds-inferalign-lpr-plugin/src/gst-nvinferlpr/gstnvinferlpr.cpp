#include <string.h>
#include <sstream>
#include <sys/time.h>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <list>
#include <thread>
#include <vector>

#include "gst-nvevent.h"
#include "gst-nvdscustomevent.h"
#include "gstnvdsmeta.h"
#include "nvdspreprocess_meta.h"

#include "gstnvinferlpr.h"
#include "gstnvinferlpr_allocator.h"
#include "gstnvinferlpr_meta_utils.h"
#include "gstnvinferlpr_property_parser.h"
#include "gstnvinferlpr_impl.h"

using namespace gstnvinferlpr;
using namespace nvdsinfer;

#define CLIP(a, min, max) (MAX(MIN(a, max), min))

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
  } \
} while (0)


GST_DEBUG_CATEGORY (gst_nvinferlpr_debug);
#define GST_CAT_DEFAULT gst_nvinferlpr_debug

#define INTERNAL_BUF_POOL_SIZE 3

#define NVDSINFER_CTX_OUT_POOL_SIZE_FLOW_META 6

/* Tracked objects will be reinferred only when their area in terms of pixels
 * increase by this ratio. */
#define REINFER_AREA_THRESHOLD 0.2

/* Tracked objects in the infer history map will be removed if they have not
 * been accessed for at least this number of frames. The tracker would definitely
 * have dropped references to an unseen object by 150 frames. */
#define CLEANUP_ACCESS_CRITERIA 150

/* Object history map cleanup interval. 1800 frames is a minute with a 30fps input */
#define MAP_CLEANUP_INTERVAL 1800

#define PROCESS_MODEL_FULL_FRAME 1
#define PROCESS_MODEL_OBJECTS 2

/* Warn about untracked objects in async mode every 5 minutes. */
#define UNTRACKED_OBJECT_WARN_INTERVAL (GST_SECOND * 60 * 5)

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

extern const int DEFAULT_REINFER_INTERVAL = G_MAXINT;

#define DS_NVINFERLPR_IMPL(gst_nvinferlpr) reinterpret_cast<DsNvInferImpl*>((gst_nvinferlpr)->impl)

#define IS_DETECTOR_INSTANCE(nvinferlpr) \
  (DS_NVINFERLPR_IMPL(nvinferlpr)->m_InitParams->networkType == NvDsInferNetworkType_Detector)
#define IS_CLASSIFIER_INSTANCE(nvinferlpr) \
  (DS_NVINFERLPR_IMPL(nvinferlpr)->m_InitParams->networkType == NvDsInferNetworkType_Classifier)
#define IS_SEGMENTATION_INSTANCE(nvinferlpr) \
  (DS_NVINFERLPR_IMPL(nvinferlpr)->m_InitParams->networkType == NvDsInferNetworkType_Segmentation)
#define IS_INSTANCE_SEGMENTATION_INSTANCE(nvinferlpr) \
  (DS_NVINFERLPR_IMPL(nvinferlpr)->m_InitParams->networkType == NvDsInferNetworkType_InstanceSegmentation)

static GQuark _dsmeta_quark = 0;

/* Gst-nvinferlpr supports runtime model updates. Refer to gstnvinferlpr_impl.h
 * for details. */

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESS_MODE PROCESS_MODEL_FULL_FRAME
#define DEFAULT_CONFIG_FILE_PATH ""
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_INTERVAL 0
#define DEFAULT_OPERATE_ON_GIE_ID -1
#define DEFAULT_GPU_DEVICE_ID 0
#define DEFAULT_OUTPUT_WRITE_TO_FILE FALSE
#define DEFAULT_OUTPUT_TENSOR_META FALSE
#define DEFAULT_OUTPUT_INSTANCE_MASK FALSE
#define DEFAULT_INPUT_TENSOR_META FALSE
#define DEFAULT_CLIP_OBJECT_OUTSIDE_ROI TRUE
#define DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY FALSE

#define NVTX_DEEPBLUE_COLOR 0xFF667EBE

/* Custom Alignment */
#define DEFAULT_ALIGNMENT_TYPE 0

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_nvinferlpr_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_nvinferlpr_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM, "{ NV12, RGBA }")));

guint gst_nvinferlpr_signals[LAST_SIGNAL] = { 0 };

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_nvinferlpr_parent_class parent_class
G_DEFINE_TYPE (GstNvInferLpr, gst_nvinferlpr, GST_TYPE_BASE_TRANSFORM);

/* Implementation of the GObject/GstBaseTransform interfaces. */
static void gst_nvinferlpr_finalize (GObject * object);
static void gst_nvinferlpr_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_nvinferlpr_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static gboolean gst_nvinferlpr_start (GstBaseTransform * btrans);
static gboolean gst_nvinferlpr_stop (GstBaseTransform * btrans);
static gboolean gst_nvinferlpr_sink_event (GstBaseTransform * trans, GstEvent * event);
static GstFlowReturn gst_nvinferlpr_submit_input_buffer (GstBaseTransform * btrans, gboolean discont, GstBuffer * inbuf);
static GstFlowReturn gst_nvinferlpr_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);
static gpointer gst_nvinferlpr_input_queue_loop (gpointer data);
static gpointer gst_nvinferlpr_output_loop (gpointer data);
static void gst_nvinferlpr_reset_init_params (GstNvInferLpr * nvinferlpr);

/* Create enum type for the process mode property. */
#define GST_TYPE_NVDSINFER_PROCESS_MODE (gst_nvinferlpr_process_mode_get_type ())

static GType
gst_nvinferlpr_process_mode_get_type (void)
{
  static volatile gsize process_mode_type = 0;
  static const GEnumValue process_mode[] = {
    {PROCESS_MODEL_FULL_FRAME, "Primary (Full Frame)", "primary"},
    {PROCESS_MODEL_OBJECTS, "Secondary (Objects)", "secondary"},
    {0, nullptr, nullptr}
  };
  if (g_once_init_enter (&process_mode_type)) {
    GType tmp = g_enum_register_static ("GstNvInferLprProcessModeType",
        process_mode);
    g_once_init_leave (&process_mode_type, tmp);
  }
  return (GType) process_mode_type;
}

static inline int
get_element_size (NvDsInferDataType data_type)
{
  switch (data_type) {
    case FLOAT:
      return 4;
    case HALF:
      return 2;
    case INT32:
      return 4;
    case INT8:
      return 1;
    default:
      return 0;
  }
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_nvinferlpr_class_init (GstNvInferLprClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_nvinferlpr_finalize);
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_nvinferlpr_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_nvinferlpr_get_property);

  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvinferlpr_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvinferlpr_stop);
  gstbasetransform_class->sink_event =
      GST_DEBUG_FUNCPTR (gst_nvinferlpr_sink_event);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_nvinferlpr_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_nvinferlpr_generate_output);

  /* Install properties. Values set through these properties override the ones in
   * the config file. */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element. Can be used to "
          "identify output of the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_MODE,
      g_param_spec_enum ("process-mode", "Process Mode",
          "Infer processing mode", GST_TYPE_NVDSINFER_PROCESS_MODE,
          DEFAULT_PROCESS_MODE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Config File Path",
          "Path to the configuration file for this instance of nvinferlpr",
          DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for inference",
          1, NVDSINFER_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_INTERVAL,
      g_param_spec_uint ("interval", "Interval",
          "Specifies number of consecutive batches to be skipped for inference",
          0, G_MAXINT, DEFAULT_INTERVAL,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_GIE_ID,
      g_param_spec_int ("infer-on-gie-id", "Infer on Gie ID",
          "Infer on metadata generated by GIE with this unique ID.\n"
          "\t\t\tSet to -1 to infer on all metadata.",
          -1, G_MAXINT, DEFAULT_OPERATE_ON_GIE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_CLASS_IDS,
      g_param_spec_string ("infer-on-class-ids", "Operate on Class ids",
          "Operate on objects with specified class ids\n"
          "\t\t\tUse string with values of class ids in ClassID (int) to set the property.\n"
          "\t\t\t e.g. 0:2:3",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property(gobject_class, PROP_FILTER_OUT_CLASS_IDS,
      g_param_spec_string ("filter-out-class-ids", "Ignore metadata for class ids",
            "Ignore metadata for objects of specified class ids\n"
            "\t\t\tUse string with values of class ids in ClassID (int) to set the property.\n"
            "\t\t\t e.g. 0;2;3",
            "",
            (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_MODEL_ENGINEFILE,
      g_param_spec_string ("model-engine-file", "Model Engine File",
          "Absolute path to the pre-generated serialized engine file for the model",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, DEFAULT_GPU_DEVICE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_WRITE_TO_FILE,
      g_param_spec_boolean ("raw-output-file-write", "Raw Output File Write",
          "Write raw inference output to file",
          DEFAULT_OUTPUT_WRITE_TO_FILE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_CALLBACK,
      g_param_spec_pointer ("raw-output-generated-callback",
          "Raw Output Generated Callback",
          "Pointer to the raw output generated callback funtion\n"
          "\t\t\t(type: gst_nvinfer_raw_output_generated_callback in 'gstnvdsinfer.h')",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_CALLBACK_USERDATA,
      g_param_spec_pointer ("raw-output-generated-userdata",
          "Raw Output Generated UserData",
          "Pointer to the userdata to be supplied with raw output generated callback",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_TENSOR_META,
      g_param_spec_boolean ("output-tensor-meta", "Output Tensor Meta",
          "Attach inference tensor outputs as buffer metadata",
          DEFAULT_OUTPUT_TENSOR_META,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_INSTANCE_MASK,
      g_param_spec_boolean ("output-instance-mask", "Output Instance Mask",
          "Instance mask expected in network output and attach it to metadata",
          DEFAULT_OUTPUT_INSTANCE_MASK,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_INPUT_TENSOR_META,
      g_param_spec_boolean ("input-tensor-meta", "Input Tensor Meta",
          "Use preprocessed input tensors attached as metadata instead of preprocessing inside the plugin",
          DEFAULT_INPUT_TENSOR_META,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ALIGNMENT_TYPE,
      g_param_spec_int ("alignment-type", "Alignment type",
          "Align surfaces before feed into infer.\n"
          "\t\t\tSet to 1 for face and to 2 for license plate.",
          -1, G_MAXINT, DEFAULT_ALIGNMENT_TYPE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLIP_OBJECT_OUTSIDE_ROI,
      g_param_spec_boolean ("clip-object-outside-roi", "Clip Object Outside Roi",
          "Clip the object bounding-box which lies outside the roi specified by nvdspreprosess plugin",
          DEFAULT_CLIP_OBJECT_OUTSIDE_ROI,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CROP_OBJECTS_TO_ROI_BOUNDARY,
      g_param_spec_boolean ("crop-objects-to-roi-boundary", "Crop Object to Roi Boundary",
          "Clip the object bounding-box which lies outside the roi boundary",
          DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  /** install signal MODEL_UPDATED */
  gst_nvinferlpr_signals[SIGNAL_MODEL_UPDATED] =
      g_signal_new ("model-updated",
      G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstNvInferLprClass, model_updated),
      NULL, NULL, NULL,
      G_TYPE_NONE, 2, G_TYPE_INT, G_TYPE_STRING);

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvinferlpr_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_nvinferlpr_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvInfer plugin",
      "NvInfer Plugin",
      "Nvidia DeepStreamSDK TensorRT plugin",
      "NVIDIA Corporation. Deepstream for Tesla forum: "
      "https://devtalk.nvidia.com/default/board/209");
}

static void
gst_nvinferlpr_init (GstNvInferLpr * nvinferlpr)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (nvinferlpr);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  nvinferlpr->impl = reinterpret_cast<GstNvInferLprImpl*>(new DsNvInferImpl(nvinferlpr));
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);

  /* Initialize all property variables to default values */
  nvinferlpr->unique_id = DEFAULT_UNIQUE_ID;
  nvinferlpr->process_full_frame = DEFAULT_PROCESS_MODE;
  nvinferlpr->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
  nvinferlpr->operate_on_class_ids = new std::vector < gboolean >;
  nvinferlpr->filter_out_class_ids = new std::set<uint>;
  nvinferlpr->output_tensor_meta = DEFAULT_OUTPUT_TENSOR_META;
  nvinferlpr->output_instance_mask = DEFAULT_OUTPUT_INSTANCE_MASK;
  nvinferlpr->clip_object_outside_roi = DEFAULT_CLIP_OBJECT_OUTSIDE_ROI;
  nvinferlpr->crop_objects_to_roi_boundary = DEFAULT_CLIP_OBJECT_TO_ROI_BOUNDARY;
  nvinferlpr->max_batch_size = impl->m_InitParams->maxBatchSize = DEFAULT_BATCH_SIZE;
  nvinferlpr->interval = DEFAULT_INTERVAL;
  nvinferlpr->operate_on_gie_id = DEFAULT_OPERATE_ON_GIE_ID;
  nvinferlpr->gpu_id = impl->m_InitParams->gpuID = DEFAULT_GPU_DEVICE_ID;
  nvinferlpr->is_prop_set = new std::vector < gboolean > (PROP_LAST, FALSE);

  nvinferlpr->untracked_object_warn_pts = GST_CLOCK_TIME_NONE;

  /* Set the default pre-processing transform params. */
  nvinferlpr->transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  nvinferlpr->transform_params.transform_filter = NvBufSurfTransformInter_Default;

  /* Custom Alignment*/
  nvinferlpr->alignment_type = DEFAULT_ALIGNMENT_TYPE;

  /* Create processing lock and condition for synchronization.*/
  g_mutex_init (&nvinferlpr->process_lock);
  g_cond_init (&nvinferlpr->process_cond);

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Free resources allocated during init. */
static void
gst_nvinferlpr_finalize (GObject * object)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (object);

  g_mutex_clear (&nvinferlpr->process_lock);
  g_cond_clear (&nvinferlpr->process_cond);

  delete nvinferlpr->perClassDetectionFilterParams;
  delete nvinferlpr->perClassColorParams;
  delete nvinferlpr->is_prop_set;
  g_free (nvinferlpr->config_file_path);
  delete nvinferlpr->operate_on_class_ids;
  delete nvinferlpr->filter_out_class_ids;

  delete DS_NVINFERLPR_IMPL(nvinferlpr);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_nvinferlpr_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (object);
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);

  if (prop_id < PROP_LAST) {
    /* Mark the property as being set through g_object_set. */
    (*nvinferlpr->is_prop_set)[prop_id] = TRUE;
  }

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      impl->m_InitParams->uniqueID = nvinferlpr->unique_id =
          g_value_get_uint (value);
      break;
    case PROP_PROCESS_MODE:
    {
      guint val = g_value_get_enum (value);
      nvinferlpr->process_full_frame = (val == PROCESS_MODEL_FULL_FRAME);
    }
      break;
    case PROP_CONFIG_FILE_PATH:
      {
        LockGMutex lock (nvinferlpr->process_lock);
        const std::string cfg_path (g_value_get_string (value));
        if (impl->isContextReady ()) {
            /* A NvDsInferContext is being used. Trigger a new model update. */
            impl->triggerNewModel (cfg_path, MODEL_LOAD_FROM_CONFIG);
            break;
        }
        g_free (nvinferlpr->config_file_path);
        nvinferlpr->config_file_path = g_value_dup_string (value);
        gst_nvinferlpr_reset_init_params (nvinferlpr);
        
        nvinferlpr->config_file_parse_successful = gst_nvinferlpr_parse_config_file (nvinferlpr, impl->m_InitParams.get(), nvinferlpr->config_file_path);
        
      }
      break;
    case PROP_OPERATE_ON_GIE_ID:
      nvinferlpr->operate_on_gie_id = g_value_get_int (value);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str (g_value_get_string (value));
      std::vector < gint > class_ids;
      gint max_class_id = -1;

      while (str.peek () != EOF) {
        gint class_id;
        str >> class_id;
        if(class_id < 0)
          continue;
        class_ids.push_back (class_id);
        max_class_id = MAX (max_class_id, class_id);
        str.get ();
      }
      nvinferlpr->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for (auto & cid:class_ids)
        nvinferlpr->operate_on_class_ids->at (cid) = TRUE;
    }
      break;
    case PROP_FILTER_OUT_CLASS_IDS:
    {
        std::stringstream str(g_value_get_string(value));
        nvinferlpr->filter_out_class_ids->clear();
        while(str.peek() != EOF) {
            gint class_id;
            str >> class_id;
            nvinferlpr->filter_out_class_ids->insert(class_id);
            str.get();
        }
    }
      break;
    case PROP_BATCH_SIZE:
      nvinferlpr->max_batch_size = impl->m_InitParams->maxBatchSize =
          g_value_get_uint (value);
      break;
    case PROP_INTERVAL:
      nvinferlpr->interval = g_value_get_uint (value);
      break;
    case PROP_MODEL_ENGINEFILE:
      {
        LockGMutex lock (nvinferlpr->process_lock);
        const std::string engine_path (g_value_get_string (value));
        if (impl->isContextReady ()) {
            /* A NvDsInferContext is being used. Trigger a new model update. */
            impl->triggerNewModel (engine_path, MODEL_LOAD_FROM_ENGINE);
            break;
        }
        g_strlcpy (impl->m_InitParams->modelEngineFilePath,
            g_value_get_string (value), _PATH_MAX);
      }
      break;
    case PROP_GPU_DEVICE_ID:
      nvinferlpr->gpu_id = impl->m_InitParams->gpuID = g_value_get_uint (value);
      break;
    case PROP_OUTPUT_WRITE_TO_FILE:
      nvinferlpr->write_raw_buffers_to_file = g_value_get_boolean (value);
      break;
    case PROP_OUTPUT_CALLBACK:
      nvinferlpr->output_generated_callback =
          (gst_nvinfer_raw_output_generated_callback)
          g_value_get_pointer (value);
      break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
      nvinferlpr->output_generated_userdata = g_value_get_pointer (value);
      break;
    case PROP_OUTPUT_TENSOR_META:
      nvinferlpr->output_tensor_meta = g_value_get_boolean (value);
      break;
    case PROP_OUTPUT_INSTANCE_MASK:
      nvinferlpr->output_instance_mask = g_value_get_boolean (value);
      break;
    case PROP_INPUT_TENSOR_META:
      nvinferlpr->input_tensor_from_meta = g_value_get_boolean (value);
      impl->m_InitParams->inputFromPreprocessedTensor =
          g_value_get_boolean (value);
      break;
    case PROP_CLIP_OBJECT_OUTSIDE_ROI:
      nvinferlpr->clip_object_outside_roi = g_value_get_boolean (value);
      break;
    case PROP_CROP_OBJECTS_TO_ROI_BOUNDARY:
      nvinferlpr->crop_objects_to_roi_boundary = g_value_get_boolean (value);
      break;
    case PROP_ALIGNMENT_TYPE:
      nvinferlpr->alignment_type = g_value_get_int (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_nvinferlpr_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (object);
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, nvinferlpr->unique_id);
      break;
    case PROP_PROCESS_MODE:
      g_value_set_enum (value,
          nvinferlpr->process_full_frame ? PROCESS_MODEL_FULL_FRAME :
          PROCESS_MODEL_OBJECTS);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, nvinferlpr->config_file_path);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      g_value_set_int (value, nvinferlpr->operate_on_gie_id);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str;
      for (size_t i = 0; i < nvinferlpr->operate_on_class_ids->size (); i++) {
        if (nvinferlpr->operate_on_class_ids->at (i))
          str << i << ":";
      }
      g_value_set_string (value, str.str ().c_str ());
    }
      break;
    case PROP_FILTER_OUT_CLASS_IDS:
    {
        std::stringstream str;
        for(const auto id : *nvinferlpr->filter_out_class_ids)
            str << id << ";";
        g_value_set_string (value, str.str ().c_str ());
    }
        break;
    case PROP_MODEL_ENGINEFILE:
      g_value_set_string (value, impl->m_InitParams->modelEngineFilePath);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, nvinferlpr->max_batch_size);
      break;
    case PROP_INTERVAL:
      g_value_set_uint (value, nvinferlpr->interval);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvinferlpr->gpu_id);
      break;
    case PROP_OUTPUT_WRITE_TO_FILE:
      g_value_set_boolean (value, nvinferlpr->write_raw_buffers_to_file);
      break;
    case PROP_OUTPUT_CALLBACK:
      g_value_set_pointer (value,
          (gpointer) nvinferlpr->output_generated_callback);
      break;
    case PROP_OUTPUT_CALLBACK_USERDATA:
      g_value_set_pointer (value, nvinferlpr->output_generated_userdata);
      break;
    case PROP_OUTPUT_TENSOR_META:
      g_value_set_boolean (value, nvinferlpr->output_tensor_meta);
      break;
    case PROP_OUTPUT_INSTANCE_MASK:
      g_value_set_boolean (value, nvinferlpr->output_instance_mask);
      break;
    case PROP_INPUT_TENSOR_META:
      g_value_set_boolean (value, nvinferlpr->input_tensor_from_meta);
      break;
    case PROP_CLIP_OBJECT_OUTSIDE_ROI:
      g_value_set_boolean (value, nvinferlpr->clip_object_outside_roi);
      break;
    case PROP_CROP_OBJECTS_TO_ROI_BOUNDARY:
      g_value_set_boolean (value, nvinferlpr->crop_objects_to_roi_boundary);
      break;
    case PROP_ALIGNMENT_TYPE:
      g_value_set_int (value, nvinferlpr->alignment_type);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

void gst_nvinferlpr_logger(NvDsInferContextHandle handle, unsigned int unique_id, NvDsInferLogLevel log_level,
    const char* log_message, void* user_ctx) {
    GstNvInferLpr* nvinferlpr = GST_NVINFERLPR(user_ctx);

    switch (log_level) {
    case NVDSINFER_LOG_ERROR:
        GST_ERROR_OBJECT(nvinferlpr, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_WARNING:
        GST_WARNING_OBJECT(nvinferlpr, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_INFO:
        GST_INFO_OBJECT(nvinferlpr, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
    case NVDSINFER_LOG_DEBUG:
        GST_DEBUG_OBJECT(nvinferlpr, "NvDsInferContext[UID %d]: %s", unique_id, log_message);
        return;
  }
}

/**
 * Reset m_InitParams structure while preserving property values set through
 * GObject set method. */
static void
gst_nvinferlpr_reset_init_params (GstNvInferLpr * nvinferlpr)
{
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL(nvinferlpr);
  auto prev_params = std::move(impl->m_InitParams);
  impl->m_InitParams.reset (new NvDsInferContextInitParams);
  assert (impl->m_InitParams);
  NvDsInferContext_ResetInitParams (impl->m_InitParams.get ());

  if (nvinferlpr->is_prop_set->at (PROP_MODEL_ENGINEFILE))
    g_strlcpy (impl->m_InitParams->modelEngineFilePath,
        prev_params->modelEngineFilePath, _PATH_MAX);

  if (nvinferlpr->is_prop_set->at (PROP_BATCH_SIZE))
    impl->m_InitParams->maxBatchSize = prev_params->maxBatchSize;

  if (nvinferlpr->is_prop_set->at (PROP_GPU_DEVICE_ID))
    impl->m_InitParams->gpuID = prev_params->gpuID;

  if (nvinferlpr->is_prop_set->at (PROP_UNIQUE_ID))
    impl->m_InitParams->uniqueID = prev_params->uniqueID;
  if (nvinferlpr->is_prop_set->at (PROP_INPUT_TENSOR_META))
    impl->m_InitParams->inputFromPreprocessedTensor =
        prev_params->inputFromPreprocessedTensor;

  delete prev_params->perClassDetectionParams;
  g_strfreev (prev_params->outputLayerNames);
  g_strfreev (prev_params->outputIOFormats);
  g_strfreev (prev_params->layerDevicePrecisions);
}

/**
 * Called when an event is recieved on the sink pad. We need to make sure
 * serialized events and buffers are pushed downstream while maintaining the order.
 * To ensure this, we push all the buffers in the internal queue to the
 * downstream element before forwarding the serialized event to the downstream element.
 */
static gboolean
gst_nvinferlpr_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (trans);
  gboolean ignore_serialized_event = FALSE;

  /** The TAG event is sent many times leading to drop in performance because of
   * buffer/event serialization. We can ignore such events which won't cause
   * issues if we don't serialize the events. */
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_TAG:
      ignore_serialized_event = TRUE;
      break;
    default:
      break;
  }

  /* Serialize events. Wait for pending buffers to be processed and pushed
   * downstream. No need to wait in case of classifier async mode since all
   * the buffers are already pushed downstream. */
  if (GST_EVENT_IS_SERIALIZED (event) && !ignore_serialized_event &&
      !nvinferlpr->classifier_async_mode) {
    GstNvInferLprBatch *batch = new GstNvInferLprBatch;
    batch->event_marker = TRUE;

    g_mutex_lock (&nvinferlpr->process_lock);
    /* Push the event marker batch in the processing queue. */
    if (nvinferlpr->input_queue_thread)
      g_queue_push_tail (nvinferlpr->input_queue, batch);
    else
      g_queue_push_tail (nvinferlpr->process_queue, batch);
    g_cond_broadcast (&nvinferlpr->process_cond);

    /* Wait for all the remaining batches in the queue including the event
     * marker to be processed. */
    while (!g_queue_is_empty (nvinferlpr->input_queue)) {
      g_cond_wait (&nvinferlpr->process_cond, &nvinferlpr->process_lock);
    }
    while (!g_queue_is_empty (nvinferlpr->process_queue)) {
      g_cond_wait (&nvinferlpr->process_cond, &nvinferlpr->process_lock);
    }
    g_mutex_unlock (&nvinferlpr->process_lock);
  }

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_ADDED) {
    /* New source added in the pipeline. Create a source info instance for it. */
    guint source_id;
    gst_nvevent_parse_pad_added (event, &source_id);
    nvinferlpr->source_info->emplace (source_id, GstNvInferLprSourceInfo ());
  }

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_PAD_DELETED) {
    /* Source removed from the pipeline. Remove the related structure. */
    guint source_id;
    gst_nvevent_parse_pad_deleted (event, &source_id);
    nvinferlpr->source_info->erase (source_id);
  }

  if ((GstNvEventType) GST_EVENT_TYPE (event) == GST_NVEVENT_STREAM_EOS) {
    /* Got EOS from a source. Clean up the object history map. */
    guint source_id;
    gst_nvevent_parse_stream_eos (event, &source_id);
    auto result = nvinferlpr->source_info->find (source_id);
    if (result != nvinferlpr->source_info->end ())
      result->second.object_history_map.clear ();
  }

  if ((GstNvDsCustomEventType)GST_EVENT_TYPE (event) == GST_NVEVENT_INFER_INTERVAL_UPDATE) {
    gchar* stream_id = NULL;

    g_mutex_lock (&nvinferlpr->process_lock);
    gst_nvevent_parse_infer_interval_update (event, &stream_id, &nvinferlpr->interval);
    g_mutex_unlock (&nvinferlpr->process_lock);
  }

  if (GST_EVENT_TYPE (event) == GST_EVENT_EOS) {
    nvinferlpr->interval_counter = 0;
  }

  /* Call the sink event handler of the base class. */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_nvinferlpr_start (GstBaseTransform * btrans)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (btrans);
  GstAllocationParams allocation_params;
  cudaError_t cudaReturn;
  NvBufSurfaceColorFormat color_format;
  NvDsInferStatus status;
  std::string nvtx_str;
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);
  NvDsInferContextHandle infer_context = nullptr;

  LockGMutex lock (nvinferlpr->process_lock);
  NvDsInferContextInitParams *init_params = impl->m_InitParams.get ();
  assert (init_params);

  nvtx_str = "GstNvInferLpr: UID=" + std::to_string(nvinferlpr->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  /* Providing a valid config file is mandatory. */
  if (!nvinferlpr->config_file_path || strlen (nvinferlpr->config_file_path) == 0) {
    GST_ELEMENT_ERROR (nvinferlpr, LIBRARY, SETTINGS,
        ("Configuration file not provided"), (nullptr));
    return FALSE;
  }
  if (nvinferlpr->config_file_parse_successful == FALSE) {
    GST_ELEMENT_ERROR (nvinferlpr, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", nvinferlpr->config_file_path));
    return FALSE;
  }

  if (nvinferlpr->output_instance_mask == TRUE &&
                 init_params->clusterMode != NVDSINFER_CLUSTER_NONE)
  {
    GST_ELEMENT_ERROR (nvinferlpr, LIBRARY, SETTINGS,
        ("Instance mask output not supported with cluster mode %d",
                               init_params->clusterMode), (nullptr));
    return FALSE;
  }

  nvinferlpr->interval_counter = 0;

  /* Should not infer on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT
   * since it will cause hardware scaling issues. */
  nvinferlpr->min_input_object_width =
      MAX(MIN_INPUT_OBJECT_WIDTH, nvinferlpr->min_input_object_width);
  nvinferlpr->min_input_object_height =
      MAX(MIN_INPUT_OBJECT_HEIGHT, nvinferlpr->min_input_object_height);

  /* Ask NvDsInferContext to copy the input layer contents to host memory if
   * CPU needs to access it. */
  init_params->copyInputToHostBuffers =
      (nvinferlpr->write_raw_buffers_to_file ||
      (nvinferlpr->output_generated_callback != nullptr));

  /* Set the number of output buffers that should be allocated by NvDsInferContext.
   * Should allocate more buffers if the output tensor buffers will be attached
   * as meta to GstBuffers and pushed downstream. */

  init_params->outputBufferPoolSize = std::max<uint>(init_params->outputBufferPoolSize,
                                        NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE);

  if (nvinferlpr->output_tensor_meta || IS_SEGMENTATION_INSTANCE (nvinferlpr))
    init_params->outputBufferPoolSize = std::max<uint>(init_params->outputBufferPoolSize,
                                        NVDSINFER_CTX_OUT_POOL_SIZE_FLOW_META);


  if (nvinferlpr->output_tensor_meta || init_params->autoIncMem == 0) {
      GST_ELEMENT_WARNING (nvinferlpr, LIBRARY, SETTINGS,
          ("NvInfer output-tensor-meta is enabled but init_params auto "
              "increase memory (auto-inc-mem) is disabled. The bufferpool "
              "will not be automatically resized."), (nullptr));
    }

  /* Create the NvDsInferContext instance. */
  status =
      createNvDsInferContext (&infer_context, *init_params,
      nvinferlpr, gst_nvinferlpr_logger);
  if (status != NVDSINFER_SUCCESS) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to create NvDsInferContext instance"),
        ("Config file path: %s, NvDsInfer Error: %s", nvinferlpr->config_file_path,
            NvDsInferStatus2Str (status)));
    return FALSE;
  }
  std::unique_ptr<INvDsInferContext> ctx_ptr (infer_context);

  /* Get the network resolution. */
  ctx_ptr->getNetworkInfo (nvinferlpr->network_info);
  nvinferlpr->network_width = nvinferlpr->network_info.width;
  nvinferlpr->network_height = nvinferlpr->network_info.height;

  /* Get information on all the bound layers. */
  nvinferlpr->layers_info = new std::vector < NvDsInferLayerInfo > ();
  ctx_ptr->fillLayersInfo (*nvinferlpr->layers_info);

  nvinferlpr->output_layers_info = new std::vector < NvDsInferLayerInfo > ();
  for (auto & layer:*(nvinferlpr->layers_info)) {
    if (!layer.isInput)
      nvinferlpr->output_layers_info->push_back (layer);
  }

  nvinferlpr->file_write_batch_num = 0;

  /* Create process queue and input queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  nvinferlpr->process_queue = g_queue_new ();
  nvinferlpr->input_queue = g_queue_new ();

  /* Set the NvBufSurfTransform config parameters. */
  nvinferlpr->transform_config_params.gpu_id = nvinferlpr->gpu_id;

  NvBufSurfTransformSetSessionParams (&nvinferlpr->transform_config_params);

  /* Based on the network input requirements decide the buffer pool color format. */
  switch (init_params->networkInputFormat) {
    case NvDsInferFormat_RGB:
    case NvDsInferFormat_BGR:
    if(nvinferlpr->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC) {
      color_format = NVBUF_COLOR_FORMAT_RGBA;
    }
    else {
      color_format = NVBUF_COLOR_FORMAT_RGB;
    }
    break;
    case NvDsInferFormat_GRAY:
    if(nvinferlpr->transform_config_params.compute_mode == NvBufSurfTransformCompute_VIC) {
      color_format = NVBUF_COLOR_FORMAT_NV12;
    }
    else {
      color_format = NVBUF_COLOR_FORMAT_GRAY8;
    }
    break;
    default:
      GST_ELEMENT_ERROR (nvinferlpr, LIBRARY, SETTINGS,
          ("Unsupported network input format: %d",
              init_params->networkInputFormat), (nullptr));
      return FALSE;
  }

  if (!nvinferlpr->input_tensor_from_meta) {
    /* Create a buffer pool for internal memory required for scaling frames to
    * network resolution / cropping objects. The pool allocates
    * INTERNAL_BUF_POOL_SIZE buffers at start and keeps reusing them. */
    auto pool_deleter = [](GstBufferPool *p) { if (p) gst_object_unref (p); };
    std::unique_ptr<GstBufferPool, decltype(pool_deleter)> pool_ptr (
        gst_buffer_pool_new (), pool_deleter);

    auto config_deleter = [](GstStructure *s) { if (s) gst_structure_free (s); };
    std::unique_ptr<GstStructure, decltype(config_deleter)> config_ptr (
        gst_buffer_pool_get_config (pool_ptr.get()), config_deleter);
    gst_buffer_pool_config_set_params (config_ptr.get(), nullptr,
        sizeof (GstNvInferLprMemory), INTERNAL_BUF_POOL_SIZE, INTERNAL_BUF_POOL_SIZE);

    /* Create a new GstNvInferLprAllocator instance. Allocator has methods to allocate
    * and free custom memories. */
    auto allocator_deleter = [](GstAllocator *a) { if (a) gst_object_unref (a); };
    std::unique_ptr<GstAllocator, decltype(allocator_deleter)> allocator_ptr (
        gst_nvinferlpr_allocator_new (nvinferlpr->network_width,
        nvinferlpr->network_height, color_format, nvinferlpr->max_batch_size,
        nvinferlpr->gpu_id),
        allocator_deleter);
    memset (&allocation_params, 0, sizeof (allocation_params));
    gst_buffer_pool_config_set_allocator (config_ptr.get (), allocator_ptr.get (),
        &allocation_params);

    if (!gst_buffer_pool_set_config (pool_ptr.get(), config_ptr.release())) {
      GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
          ("Failed to set config on buffer pool"), (nullptr));
      return FALSE;
    }

    /* Start the buffer pool and allocate all internal buffers. */
    if (!gst_buffer_pool_set_active (pool_ptr.get(), TRUE)) {
      GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
          ("Failed to set buffer pool to active"), (nullptr));
      return FALSE;
    }
    nvinferlpr->pool = pool_ptr.release ();
  }

  cudaReturn = cudaSetDevice (nvinferlpr->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinferlpr->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  cudaReturn =
      cudaStreamCreateWithFlags (&nvinferlpr->convertStream,
      cudaStreamNonBlocking);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to create cuda stream"),
        ("cudaStreamCreateWithFlags failed with error %s",
            cudaGetErrorName (cudaReturn)));
    return FALSE;
  }

  nvinferlpr->transform_config_params.cuda_stream = nvinferlpr->convertStream;

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  nvinferlpr->tmp_surf.surfaceList = new NvBufSurfaceParams[nvinferlpr->max_batch_size];
  nvinferlpr->tmp_surf.batchSize = nvinferlpr->max_batch_size;
  nvinferlpr->tmp_surf.gpuId = nvinferlpr->gpu_id;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  nvinferlpr->transform_params.src_rect =
      new NvBufSurfTransformRect[nvinferlpr->max_batch_size];
  nvinferlpr->transform_params.dst_rect =
      new NvBufSurfTransformRect[nvinferlpr->max_batch_size];
  nvinferlpr->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FLIP |
      NVBUFSURF_TRANSFORM_ALLOW_ODD_CROP;
  nvinferlpr->transform_params.transform_flip = NvBufSurfTransform_None;

  /* Initialize the object history map for source 0. */
  nvinferlpr->source_info = new std::unordered_map < gint, GstNvInferLprSourceInfo >;
  nvinferlpr->source_info->emplace (0, GstNvInferLprSourceInfo {
      GstNvInferLprObjectHistoryMap (), 0}
  );

  if (nvinferlpr->classifier_async_mode) {
    if (nvinferlpr->process_full_frame || !IS_CLASSIFIER_INSTANCE (nvinferlpr)) {
      GST_ELEMENT_WARNING (nvinferlpr, LIBRARY, SETTINGS,
          ("NvInfer asynchronous mode is applicable for secondary"
              "classifiers only. Turning off asynchronous mode"), (nullptr));
      nvinferlpr->classifier_async_mode = FALSE;
    }
  }

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  nvinferlpr->output_thread = g_thread_new ("nvinferlpr-output-thread", gst_nvinferlpr_output_loop, nvinferlpr);

  /* Start a thread which will queue input to the NvDsInfer context since
   * queueInputBatch is a blocking function. This is done to parallelize
   * input conversion and queueInputBatch. */
  if (!nvinferlpr->input_tensor_from_meta) {
    nvinferlpr->input_queue_thread = g_thread_new ("nvinferlpr-input-queue-thread", gst_nvinferlpr_input_queue_loop, nvinferlpr);
  }

  /* nvinferlpr internal resource start for loading models */
  impl->m_InferCtx = std::move (ctx_ptr);
  if (impl->start () != NVDSINFER_SUCCESS) {
      GST_ELEMENT_WARNING (nvinferlpr, RESOURCE, FAILED,
          ("NvInfer start loading model thread failed."), (nullptr));
      return FALSE;
  }

  nvinferlpr->nvtx_domain = nvtx_domain_ptr.release ();
  lock.unlock ();

  impl->notifyLoadModelStatus (
      ModelStatus {NVDSINFER_SUCCESS, nvinferlpr->config_file_path,
      "Model loaded successfully"});
  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_nvinferlpr_stop (GstBaseTransform * btrans)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (btrans);
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);

  LockGMutex locker (nvinferlpr->process_lock);
  /* Wait till all the items in the two queues are handled. */
  while (!g_queue_is_empty (nvinferlpr->input_queue)) {
    locker.wait (nvinferlpr->process_cond);
  }
  while (!g_queue_is_empty (nvinferlpr->process_queue)) {
    locker.wait (nvinferlpr->process_cond);
  }
  nvinferlpr->stop = TRUE;

  g_cond_broadcast (&nvinferlpr->process_cond);
  locker.unlock ();

  impl->stop ();

  if (nvinferlpr->input_queue_thread)
    g_thread_join (nvinferlpr->input_queue_thread);
  g_thread_join (nvinferlpr->output_thread);

  nvinferlpr->stop = FALSE;

  delete nvinferlpr->source_info;
  delete nvinferlpr->layers_info;
  delete nvinferlpr->output_layers_info;

  delete[] nvinferlpr->transform_params.src_rect;
  delete[] nvinferlpr->transform_params.dst_rect;
  delete[] nvinferlpr->tmp_surf.surfaceList;

  cudaSetDevice (nvinferlpr->gpu_id);

  if (nvinferlpr->convertStream)
    cudaStreamDestroy (nvinferlpr->convertStream);

  /* Free up the memory allocated by pool. */
  if (nvinferlpr->pool)
    gst_object_unref (nvinferlpr->pool);

  g_queue_free (nvinferlpr->process_queue);
  g_queue_free (nvinferlpr->input_queue);

  return TRUE;
}

/**
 * Calls the one of the required conversion functions based on the network
 * input format.
 */
static GstFlowReturn
get_converted_buffer (GstNvInferLpr * nvinferlpr, NvBufSurface * src_surf,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    NvBufSurface * dest_surf, NvBufSurfaceParams * dest_frame,
    gdouble & ratio_x, gdouble & ratio_y, guint & offset_left,
    guint & offset_top, void *destCudaPtr)
{
  guint src_left = GST_ROUND_UP_2 ((unsigned int)crop_rect_params->left);
  guint src_top = GST_ROUND_UP_2 ((unsigned int)crop_rect_params->top);
  guint src_width = GST_ROUND_DOWN_2 ((unsigned int)crop_rect_params->width);
  guint src_height = GST_ROUND_DOWN_2 ((unsigned int)crop_rect_params->height);
  guint dest_width, dest_height;

  guint offset_right = 0, offset_bottom = 0;
  offset_left = 0;
  offset_top = 0;
  if (nvinferlpr->maintain_aspect_ratio) {
    /* Calculate the destination width and height required to maintain
     * the aspect ratio. */
    double hdest = dest_frame->width * src_height / (double) src_width;
    double wdest = dest_frame->height * src_width / (double) src_height;
    int pixel_size;
    cudaError_t cudaReturn;

    if (hdest <= dest_frame->height) {
      dest_width = dest_frame->width;
      dest_height = hdest;
    } else {
      dest_width = wdest;
      dest_height = dest_frame->height;
    }

    switch (dest_frame->colorFormat) {
      case NVBUF_COLOR_FORMAT_RGBA:
        pixel_size = 4;
        break;
      case NVBUF_COLOR_FORMAT_RGB:
        pixel_size = 3;
        break;
      case NVBUF_COLOR_FORMAT_GRAY8:
      case NVBUF_COLOR_FORMAT_NV12:
        pixel_size = 1;
        break;
      default:
        g_assert_not_reached ();
        break;
    }
    /* Pad the scaled image with black color. */
    if (!nvinferlpr->symmetric_padding) {
      /* Right-Bottom Padding. */
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size * dest_width,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * (dest_frame->width - dest_width), dest_frame->height,
          nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * dest_height,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          dest_frame->height - dest_height, nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    } else {
      /* Symmetric Padding. */
      offset_left = (dest_frame->width - dest_width) / 2;
      offset_right = dest_frame->width - dest_width - offset_left;

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_left, dest_frame->height, nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr + pixel_size *
          (dest_width + offset_left),
          dest_frame->planeParams.pitch[0], 0,
          pixel_size * offset_right, dest_frame->height,
          nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      offset_top = (dest_frame->height - dest_height) / 2;
      offset_bottom = dest_frame->height - dest_height - offset_top;

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr,
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_top, nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }

      cudaReturn =
          cudaMemset2DAsync ((uint8_t *) destCudaPtr +
          dest_frame->planeParams.pitch[0] * (dest_height + offset_top),
          dest_frame->planeParams.pitch[0], 0, pixel_size * dest_width,
          offset_bottom, nvinferlpr->convertStream);
      if (cudaReturn != cudaSuccess) {
        GST_ERROR_OBJECT (nvinferlpr,
            "cudaMemset2DAsync failed with error %s while converting buffer",
            cudaGetErrorName (cudaReturn));
        return GST_FLOW_ERROR;
      }
    }
  } else {
    dest_width = nvinferlpr->network_width;
    dest_height = nvinferlpr->network_height;
  }
  /* Calculate the scaling ratio of the frame / object crop. This will be
   * required later for rescaling the detector output boxes to input resolution.
   */
  ratio_x = (double) dest_width / src_width;
  ratio_y = (double) dest_height / src_height;

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  nvinferlpr->tmp_surf.surfaceList[nvinferlpr->tmp_surf.numFilled] = *src_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  nvinferlpr->transform_params.src_rect[nvinferlpr->tmp_surf.numFilled] =
      { src_top, src_left, src_width, src_height };
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  if (!nvinferlpr->symmetric_padding) {
    nvinferlpr->transform_params.dst_rect[nvinferlpr->tmp_surf.numFilled] =
        { 0, 0, dest_width, dest_height };
  } else {
    nvinferlpr->transform_params.dst_rect[nvinferlpr->tmp_surf.numFilled] =
        { offset_top, offset_left, dest_width, dest_height };
  }

  nvinferlpr->tmp_surf.numFilled++;
  nvinferlpr->tmp_surf.memType = src_surf->memType;

  return GST_FLOW_OK;
}

/* Helper function to queue a batch for inferencing and push it to the element's
 * processing queue. */
static gpointer
gst_nvinferlpr_input_queue_loop (gpointer data)
{
  GstNvInferLpr *nvinferlpr = (GstNvInferLpr *) data;
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);
  std::string nvtx_str;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinferlpr->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinferlpr->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  LockGMutex locker (nvinferlpr->process_lock);

  while (nvinferlpr->stop == FALSE) {
    GstNvInferLprBatch *batch;
    GstNvInferLprMemory *mem;
    NvDsInferContextBatchInput input_batch;
    std::vector < void *>input_frames;
    unsigned int i;
    NvDsInferStatus status;

    /* Wait if input queue is empty. */
    if (g_queue_is_empty (nvinferlpr->input_queue)) {
      locker.wait (nvinferlpr->process_cond);
      continue;
    }
    batch = (GstNvInferLprBatch *) g_queue_pop_head (nvinferlpr->input_queue);
    NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;

    /* Check if this is a push buffer or event marker batch. If yes, no need to
     * queue the input for inferencing. */
    if (batch->push_buffer || batch->event_marker || batch->frames.size() == 0) {
      goto queue_batch;
    }

    mem = gst_nvinferlpr_buffer_get_memory (batch->conv_buf);

    /* Form the vector of input frame pointers. */
    for (i = 0; i < batch->frames.size (); i++) {
      input_frames.push_back (batch->frames[i].converted_frame_ptr);
    }

    input_batch.inputFrames = input_frames.data ();
    input_batch.numInputFrames = input_frames.size ();

    switch (mem->surf->surfaceList[0].colorFormat) {
      case NVBUF_COLOR_FORMAT_RGBA:
        input_batch.inputFormat = NvDsInferFormat_RGBA;
        break;
      case NVBUF_COLOR_FORMAT_RGB:
        input_batch.inputFormat = NvDsInferFormat_RGB;
        break;
      case NVBUF_COLOR_FORMAT_GRAY8:
      case NVBUF_COLOR_FORMAT_NV12:
        input_batch.inputFormat = NvDsInferFormat_GRAY;
        break;
      default:
        input_batch.inputFormat = NvDsInferFormat_Unknown;
        break;
    }
    input_batch.inputPitch = mem->surf->surfaceList[0].planeParams.pitch[0];

    input_batch.returnInputFunc =
        (NvDsInferContextReturnInputAsyncFunc) gst_buffer_unref;
    input_batch.returnFuncData = batch->conv_buf;

    locker.unlock ();

    nvtx_str = "queueInput batch_num=" + std::to_string(nvinferlpr->current_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvinferlpr->nvtx_domain, &eventAttrib);

    // Moved outside the lock, blocking prepstatusrocess thread
    if (batch->sync_obj){
      NvBufSurfTransformSyncObjWait (batch->sync_obj, -1);
      NvBufSurfTransformSyncObjDestroy (&(batch->sync_obj));
    }

    status = nvdsinfer_ctx->queueInputBatch (input_batch);

    nvtxDomainRangePop(nvinferlpr->nvtx_domain);

    locker.lock ();

    if (status != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
          ("Failed to queue input batch for inferencing"), (nullptr));
      continue;
    }

queue_batch:
    /* Push the batch info structure in the processing queue and notify the
     * output thread that a new batch has been queued. */
    g_queue_push_tail (nvinferlpr->process_queue, batch);
    g_cond_broadcast (&nvinferlpr->process_cond);
  }

  return NULL;
}

static gboolean
convert_batch_and_push_to_input_thread (GstNvInferLpr *nvinferlpr,
    GstNvInferLprBatch *batch, GstNvInferLprMemory *mem)
{
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  std::string nvtx_str;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinferlpr->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinferlpr->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  /* Set the transform session parameters for the conversions executed in this
   * thread. */
  err = NvBufSurfTransformSetSessionParams (&nvinferlpr->transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
    return FALSE;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(nvinferlpr->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(nvinferlpr->nvtx_domain, &eventAttrib);

  if (batch->frames.size() > 0) {
    /* Batched tranformation. */
    err = NvBufSurfTransformAsync (&nvinferlpr->tmp_surf, mem->surf,
              &nvinferlpr->transform_params, &batch->sync_obj);
  }

  nvtxDomainRangePop (nvinferlpr->nvtx_domain);

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer", err),
        (NULL));
    return FALSE;
  }
  

  LockGMutex locker (nvinferlpr->process_lock);
  /* Push the batch info structure in the processing queue and notify the output
   * thread that a new batch has been queued. */
  g_queue_push_tail (nvinferlpr->input_queue, batch);
  g_cond_broadcast (&nvinferlpr->process_cond);

  return TRUE;
}

std::vector<std::string> class_parent = {"car", "bus", "motorcycle", "truck"};


static cv::Mat
get_align(GstNvInferLpr *nvinferlpr, NvDsObjectMeta *object_meta, float landmarks[10]){ 
  NvBufSurface * surface = &nvinferlpr->tmp_surf;
  for (uint frameIndex = 0; frameIndex < surface->numFilled; frameIndex++) {
    gint frame_width = (gint)surface->surfaceList[frameIndex].width;
    gint frame_height = (gint)surface->surfaceList[frameIndex].height;

    void *src_data = NULL;
    CHECK_CUDA_STATUS (cudaMallocHost (&src_data,
                                       surface->surfaceList[frameIndex].dataSize), "Could not allocate cuda host buffer");


    if (src_data == NULL) {
      g_print("Error: failed to malloc src_data \n");
    }
    auto start = std::chrono::system_clock::now();
    cudaMemcpy((void *)src_data,
        (void *)surface->surfaceList[frameIndex].dataPtr,
        surface->surfaceList[frameIndex].dataSize,
        cudaMemcpyDeviceToHost);
    auto end = std::chrono::system_clock::now();
    size_t frame_step = surface->surfaceList[frameIndex].pitch;

    cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC1, src_data, frame_step);
    cv::Rect roi(object_meta->rect_params.left, 
                object_meta->rect_params.top, 
                object_meta->rect_params.width, 
                object_meta->rect_params.height);

    cv::Mat cropped_image = frame(roi);
    // cv::imwrite("/Vinh_Deepstream/deepstream_python_apps/apps/check_images/check_frame_crop.jpg", cropped_image);

    std::vector<cv::Point2f> src_landmarks = {
      cv::Point2f(landmarks[0] - roi.x, landmarks[1] - roi.y),
      cv::Point2f(landmarks[2] - roi.x, landmarks[3] - roi.y),
      cv::Point2f(landmarks[6] - roi.x, landmarks[7] - roi.y),
      cv::Point2f(landmarks[8] - roi.x, landmarks[9] - roi.y)
    };

    std::vector<cv::Point2f> dst_landmarks = {
      cv::Point2f(0.0, 0.0),
      cv::Point2f(cropped_image.cols, 0.0),
      cv::Point2f(0.0, cropped_image.rows),
      cv::Point2f(cropped_image.cols, cropped_image.rows)
    };

    cv::Mat M = cv::getPerspectiveTransform(src_landmarks, dst_landmarks);
    cv::Size outputSize(cropped_image.cols, cropped_image.rows);

    cv::Mat aligned;
    cv::warpPerspective(cropped_image, aligned, M, outputSize, cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
    
    // cv::imwrite("/Vinh_Deepstream/deepstream_python_apps/apps/check_images/check_frame_crop_aligned.jpg", aligned);

    float scaleWidth = 320.0 / aligned.cols;
    float scaleHeight = 320.0/ aligned.rows;
    float scale = std::min(scaleWidth, scaleHeight);

    int newWidth = static_cast<int>(aligned.cols * scale);
    int newHeight = static_cast<int>(aligned.rows * scale);

    cv::Mat resizedImage;
    cv::resize(aligned, resizedImage, cv::Size(newWidth, newHeight));

    cv::Mat cropped_image_resize_aligned(320, 320, aligned.type(), cv::Scalar(0));
    resizedImage.copyTo(cropped_image_resize_aligned(cv::Rect(0, 0, newWidth, newHeight)));
    
    // cv::imwrite("/Vinh_Deepstream/deepstream_python_apps/apps/check_images/check_frame_crop_resize.jpg", cropped_image_resize_aligned);
    cudaFreeHost(src_data);
    return cropped_image_resize_aligned;
  }
}


static void
perform_align(NvBufSurface * surface, NvDsObjectMeta *object_meta, float landmarks[10], cv::Mat cropped_image_resize_aligned){ 
  for (uint frameIndex = 0; frameIndex < surface->numFilled; frameIndex++) {
    gint frame_width = (gint)surface->surfaceList[frameIndex].width;
    gint frame_height = (gint)surface->surfaceList[frameIndex].height;

    void *src_data = NULL;
    CHECK_CUDA_STATUS (cudaMallocHost (&src_data,
                                       surface->surfaceList[frameIndex].dataSize), "Could not allocate cuda host buffer");

    if (src_data == NULL) {
      g_print("Error: failed to malloc src_data \n");
    }
    auto start = std::chrono::system_clock::now();
    cudaMemcpy((void *)src_data,
        (void *)surface->surfaceList[frameIndex].dataPtr,
        surface->surfaceList[frameIndex].dataSize,
        cudaMemcpyDeviceToHost);
    size_t frame_step = surface->surfaceList[frameIndex].pitch;
    auto CheckPoint_d2h = std::chrono::system_clock::now();
    

    cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC1, src_data, frame_step);
    cropped_image_resize_aligned.copyTo(frame);

    auto CheckPoint_alignment = std::chrono::system_clock::now();
    size_t sizeInBytes = surface->surfaceList[frameIndex].dataSize;
    cudaMemcpy((void *)surface->surfaceList[frameIndex].dataPtr,
        frame.ptr(0),
        sizeInBytes,
        cudaMemcpyHostToDevice);  
    auto CheckPoint_h2d = std::chrono::system_clock::now();
    
    cudaFreeHost(src_data);
  }
}

static gboolean
convert_batch_and_push_to_input_thread_alignment (GstNvInferLpr *nvinferlpr,
    GstNvInferLprBatch *batch, GstNvInferLprMemory *mem, NvDsFrameMeta *frame_meta, 
    NvDsObjectMeta *object_meta, NvOSD_RectParams * crop_rect_params, float landmarks[10])
{
  NvBufSurfTransform_Error err = NvBufSurfTransformError_Success;
  std::string nvtx_str;

  /* Set the transform session parameters for the conversions executed in this
   * thread. */
  err = NvBufSurfTransformSetSessionParams (&nvinferlpr->transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED, ("NvBufSurfTransformSetSessionParams failed with error %d", err), (NULL));
    return FALSE;
  }

  // cv::Mat cropped_image_resize_aligned = get_align(nvinferlpr, object_meta, landmarks);

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(nvinferlpr->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(nvinferlpr->nvtx_domain, &eventAttrib);

  if (batch->frames.size() > 0) {
    /* Batched tranformation. */
    // for some reason, if use async, there will be a latency and cause disorder.
    err = NvBufSurfTransform (&nvinferlpr->tmp_surf, mem->surf, &nvinferlpr->transform_params);
  }

  // perform_align(mem->surf, object_meta, landmarks, cropped_image_resize_aligned);
  

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer", err),
        (NULL));
    return FALSE;
  }


  LockGMutex locker (nvinferlpr->process_lock);
  /* Push the batch info structure in the processing queue and notify the output
   * thread that a new batch has been queued. */
  g_queue_push_tail (nvinferlpr->input_queue, batch);
  g_cond_broadcast (&nvinferlpr->process_cond);
  return TRUE;
}

/* Process entire frames in the batched buffer. */
static GstFlowReturn
gst_nvinferlpr_process_full_frame (GstNvInferLpr * nvinferlpr, GstBuffer * inbuf,
    NvBufSurface * in_surf)
{
  NvOSD_RectParams rect_params;
  NvDsBatchMeta *batch_meta = NULL;
  guint num_filled = 0;
  std::unique_ptr<GstNvInferLprBatch> batch = nullptr;
  GstBuffer *conv_gst_buf = nullptr;
  GstFlowReturn flow_ret;
  GstNvInferLprMemory *memory = nullptr;
  gdouble scale_ratio_x, scale_ratio_y;
  guint offset_left = 0, offset_top = 0;
  gboolean skip_batch;

  /* Process batch only when interval_counter is 0. */
  skip_batch = (nvinferlpr->interval_counter++ % (nvinferlpr->interval + 1) > 0);

  if (skip_batch) {
    return GST_FLOW_OK;
  }

  if (((in_surf->memType == NVBUF_MEM_DEFAULT || in_surf->memType == NVBUF_MEM_CUDA_DEVICE) &&
       ((int)in_surf->gpuId != (int)nvinferlpr->gpu_id)) ||
      (((int)in_surf->gpuId == (int)nvinferlpr->gpu_id) && (in_surf->memType == NVBUF_MEM_SYSTEM)))  {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Memory Compatibility Error:Input surface gpu-id doesnt match with configured gpu-id for element,"
         " please allocate input using unified memory, or use same gpu-ids OR,"
         " if same gpu-ids are used ensure appropriate Cuda memories are used"),
        ("surface-gpu-id=%d,%s-gpu-id=%d",in_surf->gpuId,GST_ELEMENT_NAME(nvinferlpr),
         nvinferlpr->gpu_id)); \
      return GST_FLOW_ERROR;
  }


  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  num_filled = batch_meta->num_frames_in_batch;

  /* Processing on full frames. Iterate through all the frames in the batched
   * input buffer. */
  for (guint i = 0; i < num_filled; i++) {
    guint idx;

    /* No existing GstNvInferLprBatch structure. Allocate a new structure,
     * acquire a buffer from our internal pool for conversions. */
    if (batch == nullptr) {
      batch.reset (new GstNvInferLprBatch);
      batch->push_buffer = FALSE;
      batch->inbuf = inbuf;
      batch->inbuf_batch_num = nvinferlpr->current_batch_num;

      flow_ret =
          gst_buffer_pool_acquire_buffer (nvinferlpr->pool, &conv_gst_buf,
          nullptr);
      if (flow_ret != GST_FLOW_OK) {
        return flow_ret;
      }
      memory = gst_nvinferlpr_buffer_get_memory (conv_gst_buf);
      if (!memory) {
        return GST_FLOW_ERROR;
      }
      batch->conv_buf = conv_gst_buf;
    }

    idx = batch->frames.size ();
    /* Scale the entire frame to network resolution. */
    rect_params.left = 0;
    rect_params.top = 0;
    rect_params.width = in_surf->surfaceList[i].width;
    rect_params.height = in_surf->surfaceList[i].height;

    /* Scale and convert the buffer. */
    if (get_converted_buffer (nvinferlpr, in_surf, in_surf->surfaceList + i,
            &rect_params, memory->surf, memory->surf->surfaceList + idx,
            scale_ratio_x, scale_ratio_y, offset_left, offset_top,
            memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
      GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED, ("Buffer conversion failed"),
          (NULL));
      return GST_FLOW_ERROR;
    }

    /* Adding a frame to the current batch. Set the frames members. */
    GstNvInferLprFrame frame;
    frame.converted_frame_ptr = memory->frame_memory_ptrs[idx];
    frame.scale_ratio_x = scale_ratio_x;
    frame.scale_ratio_y = scale_ratio_y;
    frame.offset_left = offset_left;
    frame.offset_top = offset_top;
    frame.obj_meta = nullptr;
    frame.frame_meta = nvds_get_nth_frame_meta (batch_meta->frame_meta_list, i);
    frame.frame_num = frame.frame_meta->frame_num;
    frame.batch_index = i;
    //frame.history = nullptr;
    frame.input_surf_params = in_surf->surfaceList + i;
    batch->frames.push_back (frame);

    /* Submit batch if the batch size has reached max_batch_size or
     * if this is the last frame in the input batched buffer. */
    if (batch->frames.size () == nvinferlpr->max_batch_size || i == num_filled - 1) {
      if (!convert_batch_and_push_to_input_thread (nvinferlpr, batch.release(), memory)) {
        return GST_FLOW_ERROR;
      }

      /* Batch submitted. Set batch to nullptr so that a new GstNvInferLprBatch
       * structure can be allocated if required. */
      conv_gst_buf = nullptr;
      nvinferlpr->tmp_surf.numFilled = 0;
    }
  }
  return GST_FLOW_OK;
}

/* The object history map should be trimmed periodically to keep the map size
 * in check. */
static void
cleanup_history_map (GstNvInferLpr * nvinferlpr, GstBuffer * inbuf)
{
  LockGMutex locker (nvinferlpr->process_lock);
  /* Find the history map for each source whose frames are present in the batch
   * and trim the map. */
  for (auto &source_iter : *(nvinferlpr->source_info)) {
    GstNvInferLprSourceInfo &source_info = source_iter.second;
    if (source_info.last_seen_frame_num - source_info.last_cleanup_frame_num < MAP_CLEANUP_INTERVAL)
      continue;
    source_info.last_cleanup_frame_num = source_info.last_seen_frame_num;

    /* Remove entries for objects which have not been seen for
     * CLEANUP_ACCESS_CRITERIA */
    auto iterator = source_info.object_history_map.begin ();
    while (iterator != source_info.object_history_map.end ()) {
      auto history = iterator->second;
      if (!history->under_inference &&
          source_info.last_seen_frame_num - history->last_accessed_frame_num >
          CLEANUP_ACCESS_CRITERIA) {
        iterator = source_info.object_history_map.erase (iterator);
      } else {
        ++iterator;
      }
    }
  }
}

/* Function to decide if object should be inferred on. */
static inline gboolean
should_infer_object (GstNvInferLpr * nvinferlpr, GstBuffer * inbuf,
    NvDsObjectMeta * obj_meta, gulong frame_num,
    GstNvInferLprObjectHistory * history)
{
  if (nvinferlpr->operate_on_gie_id > -1 &&
      obj_meta->unique_component_id != nvinferlpr->operate_on_gie_id)
    return FALSE;

  if (obj_meta->rect_params.width < nvinferlpr->min_input_object_width)
    return FALSE;

  if (obj_meta->rect_params.height < nvinferlpr->min_input_object_height)
    return FALSE;

  if (nvinferlpr->max_input_object_width > 0 &&
      obj_meta->rect_params.width > nvinferlpr->max_input_object_width)
    return FALSE;

  if (nvinferlpr->max_input_object_height > 0 &&
      obj_meta->rect_params.height > nvinferlpr->max_input_object_height)
    return FALSE;

  /* Infer on object if the operate_on_class_ids list is empty or if
   * the flag at index  class_id is TRUE. */
  if (!nvinferlpr->operate_on_class_ids->empty () &&
      ((int) nvinferlpr->operate_on_class_ids->size () <= obj_meta->class_id ||
          nvinferlpr->operate_on_class_ids->at (obj_meta->class_id) == FALSE)) {
    return FALSE;
  }

  if (history && IS_CLASSIFIER_INSTANCE (nvinferlpr)) {
    gboolean should_reinfer = FALSE;

    /* Do not reinfer if the object area has not grown by the reinference area
     * threshold and reinfer interval criteria is not met. */
    if ((history->last_inferred_coords.width *
          history->last_inferred_coords.height * (1 +
            REINFER_AREA_THRESHOLD)) <
        (obj_meta->rect_params.width * obj_meta->rect_params.height))
      should_reinfer = TRUE;

    if (frame_num - history->last_inferred_frame_num >
         nvinferlpr->secondary_reinfer_interval)
      should_reinfer = TRUE;

    return should_reinfer;
  }

  if (history && IS_DETECTOR_INSTANCE (nvinferlpr)) {
    gboolean should_reinfer = FALSE;

    if (frame_num - history->last_inferred_frame_num >
         nvinferlpr->secondary_reinfer_interval ||
         nvinferlpr->secondary_reinfer_interval == DEFAULT_REINFER_INTERVAL)
      should_reinfer = TRUE;

    return should_reinfer;
  }

  return TRUE;
}

/* Called when the last ref on the GstMiniObject inside
 * GstNvInferLprTensorOutputObject is removed. The batch output can be released
 * back to the NvDsInferContext. */
static void
gst_nvinferlpr_tensoroutput_free (GstMiniObject * obj)
{
  GstNvInferLprTensorOutputObject *output_obj =
      (GstNvInferLprTensorOutputObject *) obj;
  assert (output_obj->infer_context.get());
  output_obj->infer_context->releaseBatchOutput (output_obj->
      batch_output);
  output_obj->infer_context.reset ();
  delete output_obj;
}

/* Process on objects detected by upstream detectors.
 *
 * Secondary classifiers can work in asynchronous mode as well. In this mode,
 * tracked objects are cropped and queued for inferencing. The input buffer
 * is pushed downstream (from the input thread itself) without waiting for results.
 * When the infer results are available they are stored in the object history map
 * in the output loop. After the results are available the new/updated results
 * are attached (in the input thread) to the object whenever it is found in the
 * frame again. */
static GstFlowReturn
gst_nvinferlpr_process_objects (GstNvInferLpr * nvinferlpr, GstBuffer * inbuf, NvBufSurface * in_surf)
{
  std::unique_ptr<GstNvInferLprBatch> batch (nullptr);
  GstBuffer *conv_gst_buf = nullptr;
  GstNvInferLprMemory *memory = nullptr;
  GstFlowReturn flow_ret;
  gdouble scale_ratio_x, scale_ratio_y;
  guint offset_left = 0, offset_top = 0;
  gboolean warn_untracked_object = FALSE;
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }

  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    GstNvInferLprSourceInfo *source_info = nullptr;

    /* Find the source info instance. */
    auto iter = nvinferlpr->source_info->find (frame_meta->pad_index);
    if (iter == nvinferlpr->source_info->end ()) {
      GST_WARNING_OBJECT
          (nvinferlpr,
          "Source info not found for source %d. Maybe the GST_NVEVENT_PAD_ADDED"
          " event was never generated for the source.", frame_meta->pad_index);
      continue;
    } else {
      source_info = &iter->second;
    }
    source_info->last_seen_frame_num = frame_meta->frame_num;

    /* Iterate through all the objects. */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      NvDsObjectMeta *object_meta = (NvDsObjectMeta *) (l_obj->data);
      guint idx;
      std::shared_ptr<GstNvInferLprObjectHistory> obj_history;
      float landmarks[10] = {0.0};
      gulong frame_num = frame_meta->frame_num;

      /* Cannot infer on untracked objects in asynchronous mode. */
      if (nvinferlpr->classifier_async_mode && object_meta->object_id == UNTRACKED_OBJECT_ID) {
        if (!warn_untracked_object) {
          /* Warn periodically about untracked objects in the metadata. */
          if (nvinferlpr->untracked_object_warn_pts == GST_CLOCK_TIME_NONE ||
              (GST_BUFFER_PTS(inbuf) - nvinferlpr->untracked_object_warn_pts > UNTRACKED_OBJECT_WARN_INTERVAL)) {
            GST_WARNING_OBJECT (nvinferlpr, "Untracked objects in metadata. Cannot"
                " infer on untracked objects in asynchronous mode.");
            nvinferlpr->untracked_object_warn_pts = GST_BUFFER_PTS(inbuf);
          }
        }
        warn_untracked_object = TRUE;
        continue;
      }

      LockGMutex locker (nvinferlpr->process_lock);

      /* Find the object history if it exists only when tracking id is valid. */
      if (source_info != nullptr && object_meta->object_id != UNTRACKED_OBJECT_ID) {
        auto search = source_info->object_history_map.find (object_meta->object_id);
        if (search != source_info->object_history_map.end ()) {
          obj_history = search->second;
        }
      }

      bool needs_infer = should_infer_object (nvinferlpr, inbuf, object_meta, frame_num, obj_history.get());
      if (!needs_infer) {
        if (IS_CLASSIFIER_INSTANCE (nvinferlpr) && obj_history != nullptr) {
          /* Working in synchronous mode. Defer attachment of classifier metadata
           * in the object history to the output thread. */
          if (!nvinferlpr->classifier_async_mode) {
            /* No existing GstNvInferLprBatch structure. Allocate a new structure,
             * acquire a buffer from our internal pool for conversions. */
            if (batch == nullptr) {
              batch.reset (new GstNvInferLprBatch);
              batch->push_buffer = FALSE;
              batch->event_marker = FALSE;
              batch->inbuf = inbuf;
              batch->inbuf_batch_num = nvinferlpr->current_batch_num;
              locker.unlock ();
              flow_ret = gst_buffer_pool_acquire_buffer (nvinferlpr->pool, &conv_gst_buf, nullptr);
              locker.lock ();
              if (flow_ret != GST_FLOW_OK) {
                return flow_ret;
              }
              memory = gst_nvinferlpr_buffer_get_memory (conv_gst_buf);
              if (!memory) {
                return GST_FLOW_ERROR;
              }
              batch->conv_buf = conv_gst_buf;
            }
            obj_history->last_accessed_frame_num = frame_meta->frame_num;
            /* Let the output thread know to attach latest available classifier
             * metadata for this object. */
            batch->objs_pending_meta_attach.emplace_back(obj_history, object_meta);
            continue;
          }
        }
      }


      /* Asynchronous mode. If we have previous results for the tracked object,
       * attach the results. New results will be attached when inference on the
       * object is complete and the object is present in the frame after that. */
      if (obj_history && nvinferlpr->classifier_async_mode) {
        GstNvInferLprFrame frame;
        frame.obj_meta = object_meta;
        attach_metadata_classifier (nvinferlpr, nullptr, frame, obj_history->cached_info);
        obj_history->last_accessed_frame_num = frame_meta->frame_num;
      }

      if (!needs_infer) {
        continue;
      }

      // Custom alignment
      if (nvinferlpr->alignment_type == 2){
        NvDsMetaList * l_user_meta = NULL;
        NvDsUserMeta *user_meta = NULL;
        float *user_meta_data = NULL;
        for (l_user_meta = object_meta->obj_user_meta_list; l_user_meta != NULL; l_user_meta = l_user_meta->next) {
          user_meta = (NvDsUserMeta *) (l_user_meta->data);
          user_meta_data = (float *)user_meta->user_meta_data;
          if(user_meta->base_meta.meta_type == NVDS_USER_OBJECT_META_EXAMPLE){
            for (unsigned int i=0; i < 10; i++) {
              landmarks[i] = (float)user_meta_data[i];
            }
          }
        }
      }


      /* Object has a valid tracking id but does not have any history. Create
       * an entry in the map for the object. */
      if (source_info != nullptr && object_meta->object_id != UNTRACKED_OBJECT_ID &&
          obj_history == nullptr) {
        auto ret_iter =
            source_info->object_history_map.emplace (object_meta->object_id,
            std::make_shared<GstNvInferLprObjectHistory> ());
        obj_history = ret_iter.first->second;
      }

      /* Update the object history if it is found. */
      if (obj_history != nullptr) {
        obj_history->under_inference = TRUE;
        obj_history->last_inferred_frame_num = frame_num;
        obj_history->last_accessed_frame_num = frame_num;
        obj_history->last_inferred_coords = object_meta->rect_params;
      }

      locker.unlock ();

      /* No existing GstNvInferLprBatch structure. Allocate a new structure,
       * acquire a buffer from our internal pool for conversions. */
      if (batch == nullptr) {
        batch.reset (new GstNvInferLprBatch);
        batch->push_buffer = FALSE;
        batch->inbuf = (nvinferlpr->classifier_async_mode) ? nullptr : inbuf;
        batch->inbuf_batch_num = nvinferlpr->current_batch_num;

        flow_ret =
            gst_buffer_pool_acquire_buffer (nvinferlpr->pool, &conv_gst_buf,
            nullptr);
        if (flow_ret != GST_FLOW_OK) {
          return flow_ret;
        }
        memory = gst_nvinferlpr_buffer_get_memory (conv_gst_buf);
        if (!memory) {
          return GST_FLOW_ERROR;
        }
        batch->conv_buf = conv_gst_buf;
      }
      idx = batch->frames.size ();

      /* Crop, scale and convert the buffer. */
      if (get_converted_buffer (nvinferlpr, in_surf,
              in_surf->surfaceList + frame_meta->batch_id,
              &object_meta->rect_params, memory->surf,
              memory->surf->surfaceList + idx, scale_ratio_x, scale_ratio_y,
              offset_left, offset_top,
              memory->frame_memory_ptrs[idx]) != GST_FLOW_OK) {
        GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED, ("Buffer conversion failed"), (NULL));
        return GST_FLOW_ERROR;
      }

      /* Adding a frame to the current batch. Set the frames members. */
      GstNvInferLprFrame frame;
      frame.converted_frame_ptr = memory->frame_memory_ptrs[idx];
      frame.scale_ratio_x = scale_ratio_x;
      frame.scale_ratio_y = scale_ratio_y;
      frame.offset_left = offset_left;
      frame.offset_top = offset_top;
      frame.obj_meta = (nvinferlpr->classifier_async_mode) ? nullptr : object_meta;
      frame.frame_meta = frame_meta;
      frame.frame_num = frame_num;
      frame.batch_index = frame_meta->batch_id;
      frame.history = obj_history;
      frame.input_surf_params =
          (nvinferlpr->classifier_async_mode) ? nullptr : (in_surf->surfaceList +
          frame_meta->batch_id);
      batch->frames.push_back (frame);

      /* Submit batch if the batch size has reached max_batch_size. */

      if (batch->frames.size () == nvinferlpr->max_batch_size && nvinferlpr->alignment_type == 2) {
        if (!convert_batch_and_push_to_input_thread_alignment (nvinferlpr, batch.get(), memory, frame_meta, object_meta, &object_meta->rect_params, landmarks)) {
          return GST_FLOW_ERROR;
        }
        batch.release ();
        conv_gst_buf = nullptr;
        nvinferlpr->tmp_surf.numFilled = 0;
      }else if (batch->frames.size () == nvinferlpr->max_batch_size) {
        if (!convert_batch_and_push_to_input_thread (nvinferlpr, batch.get(), memory)) {
          return GST_FLOW_ERROR;
        }
        /* Batch submitted. Set batch to nullptr so that a new GstNvInferLprBatch
        * structure can be allocated if required. */
        batch.release ();
        conv_gst_buf = nullptr;
        nvinferlpr->tmp_surf.numFilled = 0;
      }
    }
  }

  /* Submit a non-full batch. */
  if (batch) {
    /* No frames to infer in this batch. It might contain objects that
     * have been deferred for classification metadata attachment. Return
     * intermediate memory to pool. */
    if (batch->frames.size() == 0)
      gst_buffer_unref (batch->conv_buf);

    if (!convert_batch_and_push_to_input_thread (nvinferlpr, batch.get(), memory)) {
      return GST_FLOW_ERROR;
    }
    conv_gst_buf = nullptr;
    batch.release ();
    nvinferlpr->tmp_surf.numFilled = 0;
  }

  if (nvinferlpr->current_batch_num -
      nvinferlpr->last_map_cleanup_frame_num > MAP_CLEANUP_INTERVAL) {
    cleanup_history_map (nvinferlpr, inbuf);
    nvinferlpr->last_map_cleanup_frame_num = nvinferlpr->current_batch_num;
  }

  return GST_FLOW_OK;
}


static GstFlowReturn
gst_nvinferlpr_process_tensor_input (GstNvInferLpr * nvinferlpr, GstBuffer * inbuf, NvBufSurface * in_surf)
{
  typedef struct{
    guint batch_size = 0;
    std::vector<GstNvInferLprFrame> frames;

    std::unique_ptr<GstNvInferLprBatch> batch = nullptr;
    std::vector<NvDsInferLayerInfo> tensors;
  } TensorInputBatch;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  std::unordered_map<guint, TensorInputBatch> tensormeta_map;

  for (NvDsMetaList * l_user = batch_meta->batch_user_meta_list; l_user != NULL;
      l_user = l_user->next) {

    NvDsUserMeta *user_meta = (NvDsUserMeta *) (l_user->data);
    if (user_meta->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
      continue;
    GstNvDsPreProcessBatchMeta *preproc_meta = (GstNvDsPreProcessBatchMeta *) user_meta->user_meta_data;

    const auto & uids = preproc_meta->target_unique_ids;
    if (std::find (uids.begin (), uids.end (), nvinferlpr->unique_id) == uids.end ())
      continue;

    if (!preproc_meta->tensor_meta)
      continue;

    guint meta_id = preproc_meta->tensor_meta->meta_id;

    if (preproc_meta->tensor_meta->gpu_id != nvinferlpr->gpu_id) {
      GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
          ("nvinferlpr configured with gpu-id %d but received input tensor allocated on gpu %d",
              nvinferlpr->gpu_id, preproc_meta->tensor_meta->gpu_id), (NULL));
      return GST_FLOW_ERROR;
    }

    bool layerExists = false;

    for (auto &layer : *nvinferlpr->layers_info) {
        if (layer.isInput && preproc_meta->tensor_meta->tensor_name == layer.layerName) {
          layerExists = true;
          break;
        }
    }

    if (!layerExists) {
        GST_ELEMENT_WARNING (nvinferlpr, STREAM, FAILED,
            ("nvinferlpr could not find input layer with name = %s\n",
              preproc_meta->tensor_meta->tensor_name.c_str()), (NULL));
      continue;
    }

    if (tensormeta_map[meta_id].batch == nullptr) {
      for (auto &roi : preproc_meta->roi_vector) {
        GstNvInferLprFrame frame;
        frame.batch_index = roi.frame_meta->batch_id;
        frame.scale_ratio_x = roi.scale_ratio_x;
        frame.scale_ratio_y = roi.scale_ratio_y;
        frame.offset_left = roi.offset_left;
        frame.offset_top = roi.offset_top;
        frame.roi_left = roi.roi.left;
        frame.roi_top = roi.roi.top;
        frame.roi_meta = &roi;
        frame.frame_meta = roi.frame_meta;
        frame.obj_meta = roi.object_meta;
        frame.frame_num = roi.frame_meta->frame_num;
        frame.input_surf_params =
            &in_surf->surfaceList[roi.frame_meta->batch_id];
        tensormeta_map[meta_id].frames.push_back (frame);
      }
    }

    NvDsInferLayerInfo tensor;
    tensor.isInput = TRUE;
    tensor.bindingIndex = -1;
    tensor.buffer = preproc_meta->tensor_meta->raw_tensor_buffer;
    tensor.layerName = preproc_meta->tensor_meta->tensor_name.c_str();
    tensor.inferDims.numDims = preproc_meta->tensor_meta->tensor_shape.size();
    tensor.inferDims.numElements = 1;
    for (unsigned int i = 0; i < tensor.inferDims.numDims; i++) {
      tensor.inferDims.numElements *=
          preproc_meta->tensor_meta->tensor_shape[i];
      tensor.inferDims.d[i] = preproc_meta->tensor_meta->tensor_shape[i];
    }

    if (tensormeta_map[meta_id].batch_size == 0) {
      tensormeta_map[meta_id].batch_size = tensor.inferDims.d[0];
    }
    else if (tensormeta_map[meta_id].batch_size != tensor.inferDims.d[0]) {
      GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
          ("Mismatch in input tensor batch sizes %d vs %d", tensormeta_map[meta_id].batch_size,
              tensor.inferDims.d[0]), (nullptr));
      return GST_FLOW_ERROR;
    }

    switch (preproc_meta->tensor_meta->data_type) {
      case NvDsDataType_FP32:
        tensor.dataType = FLOAT;
        break;
      case NvDsDataType_FP16:
        tensor.dataType = HALF;
        break;
      case NvDsDataType_UINT8:
      case NvDsDataType_INT8:
        tensor.dataType = INT8;
        break;
      case NvDsDataType_UINT32:
      case NvDsDataType_INT32:
        tensor.dataType = INT32;
        break;
      default:
        return GST_FLOW_ERROR;
    }

    tensormeta_map[meta_id].tensors.push_back (tensor);
  }

  for (auto & it : tensormeta_map) {
    auto & tensor_input_batch = it.second;
    for (size_t i = 0; i < tensor_input_batch.frames.size (); i++) {
      if (!tensor_input_batch.batch) {
        tensor_input_batch.batch.reset (new GstNvInferLprBatch);
        tensor_input_batch.batch->push_buffer = FALSE;
        tensor_input_batch.batch->inbuf = inbuf;
        tensor_input_batch.batch->inbuf_batch_num = nvinferlpr->current_batch_num;
      }
      tensor_input_batch.batch->frames.push_back (tensor_input_batch.frames[i]);

      if (i == tensor_input_batch.frames.size () - 1
          || tensor_input_batch.batch->frames.size () == nvinferlpr->max_batch_size) {
        NvDsInferContextBatchPreprocessedInput input_batch;
        input_batch.tensors = tensor_input_batch.tensors.data ();
        input_batch.numInputTensors = tensor_input_batch.tensors.size ();
        input_batch.returnFuncData = nullptr;
        input_batch.returnInputFunc = nullptr;

        for (auto &tensor : tensor_input_batch.tensors)
          tensor.dims.d[0] = tensor_input_batch.batch->frames.size();

        NvDsInferStatus status =
            DS_NVINFERLPR_IMPL (nvinferlpr)->m_InferCtx->
            queueInputBatchPreprocessed (input_batch);
        

        if (status != NVDSINFER_SUCCESS) {
          GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
              ("Failed to queue input batch for inferencing"), (nullptr));
          return GST_FLOW_ERROR;
        }

        g_mutex_lock (&nvinferlpr->process_lock);
        g_queue_push_tail (nvinferlpr->process_queue, tensor_input_batch.batch.release ());
        g_cond_broadcast (&nvinferlpr->process_cond);
        g_mutex_unlock (&nvinferlpr->process_lock);

        for (auto &tensor : tensor_input_batch.tensors) {
          gsize offset = 0;
          switch (tensor.dataType) {
            case FLOAT:
              offset = 4;
              break;
            case HALF:
              offset = 2;
              break;
            case INT8:
              offset = 1;
              break;
            case INT32:
              offset = 4;
              break;
            default:
              return GST_FLOW_ERROR;
          }
          for (guint j = 0; j < tensor.inferDims.numDims; j++)
            offset *= tensor.inferDims.d[j];

          tensor.buffer = (char *) tensor.buffer + offset;
        }
      }
    }
  }

  return GST_FLOW_OK;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */

static GstFlowReturn
gst_nvinferlpr_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (btrans);
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstNvInferLprBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;

  /* Check for model updates and replace the model if a new model is loaded. */
  if (impl->ensureReplaceNextContext () != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
              ("Ensure next context failed."),
              ("streaming stopped"));
      return GST_FLOW_ERROR;
  }

  nvinferlpr->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(nvinferlpr->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(nvinferlpr->nvtx_domain, &eventAttrib);

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinferlpr));

  if (nvinferlpr->input_tensor_from_meta) {
   flow_ret = gst_nvinferlpr_process_tensor_input (nvinferlpr, inbuf, in_surf);
  } else if (nvinferlpr->process_full_frame) {
   flow_ret = gst_nvinferlpr_process_full_frame (nvinferlpr, inbuf, in_surf);
  } else {
    flow_ret = gst_nvinferlpr_process_objects (nvinferlpr, inbuf, in_surf);
  }

  /* Unmap the input buffer contents. */
  if (in_map_info.data)
    gst_buffer_unmap (inbuf, &in_map_info);


  if (flow_ret == GST_FLOW_ERROR)
    return GST_FLOW_ERROR;

  if (nvinferlpr->classifier_async_mode) {
    /* Asynchronous mode. Push the buffer immediately instead of waiting for
     * the results. */
    nvtxDomainRangeEnd(nvinferlpr->nvtx_domain, buf_process_range);

    nvds_set_output_system_timestamp(inbuf, GST_ELEMENT_NAME(nvinferlpr));

    GstFlowReturn flow_ret = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvinferlpr),inbuf);
    if (nvinferlpr->last_flow_ret != flow_ret) {
      switch (flow_ret) {
        /* Signal the application for pad push errors by posting a error message
         * on the pipeline bus. */
        case GST_FLOW_ERROR:
        case GST_FLOW_NOT_LINKED:
        case GST_FLOW_NOT_NEGOTIATED:
          GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
              ("Internal data stream error."),
              ("streaming stopped, reason %s (%d)", gst_flow_get_name (flow_ret),
                  flow_ret));
          break;
        default:
          break;
      }
    }
    nvinferlpr->last_flow_ret = flow_ret;

    return flow_ret;
  } else {
    /* Queue a push buffer batch. This batch is not inferred. This batch is to
     * signal the input-queue and output thread that there are no more batches
     * belonging to this input buffer and this GstBuffer can be pushed to
     * downstream element once all the previous processing is done. */
    buf_push_batch = new GstNvInferLprBatch;
    buf_push_batch->inbuf = inbuf;
    buf_push_batch->push_buffer = TRUE;
    buf_push_batch->nvtx_complete_buf_range = buf_process_range;

    g_mutex_lock (&nvinferlpr->process_lock);
    if (nvinferlpr->input_queue_thread)
      g_queue_push_tail (nvinferlpr->input_queue, buf_push_batch);
    else
      g_queue_push_tail (nvinferlpr->process_queue, buf_push_batch);
    g_cond_broadcast (&nvinferlpr->process_cond);
    g_mutex_unlock (&nvinferlpr->process_lock);
  }

  return GST_FLOW_OK;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_nvinferlpr_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (btrans);
  return nvinferlpr->last_flow_ret;
}

/** Writes contents of the bound input and output layers to files. */
static void
gst_nvinferlpr_output_generated_file_write (GstBuffer * buf,
    NvDsInferNetworkInfo * network_info, NvDsInferLayerInfo * layers_info,
    guint num_layers, guint batch_size, GstNvInferLpr * nvinferlpr)
{
  guint i;
  gchar file_name[256];
  gchar *iter;

  for (i = 0; i < num_layers; i++) {
    NvDsInferLayerInfo *info = &layers_info[i];
    gsize layer_size = info->inferDims.numElements * batch_size;
    FILE *file;

    g_snprintf (file_name, 256,
        "gstnvdsinfer_uid-%02d_layer-%s_batch-%010lu_batchsize-%02d.bin",
        nvinferlpr->unique_id, info->layerName,
        nvinferlpr->file_write_batch_num, batch_size);
    file_name[255] = '\0';

    /* Replace '/' in a layer name with '_' */
    for (iter = file_name; *iter != '\0'; iter++) {
      if (*iter == '/')
        *iter = '_';
    }

    file = fopen (file_name, "w");
    if (!file) {
      g_printerr ("Could not open file '%s' for writing:%s\n",
          file_name, strerror (errno));
      continue;
    }
    fwrite (info->buffer, get_element_size (info->dataType), layer_size, file);
    fclose (file);
  }
  nvinferlpr->file_write_batch_num++;
}

/**
 * Output loop used to pop output from inference, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_nvinferlpr_output_loop (gpointer data)
{
  GstNvInferLpr *nvinferlpr = GST_NVINFERLPR (data);
  DsNvInferImpl *impl = DS_NVINFERLPR_IMPL (nvinferlpr);
  NvDsInferContextInitParams *init_params = impl->m_InitParams.get ();
  NvDsInferStatus status = NVDSINFER_SUCCESS;
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = NVTX_DEEPBLUE_COLOR;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;
  cudaError_t cudaReturn;

  cudaReturn = cudaSetDevice (nvinferlpr->gpu_id);
  if (cudaReturn != cudaSuccess) {
    GST_ELEMENT_ERROR (nvinferlpr, RESOURCE, FAILED,
        ("Failed to set cuda device %d", nvinferlpr->gpu_id),
        ("cudaSetDevice failed with error %s", cudaGetErrorName (cudaReturn)));
  }

  nvtx_str = "gst-nvinferlpr_output-loop_uid=" + std::to_string(nvinferlpr->unique_id);

  LockGMutex locker (nvinferlpr->process_lock);
  /* Run till signalled to stop. */
  while (!nvinferlpr->stop) {
    std::unique_ptr<GstNvInferLprBatch> batch = nullptr;
    NvDsInferContextBatchOutput *batch_output = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (nvinferlpr->process_queue)) {
      locker.wait (nvinferlpr->process_cond);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstNvInferLprBatch *) g_queue_pop_head (nvinferlpr->process_queue));
    g_cond_broadcast (&nvinferlpr->process_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    /* Attach latest available classification metadata for objects that have
     * not been inferred on in the current frame. */
    if (batch->frames.size() == 0 && !batch->push_buffer) {
      for (auto &hist : batch->objs_pending_meta_attach) {
        GstNvInferLprFrame frame;
        frame.obj_meta = hist.second;
        std::weak_ptr<GstNvInferLprObjectHistory> obj_history = hist.first;
        attach_metadata_classifier (nvinferlpr, nullptr, frame,
            obj_history.lock()->cached_info);
      }
      continue;
    }

    locker.unlock ();

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(nvinferlpr->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp(batch->inbuf, GST_ELEMENT_NAME(nvinferlpr));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (nvinferlpr), batch->inbuf);
      if (nvinferlpr->last_flow_ret != flow_ret) {
        switch (flow_ret) {
          /* Signal the application for pad push errors by posting a error message
           * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)", gst_flow_get_name (flow_ret),
                    flow_ret));
            break;
          default:
          break;
        }
      }
      nvinferlpr->last_flow_ret = flow_ret;
      locker.lock ();
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(nvinferlpr->nvtx_domain, &eventAttrib);

    NvDsInferContextPtr nvdsinfer_ctx = impl->m_InferCtx;

    /* Create and initialize the object for managing the usage of batch_output. */
    auto tensor_deleter = [] (GstNvInferLprTensorOutputObject *o) {
      if (o)
        gst_mini_object_unref (GST_MINI_OBJECT (o));
    };
    std::unique_ptr<GstNvInferLprTensorOutputObject, decltype(tensor_deleter)>
        tensor_out_object (new GstNvInferLprTensorOutputObject, tensor_deleter);
    gst_mini_object_init (GST_MINI_OBJECT (tensor_out_object.get()), 0, G_TYPE_POINTER, NULL,
        NULL, gst_nvinferlpr_tensoroutput_free);
    tensor_out_object->infer_context = nvdsinfer_ctx;

    batch_output = &tensor_out_object->batch_output;

    /* Dequeue inferencing output from NvDsInferContext */
    status = nvdsinfer_ctx->dequeueOutputBatch (*batch_output);

    locker.lock ();

    if (status != NVDSINFER_SUCCESS) {
      GST_ELEMENT_ERROR (nvinferlpr, STREAM, FAILED,
          ("Failed to dequeue output from inferencing. NvDsInferContext error: %s",
              NvDsInferStatus2Str (status)), (nullptr));
      continue;
    }

    /* Get the host buffer pointers from the latest dequeued output. */
    for (auto & layer:*nvinferlpr->layers_info) {
      layer.buffer = batch_output->hostBuffers[layer.bindingIndex];
    }

    /* Write layer contents to file if enabled. */
    if (nvinferlpr->write_raw_buffers_to_file) {
      gst_nvinferlpr_output_generated_file_write (batch->inbuf,
          &nvinferlpr->network_info,
          nvinferlpr->layers_info->data (),
          nvinferlpr->layers_info->size (), batch->frames.size (), nvinferlpr);
    }

    /* Call the output generated callback if specified. */
    if (nvinferlpr->output_generated_callback) {
      nvinferlpr->output_generated_callback (batch->inbuf,
          &nvinferlpr->network_info,
          nvinferlpr->layers_info->data (),
          nvinferlpr->layers_info->size (),
          batch->frames.size (), nvinferlpr->output_generated_userdata);
    }


    /* For each frame attach metadata output. */
    for (guint i = 0; i < batch->frames.size (); i++) {
      GstNvInferLprFrame &frame = batch->frames[i];
      NvDsInferFrameOutput &frame_output = batch_output->frames[i];
      auto obj_history = frame.history.lock ();

      /* If we have an object's history and the buffer PTS is same as last
       * inferred PTS mark the object as not being inferred. This check could be
       * useful if object is inferred multiple times before completion of an
       * existing inference. */
      if (obj_history != nullptr) {
        if (obj_history->last_inferred_frame_num == frame.frame_num)
          obj_history->under_inference = FALSE;
      }

      if (IS_DETECTOR_INSTANCE (nvinferlpr) || IS_INSTANCE_SEGMENTATION_INSTANCE (nvinferlpr)) {  
         attach_metadata_detector (nvinferlpr, GST_MINI_OBJECT (tensor_out_object.get()),
              frame, frame_output.detectionOutput, init_params->segmentationThreshold);
      } else if (IS_CLASSIFIER_INSTANCE (nvinferlpr)) {
        NvDsInferClassificationOutput &classification_output = frame_output.classificationOutput;
        GstNvInferLprObjectInfo new_info;
        new_info.attributes.assign(classification_output.attributes, classification_output.attributes + classification_output.numAttributes);
        new_info.label.assign(classification_output.label);
        for (guint i = 0; i < classification_output.numAttributes; i++) {
          classification_output.attributes[i].attributeLabel = nullptr;
        }

        /* Object history is available merge the old and new classification
         * results. */
        if (obj_history != nullptr) {
          merge_classification_output (*obj_history, new_info);
        }

        /* Use the merged classification results if available otherwise use
         * the new results. */
        auto &  info = (obj_history) ? obj_history->cached_info : new_info;

        /* Attach metadata only if not operating in async mode. In async mode,
         * the GstBuffer and the associated metadata are not valid here, since
         * the buffer is already pushed downstream. The metadata will be updated
         * in the input thread. */
        if (nvinferlpr->classifier_async_mode == FALSE) {
          attach_metadata_classifier (nvinferlpr, GST_MINI_OBJECT (tensor_out_object.get()), frame, info);
        }
      } else if (IS_SEGMENTATION_INSTANCE (nvinferlpr)) {
        attach_metadata_segmentation (nvinferlpr, GST_MINI_OBJECT (tensor_out_object.get()), frame, frame_output.segmentationOutput);
      }
    }

    /* Attach latest available classification metadata for objects that have
     * not been inferred on in the current frame. */
    for (auto &hist : batch->objs_pending_meta_attach) {
      GstNvInferLprFrame frame;
      frame.obj_meta = hist.second;
      std::weak_ptr<GstNvInferLprObjectHistory> obj_history = hist.first;
      attach_metadata_classifier (nvinferlpr, nullptr, frame, obj_history.lock()->cached_info);
    }

    if (nvinferlpr->output_tensor_meta && !nvinferlpr->classifier_async_mode) {
      /* Attach the tensor output as meta. */
      attach_tensor_output_meta (nvinferlpr, GST_MINI_OBJECT(tensor_out_object.get()), batch.get(), batch_output);
    }
    nvtxDomainRangePop (nvinferlpr->nvtx_domain);

  }
  return nullptr;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvinferlpr_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_nvinferlpr_debug, "nvinferlpr", 0, "nvinferlpr plugin");
  gst_debug_category_set_threshold (gst_nvinferlpr_debug, GST_LEVEL_INFO);

  return gst_element_register (plugin, "nvinferlpr", GST_RANK_PRIMARY, GST_TYPE_NVINFERLPR);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, nvdsgst_inferlpr,
    DESCRIPTION, nvinferlpr_plugin_init, "6.3", LICENSE, BINARY_PACKAGE, URL)

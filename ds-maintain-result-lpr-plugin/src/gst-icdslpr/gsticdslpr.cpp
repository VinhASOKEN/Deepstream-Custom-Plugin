#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <cstring>
#include <map>
#include <cmath>
#include "gsticdslpr.h"
#include "icdslpr_property_parser.h"
#include <sys/time.h>
#include <regex>
GST_DEBUG_CATEGORY_STATIC (gst_icdslpr_debug);
#define GST_CAT_DEFAULT gst_icdslpr_debug

static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_UNIQUE_ID,
  PROP_ENABLE,
  PROP_CONFIG_FILE
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 17
#define DEFAULT_WIDTH 1920
#define DEFAULT_HEIGHT 1080


/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_icdslpr_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate gst_icdslpr_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_icdslpr_parent_class parent_class
G_DEFINE_TYPE (GstIcDsLpr, gst_icdslpr, GST_TYPE_BASE_TRANSFORM);

static void gst_icdslpr_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_icdslpr_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
static gboolean gst_icdslpr_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_icdslpr_start (GstBaseTransform * btrans);
static gboolean gst_icdslpr_stop (GstBaseTransform * btrans);
static void gst_icdslpr_finalize (GObject * object);
static GstFlowReturn gst_icdslpr_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf);


/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_icdslpr_class_init (GstIcDsLprClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_icdslpr_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_icdslpr_get_property);
  gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_icdslpr_finalize);
  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR(gst_icdslpr_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_icdslpr_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_icdslpr_stop);
  gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_icdslpr_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
        "Unique ID for the element. Can be used to identify output of the"
        " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
        (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_ENABLE,
      g_param_spec_boolean ("enable", "Enable",
        "Enable DsLpr plugin, or set in passthrough mode",
        TRUE, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE,
      g_param_spec_string ("config-file", "DsLpr Config File",
        "DsLpr Config File", NULL, (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class, gst_static_pad_template_get(&gst_icdslpr_src_template));
  gst_element_class_add_pad_template (gstelement_class, gst_static_pad_template_get(&gst_icdslpr_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsLpr plugin",
      "DsLpr Plugin",
      "Process lpr algorithm on objects ",
      "NVIDIA Corporation. Post on Deepstream forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

std::vector <std::string> template_check_list;
gint template_mode = 0;
gint draw_mode = 0;
gint update_mode = 0;
gint max_frame_unseen_to_remove = 0;
gint max_pre_result_per_object = 0;

static void get_template_config (GstIcDsLpr * icdslpr)
{
  ConfigInfo &config_infor = *(icdslpr->config_infor);
  template_check_list = config_infor.template_check_list;
  template_mode = config_infor.template_mode;
  draw_mode = config_infor.draw_mode;
  update_mode = config_infor.update_mode;
  max_frame_unseen_to_remove = config_infor.max_frame_unseen_to_remove;
  max_pre_result_per_object = config_infor.max_pre_result_per_object;
}

static void
gst_icdslpr_init (GstIcDsLpr * icdslpr)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (icdslpr);
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  icdslpr->unique_id = DEFAULT_UNIQUE_ID;
  icdslpr->configuration_width = DEFAULT_WIDTH;
  icdslpr->configuration_height = DEFAULT_HEIGHT;
  icdslpr->config_file_path = NULL;
  icdslpr->config_file_parse_successful = FALSE;
  icdslpr->enable = TRUE;
  icdslpr->config_infor = new ConfigInfo;
  g_mutex_init (&icdslpr->lpr_mutex);

  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

static void reset_config_infor(ConfigInfo* config_infor) {
    if (config_infor) {
        config_infor->template_check_list.clear();
        config_infor->template_mode = INT_MIN;
        config_infor->draw_mode = INT_MIN;
        config_infor->update_mode = INT_MIN;
        config_infor->max_frame_unseen_to_remove = INT_MIN;
        config_infor->max_pre_result_per_object = INT_MIN;
        config_infor->config_width = INT_MIN;
        config_infor->config_height = INT_MIN;
    }
}

/* Free resources allocated during init. */
static void
gst_icdslpr_finalize (GObject * object)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (object);
  icdslpr->config_file_path = NULL;
  icdslpr->config_file_parse_successful = FALSE;
  reset_config_infor(icdslpr->config_infor);
  g_mutex_clear (&icdslpr->lpr_mutex);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_icdslpr_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      icdslpr->unique_id = g_value_get_uint (value);
      break;
    case PROP_ENABLE:
      icdslpr->enable = g_value_get_boolean (value);
      break;
    case PROP_CONFIG_FILE:
    {
      g_mutex_lock (&icdslpr->lpr_mutex);
      g_free (icdslpr->config_file_path);
      icdslpr->config_file_path = g_value_dup_string (value);
      icdslpr->config_file_parse_successful = icdslpr_parse_config_file (icdslpr, icdslpr->config_file_path);

      // Get config for custom plugin
      get_template_config(icdslpr);

      g_mutex_unlock (&icdslpr->lpr_mutex);
    }
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
gst_icdslpr_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, icdslpr->unique_id);
      break;
    case PROP_ENABLE:
      g_value_set_boolean (value, icdslpr->enable);
      break;
    case PROP_CONFIG_FILE:
      g_value_set_string (value, icdslpr->config_file_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_icdslpr_start (GstBaseTransform * btrans)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (btrans);
  icdslpr->batch_size = 1;

  if (!icdslpr->config_file_path || strlen (icdslpr->config_file_path) == 0) {
    GST_ELEMENT_ERROR (icdslpr, LIBRARY, SETTINGS, ("Configuration file not provided"), (nullptr));
    return FALSE;
  }
  if (icdslpr->config_file_parse_successful == FALSE) {
    GST_ELEMENT_ERROR (icdslpr, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", icdslpr->config_file_path));
    return FALSE;
  }
  return TRUE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_icdslpr_stop (GstBaseTransform * btrans)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (btrans);
  GST_DEBUG_OBJECT(icdslpr, "ctx lib released \n");
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_icdslpr_set_caps (GstBaseTransform * btrans, GstCaps * incaps, GstCaps * outcaps)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (btrans);
  gint batch_size = 1;
  GstStructure *structure = gst_caps_get_structure (incaps, 0);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&icdslpr->video_info, incaps);

  if (structure && gst_structure_get_int (structure, "batch-size", &batch_size)) {
    if (batch_size) {
      icdslpr->batch_size = batch_size;
      GST_DEBUG_OBJECT(icdslpr, "Setting batch-size %d from set caps\n", icdslpr->batch_size);
    }
  }
  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */

std::unordered_map<int, std::tuple<int, std::vector<std::string>>> global_store_id;
gint count_frames = 0;

static void control() {
    for (auto it = global_store_id.begin(); it != global_store_id.end(); ) {
        if (count_frames - std::get<0>(it->second) >= max_frame_unseen_to_remove) {
            it = global_store_id.erase(it);
        } else {
            ++it;
        }
    }
    for (auto& pair : global_store_id) {
        std::vector<std::string>& vec = std::get<1>(pair.second);
        if (vec.size() > max_pre_result_per_object) {
            vec.erase(vec.begin(), vec.begin() + (vec.size() - max_pre_result_per_object));
        }
    }
}

static std::string transform_license_plate(const char* input, std::string& stdString) {
    std::istringstream iss(input);
    std::string part;
    std::string lastPart;
    while (iss >> part) {
      lastPart = part;
    }
    return "license_plate " + lastPart + " " + stdString;
}

static gboolean check_match_template(std::string label_string, std::vector<std::string> template_check_list) {
    for (const auto& template_str : template_check_list) {
        std::regex format(template_str);
        if (std::regex_match(label_string, format)) {
            return true;
        }
    }
    return false;
}

static GstFlowReturn
gst_icdslpr_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstIcDsLpr *icdslpr = GST_ICDSLPR (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_ERROR;
  ConfigInfo &config_infor = *(icdslpr->config_infor);
  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;
  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (icdslpr));

  icdslpr->batch_num++;
  if (FALSE == icdslpr->config_file_parse_successful) {
    GST_ELEMENT_ERROR (icdslpr, LIBRARY, SETTINGS,
        ("Configuration file parsing failed"),
        ("Config file path: %s", icdslpr->config_file_path));
    return flow_ret;
  }

  if (FALSE == icdslpr->enable) {
    GST_DEBUG_OBJECT (icdslpr, "IcDsLpr in passthrough mode");
    flow_ret = GST_FLOW_OK;
    return flow_ret;
  }

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    return flow_ret;
  }

  surface = (NvBufSurface *) in_map_info.data;
  GST_DEBUG_OBJECT (icdslpr,
      "Processing Batch %" G_GUINT64_FORMAT " Surface %p\n",
      icdslpr->batch_num, surface);

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (nullptr == batch_meta) {
    GST_ELEMENT_ERROR (icdslpr, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return flow_ret;
  }
  // Using object crops as input to the algorithm. The objects are detected by
  // the primary detector
  NvDsMetaList *l_obj = nullptr;
  NvDsMetaList *l_class = nullptr;
  NvDsMetaList *l_label = nullptr;
  NvDsObjectMeta *obj_meta = nullptr;
  NvDsClassifierMeta *class_meta = nullptr;
  NvDsLabelInfo *label_info = nullptr;

  g_mutex_lock (&icdslpr->lpr_mutex);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);

    if(template_mode == 1) {
      count_frames += 1;

      // Update And Draw
      if (draw_mode == 1 && update_mode == 1){
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);

          if (obj_meta->unique_component_id == 2){
            if (global_store_id.find(obj_meta->object_id) != global_store_id.end()){
              std::get<0>(global_store_id[obj_meta->object_id]) += 1;
            }
            for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next){
              class_meta = (NvDsClassifierMeta *) (l_class->data);
              for (l_label = class_meta->label_info_list; l_label != NULL; l_label = l_label->next){
                label_info = (NvDsLabelInfo *) (l_label->data); 
                if (class_meta->unique_component_id == 3){
                  std::string stdString_check = std::string(label_info->result_label);
                  if (check_match_template(stdString_check, template_check_list) == true){
                    std::get<1>(global_store_id[obj_meta->object_id]).push_back(stdString_check);
                    std::get<0>(global_store_id[obj_meta->object_id]) = count_frames;
                    char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString_check).c_str());
                    obj_meta->text_params.display_text = display_text;
                  }else{
                    if (global_store_id.find(obj_meta->object_id) != global_store_id.end()) {
                      std::string stdString = std::get<1>(global_store_id[obj_meta->object_id]).back();
                      strncpy(label_info->result_label, stdString.c_str(), sizeof(label_info->result_label) - 1);
                      char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString).c_str());
                      obj_meta->text_params.display_text = display_text;
                    }else{
                      std::string stdString = "";
                      strncpy(label_info->result_label, stdString.c_str(), sizeof(label_info->result_label) - 1);
                      char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString).c_str());
                      obj_meta->text_params.display_text = display_text;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Only Draw
      if (draw_mode == 1 && update_mode == 0){
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);

          if (obj_meta->unique_component_id == 2){
            if (global_store_id.find(obj_meta->object_id) != global_store_id.end()){
              std::get<0>(global_store_id[obj_meta->object_id]) += 1;
            }
            for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next){
              class_meta = (NvDsClassifierMeta *) (l_class->data);
              for (l_label = class_meta->label_info_list; l_label != NULL; l_label = l_label->next){
                label_info = (NvDsLabelInfo *) (l_label->data); 
                if (class_meta->unique_component_id == 3){
                  std::string stdString_check = std::string(label_info->result_label);
                  if (check_match_template(stdString_check, template_check_list) == true){
                    std::get<1>(global_store_id[obj_meta->object_id]).push_back(stdString_check);
                    std::get<0>(global_store_id[obj_meta->object_id]) = count_frames;
                    char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString_check).c_str());
                    obj_meta->text_params.display_text = display_text;
                  }else{
                    if (global_store_id.find(obj_meta->object_id) != global_store_id.end()) {
                      std::string stdString = std::get<1>(global_store_id[obj_meta->object_id]).back();
                      char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString).c_str());
                      obj_meta->text_params.display_text = display_text;
                    }else{
                      std::string stdString = "";
                      char* display_text = g_strdup(transform_license_plate(obj_meta->text_params.display_text, stdString).c_str());
                      obj_meta->text_params.display_text = display_text;
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Only Update
      if (draw_mode == 0 && update_mode == 1){
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
          obj_meta = (NvDsObjectMeta *) (l_obj->data);

          if (obj_meta->unique_component_id == 2){
            if (global_store_id.find(obj_meta->object_id) != global_store_id.end()){
              std::get<0>(global_store_id[obj_meta->object_id]) += 1;
            }
            for (l_class = obj_meta->classifier_meta_list; l_class != NULL; l_class = l_class->next){
              class_meta = (NvDsClassifierMeta *) (l_class->data);
              for (l_label = class_meta->label_info_list; l_label != NULL; l_label = l_label->next){
                label_info = (NvDsLabelInfo *) (l_label->data); 
                if (class_meta->unique_component_id == 3){
                  std::string stdString_check = std::string(label_info->result_label);
                  if (check_match_template(stdString_check, template_check_list) == true){
                    std::get<1>(global_store_id[obj_meta->object_id]).push_back(stdString_check);
                    std::get<0>(global_store_id[obj_meta->object_id]) = count_frames;
                  }else{
                    if (global_store_id.find(obj_meta->object_id) != global_store_id.end()) {
                      std::string stdString = std::get<1>(global_store_id[obj_meta->object_id]).back();
                      strncpy(label_info->result_label, stdString.c_str(), sizeof(label_info->result_label) - 1);
                    }else{
                      std::string stdString = "";
                      strncpy(label_info->result_label, stdString.c_str(), sizeof(label_info->result_label) - 1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    control();
  }
  g_mutex_unlock (&icdslpr->lpr_mutex);
  flow_ret = GST_FLOW_OK;

  nvds_set_output_system_timestamp (inbuf, GST_ELEMENT_NAME (icdslpr));
  gst_buffer_unmap (inbuf, &in_map_info);

  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
icdslpr_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT(gst_icdslpr_debug, "icdslpr", 1, "icdslpr plugin");
  return gst_element_register(plugin, "icdslpr", GST_RANK_PRIMARY, GST_TYPE_ICDSLPR);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    icdsgst_dslpr,
    DESCRIPTION, icdslpr_plugin_init, "6.3", LICENSE, BINARY_PACKAGE,
    URL)
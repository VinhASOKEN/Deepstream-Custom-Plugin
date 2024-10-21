#include <iostream>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "icdslpr_property_parser.h"

GST_DEBUG_CATEGORY (ICDSLPR_CFG_PARSER_CAT);

#define PARSE_ERROR(details_fmt,...) \
  G_STMT_START { \
    GST_CAT_ERROR (ICDSLPR_CFG_PARSER_CAT, \
        "Failed to parse config file %s: " details_fmt, \
        cfg_file_path, ##__VA_ARGS__); \
    GST_ELEMENT_ERROR (icdslpr, LIBRARY, SETTINGS, \
        ("Failed to parse config file:%s", cfg_file_path), \
        (details_fmt, ##__VA_ARGS__)); \
    goto done; \
  } G_STMT_END

#define CHECK_IF_PRESENT(error, custom_err) \
  G_STMT_START { \
    if (error && error->code != G_KEY_FILE_ERROR_KEY_NOT_FOUND) { \
      std::string errvalue = "Error while setting property, in group ";  \
      errvalue.append(custom_err); \
      PARSE_ERROR ("%s %s", errvalue.c_str(), error->message); \
    } \
  } G_STMT_END

#define CHECK_ERROR(error, custom_err) \
  G_STMT_START { \
    if (error) { \
      std::string errvalue = "Error while setting property, in group ";  \
      errvalue.append(custom_err); \
      PARSE_ERROR ("%s %s", errvalue.c_str(), error->message); \
    } \
  } G_STMT_END

#define CHECK_INT_VALUE_NON_NEGATIVE(prop_name,value, group) \
  G_STMT_START { \
    if ((gint) value < 0) { \
      PARSE_ERROR ("Integer property '%s' in group '%s' can have value >=0", prop_name, group); \
    } \
  } G_STMT_END

#define GET_BOOLEAN_PROPERTY(group, property, field) {\
  field = g_key_file_get_boolean(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
}

#define GET_UINT_PROPERTY(group, property, field) {\
  field = g_key_file_get_integer(key_file, group, property, &error); \
  CHECK_ERROR(error, group); \
  CHECK_INT_VALUE_NON_NEGATIVE(property, field, group);\
}


#define ICDSLPR_PROPERTY "property"
#define ICDSLPR_PROPERTY_ENABLE        "enable"
#define ICDSLPR_PROPERTY_CONFIG_WIDTH  "config-width"
#define ICDSLPR_PROPERTY_CONFIG_HEIGHT "config-height"
#define ICDSLPR_PROPERTY_OBJ_CNT_WIN_MS "obj-cnt-win-in-ms"

#define ICDSLPR_PROPERTY_GROUP_FORMAT_OUTPUT_LPR "format-output-lpr-stream-"
#define ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_TEMPLATE "template-mode"
#define ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_DRAW "draw-mode"
#define ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_UPDATE "update-mode"
#define ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MAX_FRAME_UNSEEN_TO_REMOVE "max-frame-unseen-to-remove"
#define ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MAX_PRE_RESULT_PER_OBJECT "max-pre-result-per-object"


static gboolean
icdslpr_parse_format_output_lpr(GstIcDsLpr * icdslpr, gchar * cfg_file_path, GKeyFile * key_file, gchar * group, guint64 stream_id);

static gboolean
icdslpr_parse_property_group(GstIcDsLpr * icdslpr, gchar * cfg_file_path, GKeyFile * key_file)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  guint obj_cnt_win_in_ms = 0;

  GET_UINT_PROPERTY(ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_CONFIG_WIDTH, icdslpr->configuration_width)
  GET_BOOLEAN_PROPERTY(ICDSLPR_PROPERTY,ICDSLPR_PROPERTY_ENABLE, icdslpr->enable);
  GET_UINT_PROPERTY(ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_CONFIG_HEIGHT, icdslpr->configuration_height)

  obj_cnt_win_in_ms = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_OBJ_CNT_WIN_MS, &error);
  CHECK_IF_PRESENT (error, ICDSLPR_PROPERTY);

  if (error) {
    g_error_free (error);
    error = nullptr;
  }

  GST_CAT_INFO (ICDSLPR_CFG_PARSER_CAT,
      "Parsed %s=%d, %s=%d, %s=%d in group '%s'\n", ICDSLPR_PROPERTY_ENABLE,
      icdslpr->enable, ICDSLPR_PROPERTY_CONFIG_WIDTH,
      icdslpr->configuration_width, ICDSLPR_PROPERTY_CONFIG_HEIGHT,
      icdslpr->configuration_height, ICDSLPR_PROPERTY);

  ret = TRUE;
  icdslpr->obj_cnt_win_in_ms = obj_cnt_win_in_ms;

done:
  return ret;
}


static gboolean
icdslpr_parse_format_output_lpr (GstIcDsLpr * icdslpr,
    gchar * cfg_file_path, GKeyFile * key_file, gchar * group)
{
  g_autoptr (GError) error = nullptr;
  gboolean enable = FALSE;
  g_auto (GStrv) keys = nullptr;
  GStrv key = nullptr;
  gchar *template_check = nullptr;
  std::vector <std::string> template_check_vec;
  ConfigInfo *config_infor = (icdslpr->config_infor);

  keys = g_key_file_get_keys (key_file, group, nullptr, &error);
  CHECK_ERROR (error, group);

  for (key = keys; *key; key++){
    template_check = g_key_file_get_value(key_file, group, *key, &error);
    if (template_check == nullptr) {
        CHECK_ERROR (error, group);
    }
    std::string std_string_text(g_strdup(template_check));
    template_check_vec.push_back(std_string_text);

    g_free(template_check);
    template_check = nullptr;
  }
  
  config_infor->template_mode = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_TEMPLATE, &error);
  config_infor->draw_mode = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_DRAW, &error);
  config_infor->update_mode = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MODE_UPDATE, &error);
  config_infor->max_frame_unseen_to_remove = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MAX_FRAME_UNSEEN_TO_REMOVE, &error);
  config_infor->max_pre_result_per_object = g_key_file_get_integer(key_file, ICDSLPR_PROPERTY, ICDSLPR_PROPERTY_FORMAT_OUTPUT_LPR_MAX_PRE_RESULT_PER_OBJECT, &error);

  for (std::string & tem:template_check_vec)
      config_infor->template_check_list.push_back(tem);

done:
  g_free(template_check);
  return TRUE;
}


//G_DEFINE_AUTO_CLEANUP_FREE_FUNC(GStrv, g_strfreev, nullptr);
/* Parse the icdslpr config file. Returns FALSE in case of an error. */
gboolean
icdslpr_parse_config_file (GstIcDsLpr * icdslpr, gchar * cfg_file_path)
{
  g_autoptr (GError) error = nullptr;
  gboolean ret = FALSE;
  g_auto (GStrv) groups = nullptr;
  gboolean property_present = FALSE;
  GStrv group;
  g_autoptr (GKeyFile) cfg_file = g_key_file_new ();

  if (!ICDSLPR_CFG_PARSER_CAT) {
    GstDebugLevel level;
    GST_DEBUG_CATEGORY_INIT(ICDSLPR_CFG_PARSER_CAT, "icdslpr", 0, NULL);
    level = gst_debug_category_get_threshold(ICDSLPR_CFG_PARSER_CAT);
    if (level < GST_LEVEL_ERROR)
      gst_debug_category_set_threshold (ICDSLPR_CFG_PARSER_CAT, GST_LEVEL_ERROR);
  }

  if (!g_key_file_load_from_file(cfg_file, cfg_file_path, G_KEY_FILE_NONE, &error)){
    PARSE_ERROR ("%s", error->message);
  }
  // Check if 'property' group present
  if (!g_key_file_has_group(cfg_file, ICDSLPR_PROPERTY)) {
    PARSE_ERROR ("Group 'property' not specified");
  }

  g_key_file_set_list_separator(cfg_file, ';');
  groups = g_key_file_get_groups(cfg_file, nullptr);

  for (group = groups; *group; group++) {
    GST_CAT_INFO (ICDSLPR_CFG_PARSER_CAT, "Group found %s \n", *group);
    if (!strcmp (*group, ICDSLPR_PROPERTY)) {
      property_present = icdslpr_parse_property_group(icdslpr, cfg_file_path, cfg_file);
    }else
      if (!strncmp (*group, ICDSLPR_PROPERTY_GROUP_FORMAT_OUTPUT_LPR, sizeof(ICDSLPR_PROPERTY_GROUP_FORMAT_OUTPUT_LPR) - 1)) {
        if (!icdslpr_parse_format_output_lpr(icdslpr, cfg_file_path, cfg_file, *group)) {
          goto done;
        }
    }else {
      g_print ("ICDSLPR_CFG_PARSER: Group '%s' ignored\n", *group);
    }

  }
  if (FALSE == property_present) {
    ret = FALSE;
  } else {
    ret = TRUE;
  }

done:
  return ret;
}

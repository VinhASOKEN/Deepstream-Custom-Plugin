#ifndef __GST_ICDSLPR_H__
#define __GST_ICDSLPR_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "nvbufsurface.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "icds_lpr.h"

/* Package and library details required for plugin_init */
#define PACKAGE "icdslpr"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "IC dslpr plugin for integration with DeepStream on DGPU/Jetson"
#define BINARY_PACKAGE "IC DeepStream dslpr plugin"
#define URL "http://nvidia.com/"


G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstIcDsLpr GstIcDsLpr;
typedef struct _GstIcDsLprClass GstIcDsLprClass;

/* Standard boilerplate stuff */
#define GST_TYPE_ICDSLPR (gst_icdslpr_get_type())
#define GST_ICDSLPR(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_ICDSLPR, GstIcDsLpr))
#define GST_ICDSLPR_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_ICDSLPR, GstIcDsLprClass))
#define GST_ICDSLPR_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_ICDSLPR, GstIcDsLprClass))
#define GST_IS_ICDSLPR(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_ICDSLPR))
#define GST_IS_ICDSLPR_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_ICDSLPR))
#define GST_ICDSLPR_CAST(obj)  ((GstIcDsLpr *)(obj))

struct _GstIcDsLpr
{
  GstBaseTransform base_trans;
  guint unique_id;
  guint64 batch_num;
  GstVideoInfo video_info;
  gint configuration_width;
  gint configuration_height;
  guint batch_size;
  gchar *config_file_path;
  gboolean config_file_parse_successful;
  ConfigInfo *config_infor;
  GMutex lpr_mutex;
  gboolean enable;
  guint obj_cnt_win_in_ms;
};

// Boiler plate stuff
struct _GstIcDsLprClass
{
  GstBaseTransformClass parent_class;
};

GType gst_icdslpr_get_type(void);

G_END_DECLS
#endif /* __GST_ICDSLPR_H__ */

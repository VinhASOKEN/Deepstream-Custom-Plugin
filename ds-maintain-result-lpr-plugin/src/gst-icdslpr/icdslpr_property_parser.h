#ifndef ICDSLPR_PROPERTY_FILE_PARSER_H_
#define ICDSLPR_PROPERTY_FILE_PARSER_H_

#include <gst/gst.h>
#include "gsticdslpr.h"

gboolean
icdslpr_parse_config_file (GstIcDsLpr *icdslpr, gchar *cfg_file_path);


#endif /* ICDSLPR_PROPERTY_FILE_PARSER_H_ */

#ifndef _ICDS_LPR_H_
#define _ICDS_LPR_H_
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
  std::vector <std::string> template_check_list;
  int template_mode;
  int draw_mode;
  int update_mode;
  int max_frame_unseen_to_remove;
  int max_pre_result_per_object;
  int config_width;
  int config_height;

}ConfigInfo;

#ifdef __cplusplus
}
#endif

#endif

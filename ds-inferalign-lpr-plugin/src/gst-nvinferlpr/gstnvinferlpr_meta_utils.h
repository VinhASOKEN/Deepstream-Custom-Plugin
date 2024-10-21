/**
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include "gstnvinferlpr.h"
#include "gstnvinferlpr_impl.h"

#define NVDS_USER_OBJECT_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))

void attach_metadata_detector (GstNvInferLpr * nvinferlpr, GstMiniObject * tensor_out_object,
        GstNvInferLprFrame & frame, NvDsInferDetectionOutput & detection_output,
        float segmentationThreshold);

void attach_metadata_classifier (GstNvInferLpr * nvinferlpr, GstMiniObject * tensor_out_object,
        GstNvInferLprFrame & frame, GstNvInferLprObjectInfo & object_info);

void merge_classification_output (GstNvInferLprObjectHistory & history,
    GstNvInferLprObjectInfo  &new_result);

void attach_metadata_segmentation (GstNvInferLpr * nvinferlpr, GstMiniObject * tensor_out_object,
        GstNvInferLprFrame & frame, NvDsInferSegmentationOutput & segmentation_output);

/* Attaches the raw tensor output to the GstBuffer as metadata. */
void attach_tensor_output_meta (GstNvInferLpr *nvinferlpr, GstMiniObject * tensor_out_object,
        GstNvInferLprBatch *batch, NvDsInferContextBatchOutput *batch_output);

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

#include "cuda_runtime.h"
#include "gstnvinferlpr_allocator.h"

/* Standard GStreamer boiler plate macros */
#define GST_TYPE_NVINFERLPR_ALLOCATOR \
    (gst_nvinferlpr_allocator_get_type ())
#define GST_NVINFERLPR_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVINFERLPR_ALLOCATOR,GstNvInferLprAllocator))
#define GST_NVINFERLPR_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVINFERLPR_ALLOCATOR,GstNvInferLprAllocatorClass))
#define GST_IS_NVINFERLPR_ALLOCATOR(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVINFERLPR_ALLOCATOR))
#define GST_IS_NVINFERLPR_ALLOCATOR_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVINFERLPR_ALLOCATOR))

typedef struct _GstNvInferLprAllocator GstNvInferLprAllocator;
typedef struct _GstNvInferLprAllocatorClass GstNvInferLprAllocatorClass;

G_GNUC_INTERNAL GType gst_nvinferlpr_allocator_get_type (void);

GST_DEBUG_CATEGORY_STATIC (gst_nvinferlpr_allocator_debug);
#define GST_CAT_DEFAULT gst_nvinferlpr_allocator_debug

/**
 * Extends the GstAllocator class. Holds the parameters for allocator.
 */
struct _GstNvInferLprAllocator
{
  GstAllocator allocator;
  guint batch_size;
  guint width;
  guint height;
  NvBufSurfaceColorFormat color_format;
  guint gpu_id;
};

struct _GstNvInferLprAllocatorClass
{
  GstAllocatorClass parent_class;
};

/* Standard boiler plate to create a debug category and initializing the
 * allocator type.
 */
#define _do_init \
    GST_DEBUG_CATEGORY_INIT (gst_nvinferlpr_allocator_debug, "nvinferlprallocator", 0, "nvinferlpr allocator");
#define gst_nvinferlpr_allocator_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstNvInferLprAllocator, gst_nvinferlpr_allocator,
    GST_TYPE_ALLOCATOR, _do_init);

/** Type of the memory allocated by the allocator. This can be used to identify
 * buffers / memories allocated by this allocator. */
#define GST_NVINFERLPR_MEMORY_TYPE "nvinferlpr"

/** Structure allocated internally by the allocator. */
typedef struct
{
  /** Should be the first member of a structure extending GstMemory. */
  GstMemory mem;
  GstNvInferLprMemory mem_infer;
} GstNvInferLprMem;

/* Function called by GStreamer buffer pool to allocate memory using this
 * allocator. */
static GstMemory *
gst_nvinferlpr_allocator_alloc (GstAllocator * allocator, gsize size,
    GstAllocationParams * params)
{
  GstNvInferLprAllocator *inferallocator = GST_NVINFERLPR_ALLOCATOR (allocator);
  GstNvInferLprMem *nvmem = new GstNvInferLprMem;
  GstNvInferLprMemory *tmem = &nvmem->mem_infer;
  NvBufSurfaceCreateParams create_params = { 0 };

  create_params.gpuId = inferallocator->gpu_id;
  create_params.width = inferallocator->width;
  create_params.height = inferallocator->height;
  create_params.size = 0;
  create_params.isContiguous = 1;
  create_params.colorFormat = inferallocator->color_format;
  create_params.layout = NVBUF_LAYOUT_PITCH;
  create_params.memType = NVBUF_MEM_DEFAULT;

  if (NvBufSurfaceCreate (&tmem->surf, inferallocator->batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer pool for nvinferlpr");
    return nullptr;
  }

  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    if (NvBufSurfaceMapEglImage (tmem->surf, -1) != 0) {
      GST_ERROR ("Error: Could not map EglImage from NvBufSurface for nvinferlpr");
      return nullptr;
    }

    tmem->egl_frames.resize (inferallocator->batch_size);
    tmem->cuda_resources.resize (inferallocator->batch_size);
  }

  tmem->frame_memory_ptrs.assign (inferallocator->batch_size, nullptr);

  for (guint i = 0; i < inferallocator->batch_size; i++) {
    if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
      if (cuGraphicsEGLRegisterImage (&tmem->cuda_resources[i],
              tmem->surf->surfaceList[i].mappedAddr.eglImage,
              CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        g_printerr ("Failed to register EGLImage in cuda\n");
        return nullptr;
      }

      if (cuGraphicsResourceGetMappedEglFrame (&tmem->egl_frames[i],
              tmem->cuda_resources[i], 0, 0) != CUDA_SUCCESS) {
        g_printerr ("Failed to get mapped EGL Frame\n");
        return nullptr;
      }
      tmem->frame_memory_ptrs[i] = (char *) tmem->egl_frames[i].frame.pPitch[0];
    }
    else {
      /* Calculate pointers to individual frame memories in the batch memory and
      * insert in the vector. */
      tmem->frame_memory_ptrs[i] = (char *) tmem->surf->surfaceList[i].dataPtr;
    }
  }

  /* Initialize the GStreamer memory structure. */
  gst_memory_init ((GstMemory *) nvmem, (GstMemoryFlags) 0, allocator, nullptr,
      size, params->align, 0, size);

  return (GstMemory *) nvmem;
}

/* Function called by buffer pool for freeing memory using this allocator. */
static void
gst_nvinferlpr_allocator_free (GstAllocator * allocator, GstMemory * memory)
{
  GstNvInferLprMem *nvmem = (GstNvInferLprMem *) memory;
  GstNvInferLprMemory *tmem = &nvmem->mem_infer;
  if(tmem->surf->memType == NVBUF_MEM_SURFACE_ARRAY) {
    GstNvInferLprAllocator *inferallocator = GST_NVINFERLPR_ALLOCATOR (allocator);

    for (size_t i = 0; i < inferallocator->batch_size; i++) {
      cuGraphicsUnregisterResource (tmem->cuda_resources[i]);
    }
  }

  NvBufSurfaceUnMapEglImage (tmem->surf, -1);
  NvBufSurfaceDestroy (tmem->surf);

  delete nvmem;
}

/* Function called when mapping memory allocated by this allocator. Should
 * return pointer to GstNvInferLprMemory. */
static gpointer
gst_nvinferlpr_memory_map (GstMemory * mem, gsize maxsize, GstMapFlags flags)
{
  GstNvInferLprMem *nvmem = (GstNvInferLprMem *) mem;

  return (gpointer) & nvmem->mem_infer;
}

static void
gst_nvinferlpr_memory_unmap (GstMemory * mem)
{
}

/* Standard boiler plate. Assigning implemented function pointers. */
static void
gst_nvinferlpr_allocator_class_init (GstNvInferLprAllocatorClass * klass)
{
  GstAllocatorClass *allocator_class = GST_ALLOCATOR_CLASS (klass);

  allocator_class->alloc = GST_DEBUG_FUNCPTR (gst_nvinferlpr_allocator_alloc);
  allocator_class->free = GST_DEBUG_FUNCPTR (gst_nvinferlpr_allocator_free);
}

/* Standard boiler plate. Assigning implemented function pointers and setting
 * the memory type. */
static void
gst_nvinferlpr_allocator_init (GstNvInferLprAllocator * allocator)
{
  GstAllocator *parent = GST_ALLOCATOR_CAST (allocator);

  parent->mem_type = GST_NVINFERLPR_MEMORY_TYPE;
  parent->mem_map = gst_nvinferlpr_memory_map;
  parent->mem_unmap = gst_nvinferlpr_memory_unmap;
}

/* Create a new allocator of type GST_TYPE_NVINFERLPR_ALLOCATOR and initialize
 * members. */
GstAllocator *
gst_nvinferlpr_allocator_new (guint width, guint height,
    NvBufSurfaceColorFormat color_format, guint batch_size, guint gpu_id)
{
  GstNvInferLprAllocator *allocator = (GstNvInferLprAllocator *)
      g_object_new (GST_TYPE_NVINFERLPR_ALLOCATOR,
      nullptr);

  allocator->width = width;
  allocator->height = height;
  allocator->batch_size = batch_size;
  allocator->gpu_id = gpu_id;
  allocator->color_format = color_format;

  return (GstAllocator *) allocator;
}

GstNvInferLprMemory *
gst_nvinferlpr_buffer_get_memory (GstBuffer * buffer)
{
  GstMemory *mem;

  mem = gst_buffer_peek_memory (buffer, 0);

  if (!mem || !gst_memory_is_type (mem, GST_NVINFERLPR_MEMORY_TYPE))
    return nullptr;

  return &(((GstNvInferLprMem *) mem)->mem_infer);
}

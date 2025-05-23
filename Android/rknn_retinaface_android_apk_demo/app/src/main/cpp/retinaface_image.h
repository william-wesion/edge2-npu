/**
  * @ClassName retinaface_image
  * @Description TODO
  * @Author raul.rao
  * @Date 2022/5/23 11:43
  * @Version 1.0
  */
#ifndef RK_RETINAFACE_DEMO_YOLO_IMAGE_H
#define RK_RETINAFACE_DEMO_YOLO_IMAGE_H

#include <android/log.h>
#include "rknn_api.h"
#include "rga/rga.h"
#include "rga/im2d.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rkretinaface4j", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rkretinaface4j", ##__VA_ARGS__);

typedef struct img_npu_buffer_t
{
  rknn_tensor_mem *p_npu_buf;
  rga_buffer_t rga_buf;
  rga_buffer_handle_t rgb_handle;
  rknn_tensor_attr in_attrs;
}img_npu_buffer;


int create(int im_height, int im_width, int im_channel, char *model_path);
void destroy();
bool run_retinaface(long npu_buf_handle, int camera_width, int camera_height, char *y0, float *y1, char *y2);
int retinaface_post_process(char *grid0_buf, float *grid1_buf, char *grid2_buf,
                      int *boxes, float *scores, int *points);
int colorConvertAndFlip(void *src, int srcFmt, long npu_buf_handle, int dstFmt, int width, int height, int flip);
long create_npu_mem(int img_format); 
void release_npu_mem(long npu_buf_handle); 

#endif //RK_RETINAFACE_DEMO_YOLO_IMAGE_H

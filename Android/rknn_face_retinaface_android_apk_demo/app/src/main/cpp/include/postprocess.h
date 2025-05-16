#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <opencv2/opencv.hpp>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.6
#define BOX_THRESH        0.7
#define FACENET_THRESH    0.5
#define PROP_BOX_SIZE     (OBJ_CLASS_NUM+64)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _KEY_POINT
{
    int point_1_x;
    int point_1_y;
    int point_2_x;
    int point_2_y;
    int point_3_x;
    int point_3_y;
    int point_4_x;
    int point_4_y;
    int point_5_x;
    int point_5_y;
} KEY_POINT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    KEY_POINT point;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, float *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

cv::Mat similarTransform(cv::Mat src,cv::Mat dst);

void l2_normalize(float* input);

float compare_eu_distance(float* input1, float* input2);

float cos_similarity(float* input1, float* input2);

void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

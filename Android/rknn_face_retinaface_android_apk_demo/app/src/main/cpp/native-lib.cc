#include <jni.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <dirent.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <android/log.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "retinaface.h"
#include "facenet.h"
#include "camera_util.h"
#include "rga.h"
#include "rknn_api.h"
using namespace std;

#define LOG_TAG "rknn_demo"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

#define _BASETSD_H

#define PERF_WITH_POST 1

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

rknn_context   retinaface_ctx;
rknn_context   facenet_ctx;
static unsigned char *retinaface_model_data;
static unsigned char *facenet_model_data;

int            retinaface_width      = 0;
int            retinaface_height     = 0;
int            retinaface_channel    = 0;
int            facenet_width      = 0;
int            facenet_height     = 0;
int            facenet_channel    = 0;

float dst_landmark[5][2] = {{54.7065, 73.8519},
                            {105.0454, 73.5734},
                            {80.036, 102.4808},
                            {59.3561, 131.9507},
                            {89.6141, 131.7201}};

cv::Mat dst(5, 2, CV_32FC1, dst_landmark);

rknn_input_output_num facenet_io_num;
const float    nms_threshold      = NMS_THRESH;
const float    box_conf_threshold = BOX_THRESH;
const float    facenet_threshold  = FACENET_THRESH;
rknn_input facenet_inputs[1];
rknn_output facenet_outputs[1];
rknn_input_output_num retinaface_io_num;
rknn_input retinaface_inputs[1];
rknn_output retinaface_outputs[3];
std::vector<float> retinaface_out_scales;
std::vector<int32_t> retinaface_out_zps;
std::vector<float>    facenet_out_scales;
std::vector<int32_t>  facenet_out_zps;
std::vector<float*> lib_feature;

static char* jstringToChar(JNIEnv* env, jstring jstr) {
    char* rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);

    if (alen > 0) {
        rtn = new char[alen + 1];
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}

int identify(int pic_width, int pic_height, int pic_channgel, int flip, unsigned char *pic_data, int pic_len, int *ids, float *scores, int *boxes, int *points)
{
    float *facenet_result;
    cv::Mat img;
    cv::Mat orig_img = cv::imdecode(cv::Mat(1, pic_len, CV_8UC1, pic_data),cv::IMREAD_COLOR);
    if (!orig_img.data) {
        return -1;
    }
    cv::flip(orig_img, orig_img, flip);

    memcpy(dst.data, dst_landmark, 2 * 5 * sizeof(float));

    float scale_w, scale_h;
    int resize_w, resize_h, padding;
    if (pic_width > pic_height) {
        scale_w = (float)retinaface_width / pic_width;
        scale_h = scale_w;
        resize_w = retinaface_width;
        resize_h = (int)(resize_w * pic_height / pic_width);
        padding = resize_w - resize_h;
    } else {
        scale_h = (float)retinaface_height / pic_height;
        scale_w = scale_h;
        resize_h = retinaface_height;
        resize_w = (int)(resize_h * pic_width / pic_height);
        padding = resize_h - resize_w;
    }

    cv::resize(orig_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    detect_result_group_t retinaface_detect_result_group;
    if (pic_width > pic_height) {
        cv::copyMakeBorder(img, img, 0, padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_width, pic_width, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    } else {
        cv::copyMakeBorder(img, img, 0, 0, 0, padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_height, pic_height, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    }
    retinaface_output_release(&retinaface_ctx, retinaface_io_num, retinaface_outputs);

    int identifyCnt = 0;
    for (int i = 0; i < retinaface_detect_result_group.count; i++) {

        float landmark[5][2] = {{(float)retinaface_detect_result_group.results[i].point.point_1_x, (float)retinaface_detect_result_group.results[i].point.point_1_y},
                                {(float)retinaface_detect_result_group.results[i].point.point_2_x, (float)retinaface_detect_result_group.results[i].point.point_2_y},
                                {(float)retinaface_detect_result_group.results[i].point.point_3_x, (float)retinaface_detect_result_group.results[i].point.point_3_y},
                                {(float)retinaface_detect_result_group.results[i].point.point_4_x, (float)retinaface_detect_result_group.results[i].point.point_4_y},
                                {(float)retinaface_detect_result_group.results[i].point.point_5_x, (float)retinaface_detect_result_group.results[i].point.point_5_y}};

        cv::Mat src(5, 2, CV_32FC1, landmark);
        memcpy(src.data, landmark, 2 * 5 * sizeof(float));

        cv::Mat M = similarTransform(src, dst);
        cv::Mat warp;
        cv::warpPerspective(orig_img, warp, M, cv::Size(facenet_width, facenet_height));
        cv::cvtColor(warp, warp, cv::COLOR_BGR2RGB);

        facenet_inference(&facenet_ctx, warp, facenet_io_num, facenet_inputs, facenet_outputs, &facenet_result);

        float max_score = 0;
        for (int j = 0; j < lib_feature.size(); j++)
        {
            float cos_similar;
            cos_similar = cos_similarity(facenet_result, lib_feature[j]);
            if (cos_similar >= facenet_threshold && cos_similar > max_score)
            {
                max_score = cos_similar;
                scores[identifyCnt] = max_score;
                ids[identifyCnt] = j;
                boxes[identifyCnt * 4 + 0] = retinaface_detect_result_group.results[i].box.left;
                boxes[identifyCnt * 4 + 1] = retinaface_detect_result_group.results[i].box.top;
                boxes[identifyCnt * 4 + 2] = retinaface_detect_result_group.results[i].box.right;
                boxes[identifyCnt * 4 + 3] = retinaface_detect_result_group.results[i].box.bottom;
                points[identifyCnt * 10 + 0] = retinaface_detect_result_group.results[i].point.point_1_x;
                points[identifyCnt * 10 + 1] = retinaface_detect_result_group.results[i].point.point_1_y;
                points[identifyCnt * 10 + 2] = retinaface_detect_result_group.results[i].point.point_2_x;
                points[identifyCnt * 10 + 3] = retinaface_detect_result_group.results[i].point.point_2_y;
                points[identifyCnt * 10 + 4] = retinaface_detect_result_group.results[i].point.point_3_x;
                points[identifyCnt * 10 + 5] = retinaface_detect_result_group.results[i].point.point_3_y;
                points[identifyCnt * 10 + 6] = retinaface_detect_result_group.results[i].point.point_4_x;
                points[identifyCnt * 10 + 7] = retinaface_detect_result_group.results[i].point.point_4_y;
                points[identifyCnt * 10 + 8] = retinaface_detect_result_group.results[i].point.point_5_x;
                points[identifyCnt * 10 + 9] = retinaface_detect_result_group.results[i].point.point_5_y;
            }
        }

        if(max_score >= facenet_threshold) {
            identifyCnt++;
        }

        facenet_output_release(&facenet_ctx, facenet_io_num, facenet_outputs);
    }

    return identifyCnt;
}

int init(char *retinaface_model_name, char *facenet_model_name)
{
    create_retinaface(retinaface_model_name, &retinaface_ctx, retinaface_width, retinaface_height, retinaface_channel, retinaface_out_scales, retinaface_out_zps, retinaface_io_num, retinaface_model_data);
    create_facenet(facenet_model_name, &facenet_ctx, facenet_width, facenet_height, facenet_channel, facenet_io_num, facenet_model_data);

    memset(retinaface_inputs, 0, sizeof(retinaface_inputs));
    retinaface_inputs[0].index        = 0;
    retinaface_inputs[0].type         = RKNN_TENSOR_UINT8;
    retinaface_inputs[0].size         = retinaface_width * retinaface_height * retinaface_channel;
    retinaface_inputs[0].fmt          = RKNN_TENSOR_NHWC;
    retinaface_inputs[0].pass_through = 0;

    memset(retinaface_outputs, 0, sizeof(retinaface_outputs));
    for (int i = 0; i < retinaface_io_num.n_output; i++) {
        if (i != 1)
        {
            retinaface_outputs[i].want_float = 0;
        }
        else
        {
            retinaface_outputs[i].want_float = 1;
        }
    }

    memset(facenet_inputs, 0, sizeof(facenet_inputs));
    facenet_inputs[0].index        = 0;
    facenet_inputs[0].type         = RKNN_TENSOR_UINT8;
    facenet_inputs[0].size         = facenet_width * facenet_height * facenet_channel;
    facenet_inputs[0].fmt          = RKNN_TENSOR_NHWC;
    facenet_inputs[0].pass_through = 0;

    memset(facenet_outputs, 0, sizeof(facenet_outputs));
    for (int i = 0; i < facenet_io_num.n_output; i++) {
        facenet_outputs[i].want_float = 1;
    }

    return 0;
}

int deInit()
{
    release_retinaface(&retinaface_ctx, retinaface_model_data);
    release_facenet(&facenet_ctx, facenet_model_data);
    return 0;
}

int face_detect(int pic_width, int pic_height, int pic_channgel, int flip, unsigned char *pic_data, int pic_len, float *scores, int *boxes, int *points)
{
    cv::Mat img;
    cv::Mat orig_img = cv::imdecode(cv::Mat(1, pic_len, CV_8UC1, pic_data),cv::IMREAD_COLOR);
    if (!orig_img.data) {
        return -1;
    }
    cv::flip(orig_img, orig_img, flip);

    float scale_w, scale_h;
    int resize_w, resize_h, padding;
    if (pic_width > pic_height) {
        scale_w = (float)retinaface_width / pic_width;
        scale_h = scale_w;
        resize_w = retinaface_width;
        resize_h = (int)(resize_w * pic_height / pic_width);
        padding = resize_w - resize_h;
    } else {
        scale_h = (float)retinaface_height / pic_height;
        scale_w = scale_h;
        resize_h = retinaface_height;
        resize_w = (int)(resize_h * pic_width / pic_height);
        padding = resize_h - resize_w;
    }

    cv::resize(orig_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    detect_result_group_t retinaface_detect_result_group;
    if (pic_width > pic_height) {
        cv::copyMakeBorder(img, img, 0, padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_width, pic_width, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    } else {
        cv::copyMakeBorder(img, img, 0, 0, 0, padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_height, pic_height, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    }
    retinaface_output_release(&retinaface_ctx, retinaface_io_num, retinaface_outputs);

    for (int i = 0; i < retinaface_detect_result_group.count; i++) {
        scores[i] = retinaface_detect_result_group.results[i].prop;
        boxes[i * 4 + 0] = retinaface_detect_result_group.results[i].box.left;
        boxes[i * 4 + 1] = retinaface_detect_result_group.results[i].box.top;
        boxes[i * 4 + 2] = retinaface_detect_result_group.results[i].box.right;
        boxes[i * 4 + 3] = retinaface_detect_result_group.results[i].box.bottom;
        points[i * 10 + 0] = retinaface_detect_result_group.results[i].point.point_1_x;
        points[i * 10 + 1] = retinaface_detect_result_group.results[i].point.point_1_y;
        points[i * 10 + 2] = retinaface_detect_result_group.results[i].point.point_2_x;
        points[i * 10 + 3] = retinaface_detect_result_group.results[i].point.point_2_y;
        points[i * 10 + 4] = retinaface_detect_result_group.results[i].point.point_3_x;
        points[i * 10 + 5] = retinaface_detect_result_group.results[i].point.point_3_y;
        points[i * 10 + 6] = retinaface_detect_result_group.results[i].point.point_4_x;
        points[i * 10 + 7] = retinaface_detect_result_group.results[i].point.point_4_y;
        points[i * 10 + 8] = retinaface_detect_result_group.results[i].point.point_5_x;
        points[i * 10 + 9] = retinaface_detect_result_group.results[i].point.point_5_y;
    }

    return retinaface_detect_result_group.count;
}


int generate_features(int pic_width, int pic_height, unsigned char *pic_data, int pic_len, float *features)
{
    float *facenet_result = nullptr;
    cv::Mat img;
    cv::Mat orig_img = cv::imdecode(cv::Mat(1, pic_len, CV_8UC1, pic_data),cv::IMREAD_COLOR);
    if (!orig_img.data) {
        return -1;
    }

    memcpy(dst.data, dst_landmark, 2 * 5 * sizeof(float));

    float scale_w, scale_h;
    int resize_w, resize_h, padding;
    if (pic_width > pic_height) {
        scale_w = (float)retinaface_width / pic_width;
        scale_h = scale_w;
        resize_w = retinaface_width;
        resize_h = (int)(resize_w * pic_height / pic_width);
        padding = resize_w - resize_h;
    } else {
        scale_h = (float)retinaface_height / pic_height;
        scale_w = scale_h;
        resize_h = retinaface_height;
        resize_w = (int)(resize_h * pic_width / pic_height);
        padding = resize_h - resize_w;
    }

    cv::resize(orig_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
    detect_result_group_t retinaface_detect_result_group;
    if (pic_width > pic_height) {
        cv::copyMakeBorder(img, img, 0, padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_width, pic_width, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    } else {
        cv::copyMakeBorder(img, img, 0, 0, 0, padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, pic_height, pic_height, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
    }
    retinaface_output_release(&retinaface_ctx, retinaface_io_num, retinaface_outputs);

    for (int i = 0; i < retinaface_detect_result_group.count; i++) {
        float landmark[5][2] = {{(float) retinaface_detect_result_group.results[i].point.point_1_x, (float) retinaface_detect_result_group.results[i].point.point_1_y},
                                {(float) retinaface_detect_result_group.results[i].point.point_2_x, (float) retinaface_detect_result_group.results[i].point.point_2_y},
                                {(float) retinaface_detect_result_group.results[i].point.point_3_x, (float) retinaface_detect_result_group.results[i].point.point_3_y},
                                {(float) retinaface_detect_result_group.results[i].point.point_4_x, (float) retinaface_detect_result_group.results[i].point.point_4_y},
                                {(float) retinaface_detect_result_group.results[i].point.point_5_x, (float) retinaface_detect_result_group.results[i].point.point_5_y}};

        cv::Mat src(5, 2, CV_32FC1, landmark);
        memcpy(src.data, landmark, 2 * 5 * sizeof(float));

        cv::Mat M = similarTransform(src, dst);
        cv::Mat warp;
        cv::warpPerspective(orig_img, warp, M, cv::Size(facenet_width, facenet_height));
        cv::cvtColor(warp, warp, cv::COLOR_BGR2RGB);

        facenet_inference(&facenet_ctx, warp, facenet_io_num, facenet_inputs, facenet_outputs,
                          &facenet_result);
        memcpy(features, facenet_result, sizeof(float) * 128);
        facenet_output_release(&facenet_ctx, facenet_io_num, facenet_outputs);
    }

    return retinaface_detect_result_group.count;
}

int add_feature(float *feature, int size) {
   float *tmp_feature = new float[size];
    memcpy(tmp_feature, feature, sizeof(float) * size);
    lib_feature.push_back(tmp_feature);
    return 0;
}

int clear_features() {
    for(int i = 0; i < lib_feature.size(); i++) {
        float* feature = lib_feature[i];
        delete[] feature;
    }
    lib_feature.clear();
    return 0;
}

int get_features_num() {
    return lib_feature.size();
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1detect(JNIEnv* env, jobject object, jint width, jint height,
                                                       jint channel,
                                                       jint flip,
                                                       jbyteArray data,
                                                       jfloatArray scores,
                                                       jintArray boxes,
                                                       jintArray points
) {
    int ret = 0;
    int len = env->GetArrayLength (data);
    jboolean outputCopy = JNI_FALSE;
    jfloat* const s = env->GetFloatArrayElements(scores, &outputCopy);
    jint*  const b = env->GetIntArrayElements(boxes, &outputCopy);
    jint* const p = env->GetIntArrayElements(points, &outputCopy);
    unsigned char* buf = (unsigned char *)malloc(len);
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(buf));
    ret = face_detect(width, height, channel, flip, buf, len, s, b, p);
    if (buf != nullptr) {
        free(buf);
    }
    env->ReleaseFloatArrayElements(scores, s, 0);
    env->ReleaseIntArrayElements(boxes, b, 0);
    env->ReleaseIntArrayElements(points, p, 0);
    return ret;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1identify(JNIEnv* env, jobject object, jint width, jint height,
                                                      jint channel,
                                                      jint flip,
                                                      jbyteArray data,
                                                      jintArray ids,
                                                      jfloatArray scores,
                                                      jintArray boxes,
                                                      jintArray points
) {
    int ret = 0;
    int len = env->GetArrayLength (data);
    jboolean outputCopy = JNI_FALSE;
    jint*  const i = env->GetIntArrayElements(ids, &outputCopy);
    jfloat* const s = env->GetFloatArrayElements(scores, &outputCopy);
    jint*  const b = env->GetIntArrayElements(boxes, &outputCopy);
    jint* const p = env->GetIntArrayElements(points, &outputCopy);

    unsigned char* buf = (unsigned char *)malloc(len);
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(buf));
    ret = identify(width, height, channel, flip, buf, len, i, s, b, p);
    if (buf != nullptr) {
        free(buf);
    }

    env->ReleaseIntArrayElements(ids, i, 0);
    env->ReleaseFloatArrayElements(scores, s, 0);
    env->ReleaseIntArrayElements(boxes, b, 0);
    env->ReleaseIntArrayElements(points, p, 0);
    return ret;
}


extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1add_1feature(JNIEnv* env, jobject object,
                                                      jfloatArray feature,
                                                      int size) {
    int ret = 0;
    jboolean inputCopy = JNI_FALSE;
    jfloat* const f = env->GetFloatArrayElements(feature, &inputCopy);
    add_feature(f, size);
    env->ReleaseFloatArrayElements(feature, f, 0);
    return ret;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1clear_1features(JNIEnv* env, jobject object) {
    int ret = 0;
    clear_features();
    return ret;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1get_1features_1num(JNIEnv* env, jobject object) {
    int ret = 0;
    return get_features_num();
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1generate_1features(JNIEnv* env, jobject object, jint width, jint height,
                                                      jbyteArray data,
                                                      jint dataLen,
                                                      jfloatArray features
) {
    int ret = 0;
    int len = env->GetArrayLength (data);
    jboolean outputCopy = JNI_FALSE;
    jfloat* const f = env->GetFloatArrayElements(features, &outputCopy);
    unsigned char* buf = (unsigned char *)malloc(len);
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(buf));
    ret = generate_features(width, height, buf, dataLen, f);
    if (buf != nullptr) {
        free(buf);
    }
    env->ReleaseFloatArrayElements(features, f, 0);
    return ret;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1init(JNIEnv* env, jobject object, jstring retinaface_model_path, jstring facenet_model_path) {
    char *retinaface_model_path_p = jstringToChar(env, retinaface_model_path);
    char *facenet_model_path_p = jstringToChar(env, facenet_model_path);
    init(retinaface_model_path_p, facenet_model_path_p);
    return 0;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_FaceRecognition_native_1deInit(JNIEnv* env, jobject object) {
    deInit();
    return 0;
}


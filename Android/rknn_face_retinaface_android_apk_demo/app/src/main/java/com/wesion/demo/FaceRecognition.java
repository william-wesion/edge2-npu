package com.wesion.demo;

public class FaceRecognition {

    static {
        System.loadLibrary("demo");
    }

    public static native int native_init(String retinafaceModelPath, String facenetModelPath);
    public static native int native_deInit();
    public static native int native_identify(int width, int height, int channel, int flip, byte[] data, int[] ids, float[] scores, int[] boxes, int[] points);
    public static native int native_detect(int width, int height, int channel, int flip, byte[] data, float[] scores, int[] boxes, int[] points);
    public static native int native_generate_features(int width, int height,  byte[] data, int dateLen, float[] features);

    public static native int native_add_feature(float[] features, int size);

    public static native int native_get_features_num();
    public static native int native_clear_features();


}

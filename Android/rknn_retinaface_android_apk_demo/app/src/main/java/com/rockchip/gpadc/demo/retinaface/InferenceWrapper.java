package com.rockchip.gpadc.demo.retinaface;

import android.graphics.RectF;

import com.rockchip.gpadc.demo.InferenceResult;
import com.rockchip.gpadc.demo.InferenceResult.OutputBuffer;
import com.rockchip.gpadc.demo.InferenceResult.DetectResultGroup;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by randall on 18-4-18.
 */

public class InferenceWrapper {
    private final String TAG = "rkyolo.InferenceWrapper";

    static {
        System.loadLibrary("rknn4j");
    }

    OutputBuffer mOutputs;
    DetectResultGroup mDetectResults;

    public int OBJ_NUMB_MAX_SIZE = 64;

    public InferenceWrapper() {
    }

    public int initModel(int im_height, int im_width, int im_channel, String modelPath) throws Exception {
        mOutputs = new InferenceResult.OutputBuffer();
        mOutputs.mGrid0Out = new byte[16800 * 4];
        mOutputs.mGrid1Out = new float[16800 * 2];
        mOutputs.mGrid2Out = new byte[16800 * 10];
        if (navite_init(im_height, im_width, im_channel, modelPath) != 0) {
            throw new IOException("rknn init fail!");
        }
        return 0; 
    }


    public void deinit() {
        native_deinit();
        mOutputs.mGrid0Out = null;
        mOutputs.mGrid1Out = null;
        mOutputs.mGrid2Out = null;
        mOutputs = null;

    }

    public InferenceResult.OutputBuffer run(long img_buf_handle, int camera_width, int camera_height) {
//        long startTime = System.currentTimeMillis();
//        long endTime;
        native_run(img_buf_handle, camera_width, camera_height, mOutputs.mGrid0Out, mOutputs.mGrid1Out, mOutputs.mGrid2Out);
//        this.inf_count += 1;
//        endTime = System.currentTimeMillis();
//        this.inf_time += (endTime - startTime);
//        if (this.inf_count >= 100) {
//            float inf_avg = this.inf_time * 1.0f / this.inf_count;
//            Log.w(TAG, String.format("inference avg cost: %.5f ms", inf_avg));
//            this.inf_count = 0;
//            this.inf_time = 0;
//        }
//        Log.i(TAG, String.format("inference count: %d", this.inf_count));
        return  mOutputs;
    }

    public ArrayList<InferenceResult.Recognition> postProcess(InferenceResult.OutputBuffer outputs) {
        ArrayList<InferenceResult.Recognition> recognitions = new ArrayList<InferenceResult.Recognition>();

        mDetectResults = new DetectResultGroup();
        mDetectResults.count = 0;
        mDetectResults.boxes = new int[OBJ_NUMB_MAX_SIZE * 4];
        mDetectResults.scores = new float[OBJ_NUMB_MAX_SIZE];
        mDetectResults.points = new int[OBJ_NUMB_MAX_SIZE * 5 * 2];

        if (null == outputs || null == outputs.mGrid0Out || null == outputs.mGrid1Out
                || null == outputs.mGrid2Out) {
            return recognitions;
        }

        int count = native_post_process(outputs.mGrid0Out, outputs.mGrid1Out, outputs.mGrid2Out,
                mDetectResults.boxes,  mDetectResults.scores, mDetectResults.points);
        if (count < 0) {
            mDetectResults.count = 0;
        } else {
            mDetectResults.count = count;
        }

        for (int i = 0; i < count; ++i) {

            RectF rect = new RectF();
            rect.left = mDetectResults.boxes[i * 4 + 0];
            rect.top = mDetectResults.boxes[i * 4 + 1];
            rect.right = mDetectResults.boxes[i * 4 + 2];
            rect.bottom = mDetectResults.boxes[i * 4 + 3];

            int [] points = new int[10];
            points[0] = mDetectResults.points[i * 10 + 0];
            points[1] = mDetectResults.points[i * 10 + 1];
            points[2] = mDetectResults.points[i * 10 + 2];
            points[3] = mDetectResults.points[i * 10 + 3];
            points[4] = mDetectResults.points[i * 10 + 4];
            points[5] = mDetectResults.points[i * 10 + 5];
            points[6] = mDetectResults.points[i * 10 + 6];
            points[7] = mDetectResults.points[i * 10 + 7];
            points[8] = mDetectResults.points[i * 10 + 8];
            points[9] = mDetectResults.points[i * 10 + 9];

            InferenceResult.Recognition recog = new InferenceResult.Recognition(i, mDetectResults.scores[i], rect, points);

            recognitions.add(recog);
        }
        return recognitions;
    }

    private native int navite_init(int im_height, int im_width, int im_channel, String modelPath);
    private native void native_deinit();
    private native int native_run(long img_buf_handle, int cam_width, int cam_height, byte[] grid0Out, float[] grid1Out, byte[] grid2Out);
    private native int native_post_process(byte[] grid0Out, float[] grid1Out, byte[] grid2Out,
                                           int[] boxes, float[] scores, int [] points);

}
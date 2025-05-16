package com.rockchip.gpadc.demo;

import android.content.res.AssetManager;
import android.graphics.RectF;

import com.rockchip.gpadc.demo.retinaface.InferenceWrapper;

import java.io.IOException;
import java.util.ArrayList;

import static java.lang.System.arraycopy;

public class InferenceResult {

    OutputBuffer mOutputBuffer;
    ArrayList<Recognition> recognitions = null;
    private boolean mIsVaild = false;   //是否需要重新计算

    public void init(AssetManager assetManager) throws IOException {
        mOutputBuffer = new OutputBuffer();
//        mSSDObjectTracker = new ObjectTracker(CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT, 3);
    }

    public void reset() {
        if (recognitions != null) {
            recognitions.clear();
            mIsVaild = true;
        }
    }
    public synchronized void setResult(OutputBuffer outputs) {

        if (mOutputBuffer.mGrid0Out == null) {
            mOutputBuffer.mGrid0Out = outputs.mGrid0Out.clone();
            mOutputBuffer.mGrid1Out = outputs.mGrid1Out.clone();
            mOutputBuffer.mGrid2Out = outputs.mGrid2Out.clone();
        } else {
            arraycopy(outputs.mGrid0Out, 0, mOutputBuffer.mGrid0Out, 0,
                    outputs.mGrid0Out.length);
            arraycopy(outputs.mGrid1Out, 0, mOutputBuffer.mGrid1Out, 0,
                    outputs.mGrid1Out.length);
            arraycopy(outputs.mGrid2Out, 0, mOutputBuffer.mGrid2Out, 0,
                    outputs.mGrid2Out.length);
        }
        mIsVaild = false;
    }

    public synchronized ArrayList<Recognition> getResult(InferenceWrapper mInferenceWrapper) {
        if (!mIsVaild) {
            mIsVaild = true;
            recognitions = mInferenceWrapper.postProcess(mOutputBuffer);
        }

        return recognitions;
    }

    public static class OutputBuffer {
        public byte[] mGrid0Out;
        public float[] mGrid1Out;
        public byte[] mGrid2Out;
    }

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public static class Recognition {


        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final int id;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final float score;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        private int [] points;

        public Recognition(
                final int id, final float score, final RectF location, final int [] points) {
            this.id = id;
            this.score = score;
            this.location = location;
            this.points = points;
            // TODO -- add name field, and show it.
        }

        public int getId() {
            return id;
        }

        public float getScore() {
            return score;
        }

        public int [] getPoints() {
            return points;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

    }

    /**
     * Detected objects, returned from native yolo_post_process
     */
    public static class DetectResultGroup {
        /**
         * detected objects count.
         */
        public int count = 0;

        /**
         * id for each detected object.
         */
        public int[] points;

        /**
         * score for each detected object.
         */
        public float[] scores;

        /**
         * box for each detected object.
         */
        public int[] boxes;

//        public DetectResultGroup(
//                int count, int[] ids, float[] scores, float[] boxes
//        ) {
//            this.count = count;
//            this.ids = ids;
//            this.scores = scores;
//            this.boxes = boxes;
//        }
//
//        public int getCount() {
//            return count;
//        }
//
//        public void setCount(int count) {
//            this.count = count;
//        }
//
//        public int[] getIds() {
//            return ids;
//        }
//
//        public void setIds(int[] ids) {
//            this.ids = ids;
//        }
//
//        public float[] getScores() {
//            return scores;
//        }
//
//        public void setScores(float[] scores) {
//            this.scores = scores;
//        }
//
//        public float[] getBoxes() {
//            return boxes;
//        }
//
//        public void setBoxes(float[] boxes) {
//            this.boxes = boxes;
//        }
    }
}

package com.wesion.demo;

import static java.lang.Thread.sleep;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.ImageView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class IdentifyActivity extends Activity {

    private static final String TAG = "IdentifyActivity";
    private SurfaceView mSurfaceview;
    private SurfaceHolder mSurfaceholder;
    private static final int mPreviewWidth = 1920;
    private static final int mPreviewHeight = 1080;
    private int mVideoWidth;
    private int mVideoHeight;
    private ExecutorService executorService = Executors.newSingleThreadExecutor();
    private Future mFuture;
    private Camera mCamera = null;
    public int mFlip = 1;
    private ImageView mTrackResultView;
    private Bitmap mTrackResultBitmap = null;
    private Canvas mTrackResultCanvas = null;
    private Paint mTrackResultPaint = null;
    private Paint mTrackResultTextPaint = null;
    private PorterDuffXfermode mPorterDuffXfermodeClear;
    private PorterDuffXfermode mPorterDuffXfermodeSRC;
    public static final int OBJ_NUMB_MAX_SIZE = 128;
    private String mCacheDirPath;
    private String mFeatureDirPath;
    private List<String> mNameList = new ArrayList<String>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_identify);
        mSurfaceview = (SurfaceView) findViewById(R.id.surfaceViewCamera);
        mTrackResultView = (ImageView) findViewById(R.id.canvasView);

        mCacheDirPath = getCacheDir().getAbsolutePath();
        File featureDir = new File(mCacheDirPath, "features");
        if (!featureDir.exists()) {
            featureDir.mkdir();
        }
        mFeatureDirPath = mCacheDirPath + "/features";

        String retinaface_model_name = mCacheDirPath + "/retinaface.rknn";
        String facenet_model_name = mCacheDirPath + "/facenet.rknn";
        FaceRecognition.native_init(retinaface_model_name, facenet_model_name);
        FaceRecognition.native_clear_features();

        File folder = new File(mFeatureDirPath);
        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();
            for (File file : files) {
                float[] features = readFloatArrayFromBytes(mFeatureDirPath  + "/" + file.getName());
                Log.d(TAG, "file " + file.getAbsolutePath());
                if (features != null) {
                    mNameList.add(file.getName());
                    FaceRecognition.native_add_feature(features, features.length);
                }
            }
        }

        Log.e(TAG, "features: " + FaceRecognition.native_get_features_num());
    }

    private float[] readFloatArrayFromBytes(String filePath) {
        try (FileInputStream fis = new FileInputStream(filePath)) {
            byte[] bytes = new byte[fis.available()];
            fis.read(bytes);
            ByteBuffer buffer = ByteBuffer.wrap(bytes);
            FloatBuffer floatBuffer = buffer.asFloatBuffer();
            float[] result = new float[floatBuffer.remaining()];
            floatBuffer.get(result);
            return result;
        } catch (IOException e) {
            return null;
        }
    }

    @Override
    public void onStart() {
        super.onStart();
        openCamera();
    }

    @Override
    public void onPause() {
        super.onPause();
        releaseCamera();
    }

    private Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int w, int h) {
        final double aspectTolerance = 0.1;
        double targetRatio = (double) w / h;
        if (sizes == null) {
            return null;
        }

        Camera.Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // Try to find an size match aspect ratio and size
        for (Camera.Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > aspectTolerance) {
                continue;
            }
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Camera.Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        return optimalSize;
    }

    private void openCamera() {
        if (this.checkCameraHardware(this)) {
            try {
                mCamera = Camera.open(0);
                mSurfaceholder = mSurfaceview.getHolder();
                mSurfaceholder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
                mSurfaceholder.addCallback(new surfaceholderCallbackBack());
                if (mCamera != null && mSurfaceholder != null) {
                    Camera.Parameters params = mCamera.getParameters();
                    List<Camera.Size> sizeList = params.getSupportedPreviewSizes();
                    final Camera.Size optionSize = getOptimalPreviewSize(sizeList, mPreviewWidth, mPreviewHeight);

                    if (optionSize.width == mPreviewWidth && optionSize.height == mPreviewHeight) {
                        mVideoWidth = mPreviewWidth;
                        mVideoHeight = mPreviewHeight;
                    } else {
                        mVideoWidth = optionSize.width;
                        mVideoHeight = optionSize.height;
                    }

                    params.setPreviewSize(mVideoWidth, mVideoHeight);
                    mCamera.setParameters(params);
                    mCamera.setPreviewCallback(new Camera.PreviewCallback() {
                        @Override
                        public void onPreviewFrame(byte[] data, Camera cam) {
                            onGetCameraData(data, cam, mVideoWidth, mVideoHeight);
                        }
                    });

                    mCamera.setPreviewDisplay(mSurfaceholder);
                    mCamera.setDisplayOrientation(0);
                    mCamera.startPreview();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static int sp2px(float spValue) {
        Resources r = Resources.getSystem();
        final float scale = r.getDisplayMetrics().scaledDensity;
        return (int) (spValue * scale + 0.5f);
    }

    private void showTrackSelectResults(int width, int height, DetectResultGroup detectResultGroup) {

        if (mTrackResultBitmap == null) {
            mTrackResultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            mTrackResultCanvas = new Canvas(mTrackResultBitmap);

            mTrackResultPaint = new Paint();
            mTrackResultPaint.setColor(0xff416FDA);
            mTrackResultPaint.setStrokeJoin(Paint.Join.ROUND);
            mTrackResultPaint.setStrokeCap(Paint.Cap.ROUND);
            mTrackResultPaint.setStrokeWidth(4);
            mTrackResultPaint.setStyle(Paint.Style.STROKE);
            mTrackResultPaint.setTextAlign(Paint.Align.LEFT);
            mTrackResultPaint.setTextSize(sp2px(10));
            mTrackResultPaint.setTypeface(Typeface.SANS_SERIF);
            mTrackResultPaint.setFakeBoldText(false);

            mTrackResultTextPaint = new Paint();
            mTrackResultTextPaint.setColor(0xff416FDA);
            mTrackResultTextPaint.setStrokeWidth(2);
            mTrackResultTextPaint.setTextAlign(Paint.Align.LEFT);
            mTrackResultTextPaint.setTextSize(sp2px(12));
            mTrackResultTextPaint.setTypeface(Typeface.SANS_SERIF);
            mTrackResultTextPaint.setFakeBoldText(false);
            mPorterDuffXfermodeClear = new PorterDuffXfermode(PorterDuff.Mode.CLEAR);
            mPorterDuffXfermodeSRC = new PorterDuffXfermode(PorterDuff.Mode.SRC);
        }

        // clear canvas
        mTrackResultPaint.setXfermode(mPorterDuffXfermodeClear);
        mTrackResultCanvas.drawPaint(mTrackResultPaint);
        mTrackResultPaint.setXfermode(mPorterDuffXfermodeSRC);

        for (int i = 0; i < detectResultGroup.count; ++i) {
            RectF detection = new RectF();
            detection.left = detectResultGroup.boxes[i * 4 + 0];
            detection.top = detectResultGroup.boxes[i * 4 + 1];
            detection.right = detectResultGroup.boxes[i * 4 + 2];
            detection.bottom = detectResultGroup.boxes[i * 4 + 3];

            Paint paint = new Paint();
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.RED);
            mTrackResultCanvas.drawCircle(detectResultGroup.points[i * 10 + 0], detectResultGroup.points[i * 10 + 1], 3, paint);
            mTrackResultCanvas.drawCircle(detectResultGroup.points[i * 10 + 2], detectResultGroup.points[i * 10 + 3], 3, paint);
            paint.setColor(Color.GREEN);
            mTrackResultCanvas.drawCircle(detectResultGroup.points[i * 10 + 4], detectResultGroup.points[i * 10 + 5], 3, paint);
            paint.setColor(Color.BLUE);
            mTrackResultCanvas.drawCircle(detectResultGroup.points[i * 10 + 6], detectResultGroup.points[i * 10 + 7], 3, paint);
            mTrackResultCanvas.drawCircle(detectResultGroup.points[i * 10 + 8], detectResultGroup.points[i * 10 + 9], 3, paint);

            mTrackResultCanvas.drawRect(detection, mTrackResultPaint);
            mTrackResultCanvas.drawText( mNameList.get(detectResultGroup.ids[i]), detection.left+5, detection.bottom-5, mTrackResultTextPaint);
        }

        mTrackResultView.setScaleType(ImageView.ScaleType.FIT_XY);
        mTrackResultView.setImageBitmap(mTrackResultBitmap);
    }

    private void onGetCameraData(final byte[] data, final Camera camera, final int width, final int height) {
        if (mFuture != null && !mFuture.isDone()) {
            return;
        }

        mFuture = executorService.submit(new Runnable() {
            @Override
            public void run() {
                Camera.Parameters parameters = camera.getParameters();
                int format = parameters.getPreviewFormat();
                final YuvImage image = new YuvImage(data, format, width, height, null);
                ByteArrayOutputStream os = new ByteArrayOutputStream(data.length);
                if (!image.compressToJpeg(new Rect(0, 0, width, height), 100, os)) {
                    return;
                }

                byte[] imageBytes = os.toByteArray();
                if (imageBytes != null) {
                    DetectResultGroup detectResultGroup = new DetectResultGroup();
                    detectResultGroup.count = 0;
                    detectResultGroup.scores = new float[OBJ_NUMB_MAX_SIZE];
                    detectResultGroup.ids = new int[OBJ_NUMB_MAX_SIZE];
                    detectResultGroup.boxes = new int[OBJ_NUMB_MAX_SIZE * 4];
                    detectResultGroup.points = new int[OBJ_NUMB_MAX_SIZE * 5 * 2];
                    detectResultGroup.count = FaceRecognition.native_identify(width, height, 3, mFlip, imageBytes, detectResultGroup.ids, detectResultGroup.scores, detectResultGroup.boxes, detectResultGroup.points);
                    showTrackSelectResults(width, height, detectResultGroup);
                }
            }
        });
    }


    private boolean checkCameraHardware(Context context) {
        if (context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)) {
            // this device has a camera
            return true;
        } else {
            // no camera on this device
            return false;
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        releaseCamera();
        FaceRecognition.native_deInit();
    }

    private void releaseCamera() {
        if (mCamera != null) {
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
            mSurfaceholder = null;
        }
    }

    class surfaceholderCallbackBack implements SurfaceHolder.Callback {
        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            int cameraCount = Camera.getNumberOfCameras();
            if (cameraCount > 0 && mCamera != null) {
                try {
                    mCamera.setPreviewDisplay(holder);
                    mCamera.startPreview();
                } catch (IOException e) {
                    e.printStackTrace();
                    mCamera.release();
                }
            }
        }

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            if(null != mCamera) {
                mCamera.setPreviewCallback(null);
                mCamera.stopPreview();
                mCamera.release();
                mCamera = null;
            }
        }
    }

}

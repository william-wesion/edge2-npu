package com.wesion.demo;

public class DetectResultGroup {
    /**
     * detected objects count.
     */
    public int count = 0;

    public int [] ids;
    /**
     * score for each detected object.
     */
    public float[] scores;

    /**
     * box for each detected object.
     */
    public int[] boxes;

    public int[] points;

}

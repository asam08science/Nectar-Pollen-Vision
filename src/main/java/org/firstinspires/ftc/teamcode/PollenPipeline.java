package org.firstinspires.ftc.teamcode;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

public class PollenPipeline extends OpenCvPipeline {

    private Mat circles = new Mat();
    private Mat hsv = new Mat();
    private Mat pollenMask = new Mat();
    private Mat binary = new Mat();
    private Mat erode = new Mat();
    private Mat dilate = new Mat();
    private Mat reset = new Mat();   // cleaned binary mask used for circle detection
    private Mat binaryPreview = new Mat(); // RGB binary preview with circle overlays
    private Mat pollen = new Mat();  // RGB preview frame used for overlays and return value
    private Mat pollenPositions = new Mat();  // Field coordinates: [fieldX, fieldY] for each pollen

    Mat erodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new org.opencv.core.Size(3,3));
    Mat diolateKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new org.opencv.core.Size(3,3));
    Mat resetKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new org.opencv.core.Size(2,2));

    // Private HoughCircles tuning value not to be tuned
    private static double dp = 1.0;
    // HoughCircles tuning values; declared public and static so they can be adjusted at runtime for tuning
    public static double minDist = 12;
    public static int param1 = 40;
    public static int param2 = 11;
    public static int minRadius = 3;
    public static int maxRadius = 22;
    public static int isBinary = 0;
    public static Scalar pollenLower = new Scalar(22, 30, 100);
    public static Scalar pollenUpper = new Scalar(36, 255, 255);
    
    // Useless, ignore
    private double sphereX = -1;
    private double sphereY = -1;
    private double sphereRadius = -1;
    private boolean targetDetected = false;

    @Override
    public Mat processFrame(Mat input) {
        Mat topHalf = input.submat(0, input.rows() / 2, 0, input.cols());
    
        // Set all pixels in this sub-matrix to black
        topHalf.setTo(new Scalar(0, 0, 0));
    
        // Always release submats to prevent memory leaks in loop iterations
        topHalf.release();
        
        // Isolate pollen pixels by HSV range thresholding.
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        Core.inRange(hsv, pollenLower, pollenUpper, pollenMask);
        binary = pollenMask;

        // Morphological cleanup tuned for recall: gentle open/close, then light blur.
        Imgproc.morphologyEx(binary, erode, Imgproc.MORPH_OPEN, erodeKernel);
        Imgproc.morphologyEx(erode, dilate, Imgproc.MORPH_CLOSE, diolateKernel);
        Imgproc.erode(dilate, reset, resetKernel);
        Imgproc.GaussianBlur(reset, reset, new org.opencv.core.Size(3, 3), 0);

        // Keep a binary-view debug frame for tuning.
        Imgproc.cvtColor(reset, binaryPreview, Imgproc.COLOR_GRAY2RGB);

        // Draw annotations on a copy of the source frame and return it.
        input.copyTo(pollen);

        // Clear previous frame detections so stale circles are never reused.
        circles.release();
        circles = new Mat();

        // // Detect circular blobs from the cleaned binary mask.
        Imgproc.HoughCircles(reset, circles, Imgproc.HOUGH_GRADIENT,
                             dp, minDist, param1, param2, minRadius, maxRadius);

        int numCircles = circles.empty() ? 0 : circles.cols();

        if (numCircles > 0) {
            targetDetected = true;
            convertCirclesToPollenPositions(numCircles);

            double[] firstData = circles.get(0, 0);
            if (firstData != null) {
                sphereX = firstData[0];
                sphereY = firstData[1];
                sphereRadius = firstData[2];
            }

            for (int i = 0; i < numCircles; i++) {
                double[] data = circles.get(0, i);
                if (data == null) continue;

                Point center = new Point(Math.round(data[0]), Math.round(data[1]));
                int r = (int) Math.round(data[2]);
                
                if (isBinary == 0) {
                    Imgproc.circle(pollen, center, r, new Scalar(0, 255, 0), 2);
                    Imgproc.circle(pollen, center, 3, new Scalar(255, 0, 0), -1);
                } else {
                    Imgproc.circle(binaryPreview, center, r, new Scalar(0, 255, 0), 2);
                    Imgproc.circle(binaryPreview, center, 3, new Scalar(255, 0, 0), -1);
                }
            }
        } else {
            targetDetected = false;
            sphereX = -1;
            sphereY = -1;
            sphereRadius = -1;
            pollenPositions.release();
            pollenPositions = new Mat();
        }
        
        if (isBinary == 0) {
            return pollen;
        } else {
            return binaryPreview;
        }
    }

    /**
     * Converts circle detections (pixel coordinates + radius) to field positions.
     * For each circle: radius → distance (via Desmos function) → field coordinates
     */
    private void convertCirclesToPollenPositions(int numCircles) {
        pollenPositions.release();
        pollenPositions = new Mat();

        for (int i = 0; i < numCircles; i++) {
            double[] circleData = circles.get(0, i);
            if (circleData == null) continue;

            double pixelX = circleData[0];
            double pixelY = circleData[1];
            double radiusPixels = circleData[2];

            // Convert radius to real-world distance
            double distance = radiusToDistance(radiusPixels);

            // Convert pixel coordinates + distance to field coordinates
            double[] fieldCoords = pixelToFieldCoordinates(pixelX, pixelY, distance);

            // Store as [fieldX, fieldY] in pollenPositions
            // pollenPositions will be 1 row × numCircles columns, each column = [fieldX, fieldY]
            pollenPositions.push_back(new org.opencv.core.Mat(1, 2, org.opencv.core.CvType.CV_64F));
            pollenPositions.put(0, i * 2, fieldCoords);
        }
    }

    /**
     * Convert pixel radius to real-world distance using calibration function.
     * TODO: Replace with actual Desmos equation once calibrated
     * 
     * Example Desmos functions to try:
     * - Linear: distance = a * radius + b
     * - Inverse: distance = c / (radius - offset)
     * - Polynomial: distance = a*radius^2 + b*radius + c
     */
    private double radiusToDistance(double radiusPixels) {
        // PLACEHOLDER: Replace with actual calibration curve
        // distance = 50 / (radiusPixels - 5);  // Example inverse relationship
        // distance = -2.5 * radiusPixels + 80;  // Example linear relationship
        
        double distance = 0;  //TODO: Implement Desmos function here
        return distance;
    }

    /**
     * Convert pixel coordinates (camera frame) + distance to field coordinates.
     * 
     * Assumes:
     * - pixelX, pixelY are from camera frame (origin at top-left)
     * - distance is computed from radius
     * - Field coordinates have their own origin and orientation
     * 
     * TODO: Update with actual camera calibration parameters
     */
    private double[] pixelToFieldCoordinates(double pixelX, double pixelY, double distance) {
        double fieldX = 0;  //TODO: Implement transformation
        double fieldY = 0;  //TODO: Implement transformation

        // Camera intrinsics / calibration needed here:
        // - Principal point (cx, cy)
        // - Focal length (fx, fy)
        // - Field origin and orientation relative to camera
        // - Camera mounting height/angle

        return new double[]{fieldX, fieldY};
    }

    /**
     * Getter for pollen field positions
     */
    public Mat getPollenPositions() {
        return pollenPositions;
    }
}
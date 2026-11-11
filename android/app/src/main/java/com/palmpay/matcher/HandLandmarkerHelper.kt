package com.palmpay.matcher

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

/**
 * Helper to initialize and run Google MediaPipe Hand Landmarker (21 keypoints).
 */
class HandLandmarkerHelper(
    private val context: Context,
    private val minHandDetectionConfidence: Float = 0.6f,
    private val minHandPresenceConfidence: Float = 0.6f
) {
    private var handLandmarker: HandLandmarker? = null

    init {
        setupHandLandmarker()
    }

    private fun setupHandLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setMinHandDetectionConfidence(minHandDetectionConfidence)
            .setMinHandPresenceConfidence(minHandPresenceConfidence)
            .setMinTrackingConfidence(0.6f)
            .setNumHands(1)
            .setRunningMode(RunningMode.IMAGE)
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    /**
     * Detects 21 landmarks from a Bitmap.
     * Returns list of 21 PointF pixel coordinates on the image, or null if no hand found.
     */
    fun detect(bitmap: Bitmap): List<PointF>? {
        val landmarker = handLandmarker ?: return null
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: HandLandmarkerResult = landmarker.detect(mpImage)

        val landmarksList = result.landmarks()
        if (landmarksList.isEmpty()) {
            return null
        }

        val firstHand = landmarksList[0]
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        return firstHand.map { lm ->
            PointF(lm.x() * w, lm.y() * h)
        }
    }

    fun close() {
        handLandmarker?.close()
        handLandmarker = null
    }
}


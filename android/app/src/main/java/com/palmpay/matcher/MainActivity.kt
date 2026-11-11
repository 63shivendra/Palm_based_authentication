package com.palmpay.matcher

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Base64
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.palmpay.matcher.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.io.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var handHelper: HandLandmarkerHelper? = null
    private val palmMatcher = PalmMatcher()

    private var isEnrollMode = true
    private val enrolledTemplates = mutableListOf<PalmMatcher.Template>()

    private var lastAnalyzedBitmap: Bitmap? = null
    private var lastLandmarks: List<android.graphics.PointF>? = null

    private var isProcessing = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, R.string.camera_permission_request, Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        handHelper = HandLandmarkerHelper(this)

        loadTemplatesFromStorage()
        setupUI()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupUI() {
        binding.toggleModeGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (isChecked) {
                when (checkedId) {
                    R.id.btnEnrollMode -> {
                        isEnrollMode = true
                        binding.tvModeTitle.text = "MODE: ENROLLMENT"
                        binding.tvModeTitle.setTextColor(ContextCompat.getColor(this, R.color.guide_cyan))
                        binding.btnCaptureEnroll.isEnabled = true
                        binding.tvStatus.text = "Position palm & tap Capture Sample"
                    }
                    R.id.btnVerifyMode -> {
                        if (enrolledTemplates.isEmpty()) {
                            Toast.makeText(this, "Please enroll at least 1 palm first!", Toast.LENGTH_SHORT).show()
                            binding.toggleModeGroup.check(R.id.btnEnrollMode)
                            return@addOnButtonCheckedListener
                        }
                        isEnrollMode = false
                        binding.tvModeTitle.text = "MODE: LIVE VERIFICATION"
                        binding.tvModeTitle.setTextColor(ContextCompat.getColor(this, R.color.match_green))
                        binding.btnCaptureEnroll.isEnabled = false
                        binding.tvStatus.text = "Show palm to camera for live match"
                    }
                }
            }
        }

        binding.btnCaptureEnroll.setOnClickListener {
            captureEnrollmentSample()
        }

        binding.btnResetTemplate.setOnClickListener {
            enrolledTemplates.clear()
            clearTemplatesFromStorage()
            binding.tvMetrics.text = "Templates cleared. Enroll a palm to begin."
            binding.toggleModeGroup.check(R.id.btnEnrollMode)
            Toast.makeText(this, "Enrolled templates reset.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageProxy(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("PalmPay", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    private fun processImageProxy(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }
        isProcessing = true

        val bitmap = imageProxyToBitmap(imageProxy)
        imageProxy.close()

        val helper = handHelper ?: run {
            isProcessing = false
            return
        }

        val landmarks = helper.detect(bitmap)
        lastAnalyzedBitmap = bitmap
        lastLandmarks = landmarks

        if (landmarks == null) {
            runOnUiThread {
                binding.overlayView.setResults(null, bitmap.width, bitmap.height, OverlayView.MatchStatus.SEARCHING)
                if (!isEnrollMode) {
                    binding.tvStatus.text = "Waiting for open palm..."
                }
            }
            isProcessing = false
            return
        }

        // Hand is detected!
        if (isEnrollMode) {
            runOnUiThread {
                binding.overlayView.setResults(landmarks, bitmap.width, bitmap.height, OverlayView.MatchStatus.TRACKING)
                binding.tvStatus.text = "Hand detected! Hold still and tap Capture."
                binding.tvMetrics.text = "Enrolled samples: ${enrolledTemplates.size}/3"
            }
        } else {
            // Live Verify Mode
            try {
                val roi = palmMatcher.alignPalm(bitmap, landmarks)
                val (isMatch, distance) = palmMatcher.match(enrolledTemplates, roi)

                runOnUiThread {
                    val status = if (isMatch) OverlayView.MatchStatus.MATCHED else OverlayView.MatchStatus.NOT_MATCHED
                    binding.overlayView.setResults(landmarks, bitmap.width, bitmap.height, status)

                    if (isMatch) {
                        binding.tvStatus.text = getString(R.string.status_matched)
                        binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.match_green))
                        triggerHapticFeedback()
                    } else {
                        binding.tvStatus.text = getString(R.string.status_not_matched)
                        binding.tvStatus.setTextColor(ContextCompat.getColor(this, R.color.reject_red))
                    }
                    binding.tvMetrics.text = "Distance: ${String.format("%.3f", distance)} | Threshold: 0.30"
                }
            } catch (qe: PalmMatcher.QualityException) {
                runOnUiThread {
                    binding.overlayView.setResults(landmarks, bitmap.width, bitmap.height, OverlayView.MatchStatus.TRACKING)
                    binding.tvStatus.text = qe.message
                }
            } catch (e: Exception) {
                Log.e("PalmPay", "Verification error", e)
            }
        }

        isProcessing = false
    }

    private fun captureEnrollmentSample() {
        val bmp = lastAnalyzedBitmap
        val lms = lastLandmarks

        if (bmp == null || lms == null) {
            Toast.makeText(this, "No hand detected. Position your palm clearly.", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val roi = palmMatcher.alignPalm(bmp, lms)
            val template = palmMatcher.encodeRoi(roi)
            enrolledTemplates.add(template)
            saveTemplatesToStorage()

            triggerHapticFeedback()
            Toast.makeText(this, "Captured sample ${enrolledTemplates.size}!", Toast.LENGTH_SHORT).show()

            if (enrolledTemplates.size >= 1) {
                binding.tvMetrics.text = "Enrolled: ${enrolledTemplates.size} photo(s). Switch to Live Verify!"
            }
        } catch (qe: PalmMatcher.QualityException) {
            Toast.makeText(this, qe.message, Toast.LENGTH_SHORT).show()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val width = imageProxy.width
        val height = imageProxy.height

        val bitmap = Bitmap.createBitmap(
            width + (rowStride - pixelStride * width) / pixelStride,
            height,
            Bitmap.Config.ARGB_8888
        )
        bitmap.copyPixelsFromBuffer(buffer)

        // Rotate bitmap according to sensor orientation
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotationDegrees.toFloat())
            return Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, width, height)
    }

    private fun triggerHapticFeedback() {
        try {
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
                val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vibratorManager.defaultVibrator.vibrate(VibrationEffect.createOneShot(80, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                @Suppress("DEPRECATION")
                vibrator.vibrate(80)
            }
        } catch (_: Exception) {}
    }

    private fun saveTemplatesToStorage() {
        val prefs = getSharedPreferences("palm_vault", Context.MODE_PRIVATE)
        val baos = ByteArrayOutputStream()
        val oos = ObjectOutputStream(baos)
        oos.writeInt(enrolledTemplates.size)
        for (t in enrolledTemplates) {
            oos.writeObject(t)
        }
        oos.flush()
        val base64 = Base64.encodeToString(baos.toByteArray(), Base64.DEFAULT)
        prefs.edit().putString("templates_blob", base64).apply()
    }

    private fun loadTemplatesFromStorage() {
        val prefs = getSharedPreferences("palm_vault", Context.MODE_PRIVATE)
        val blob = prefs.getString("templates_blob", null) ?: return
        try {
            val bytes = Base64.decode(blob, Base64.DEFAULT)
            val ois = ObjectInputStream(ByteArrayInputStream(bytes))
            val count = ois.readInt()
            enrolledTemplates.clear()
            for (i in 0 until count) {
                enrolledTemplates.add(ois.readObject() as PalmMatcher.Template)
            }
            if (enrolledTemplates.isNotEmpty()) {
                binding.tvMetrics.text = "Loaded ${enrolledTemplates.size} enrolled sample(s)."
            }
        } catch (e: Exception) {
            Log.e("PalmPay", "Error loading templates", e)
        }
    }

    private fun clearTemplatesFromStorage() {
        val prefs = getSharedPreferences("palm_vault", Context.MODE_PRIVATE)
        prefs.edit().remove("templates_blob").apply()
    }

    private fun allPermissionsGranted() = ContextCompat.checkSelfPermission(
        baseContext, Manifest.permission.CAMERA
    ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        handHelper?.close()
    }
}


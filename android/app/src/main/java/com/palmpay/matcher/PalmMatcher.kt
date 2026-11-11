package com.palmpay.matcher

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
import java.io.Serializable
import kotlin.math.*

/**
 * PalmMatcher: 100% Kotlin port of palm_match.py
 *
 * Implements Competitive Code (CompCode):
 * 1. Landmark-based geometric upright alignment & palm center crop (128x128).
 * 2. Quality gate: distance bounds, aspect ratio, Laplacian sharpness check.
 * 3. Local illumination bandpass normalization.
 * 4. 6-orientation Gabor filter bank (dominant crease direction).
 * 5. Multi-rotation & shift-invariant angular circular distance matching.
 */
class PalmMatcher {

    companion object {
        const val SIZE = 128
        const val N_ORI = 6
        const val MAX_SHIFT = 4
        val ROTATIONS = floatArrayOf(-6f, -3f, 0f, 3f, 6f)
        const val KEEP_FRACTION = 0.40f
        const val DEFAULT_THRESHOLD = 0.30f

        const val ROI_BETA = 0.50f
        const val ROI_GAMMA = 0.90f
        const val MIN_PALM_FRACTION = 0.12f
        const val MAX_PALM_FRACTION = 0.85f
        const val MIN_SHARPNESS = 15.0f

        const val KERNEL_SIZE = 21
    }

    data class Template(
        val codes: ByteArray,    // 128x128 byte array (values 0..5)
        val weights: FloatArray  // 128x128 float array (0.0 or 1.0)
    ) : Serializable

    class QualityException(message: String) : Exception(message)

    // Pre-computed 6 Gabor filters (21x21, zero-meaned)
    private val gaborBank: Array<FloatArray> = createGaborBank()

    private fun createGaborBank(): Array<FloatArray> {
        val bank = Array(N_ORI) { FloatArray(KERNEL_SIZE * KERNEL_SIZE) }
        val half = KERNEL_SIZE / 2
        val sigma = 3.5
        val lambd = 10.0
        val gamma = 0.5
        val psi = 0.0

        for (k in 0 until N_ORI) {
            val theta = PI * k / N_ORI
            val cosT = cos(theta)
            val sinT = sin(theta)
            var sum = 0.0

            for (y in -half..half) {
                for (x in -half..half) {
                    val xPrime = x * cosT + y * sinT
                    val yPrime = -x * sinT + y * cosT
                    val expVal = exp(-(xPrime * xPrime + gamma * gamma * yPrime * yPrime) / (2 * sigma * sigma))
                    val cosVal = cos(2 * PI * xPrime / lambd + psi)
                    val v = (expVal * cosVal).toFloat()
                    val idx = (y + half) * KERNEL_SIZE + (x + half)
                    bank[k][idx] = v
                    sum += v
                }
            }
            // Zero-mean the filter
            val mean = (sum / (KERNEL_SIZE * KERNEL_SIZE)).toFloat()
            for (i in bank[k].indices) {
                bank[k][i] -= mean
            }
        }
        return bank
    }

    /**
     * Extracts an aligned 128x128 grayscale palm ROI from a full bitmap using 21 MediaPipe landmarks.
     */
    fun alignPalm(bitmap: Bitmap, landmarks: List<PointF>): FloatArray {
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        val p0 = landmarks[0]   // Wrist
        val p5 = landmarks[5]   // Index MCP
        val p9 = landmarks[9]   // Middle MCP
        val p17 = landmarks[17] // Pinky MCP

        val palmW = hypot(p17.x - p5.x, p17.y - p5.y)
        if (palmW < MIN_PALM_FRACTION * w || palmW > MAX_PALM_FRACTION * w) {
            throw QualityException("Hand too far or too close. Keep palm at normal distance.")
        }

        // Wrist -> Middle knuckle vector
        val vX = p9.x - p0.x
        val vY = p9.y - p0.y
        val angleDeg = Math.toDegrees(atan2(vX.toDouble(), -vY.toDouble())).toFloat()

        // Rotate hand upright around Middle MCP (p9)
        val rotMatrix = Matrix()
        rotMatrix.setRotate(angleDeg, p9.x, p9.y)

        val pts = floatArrayOf(p5.x, p5.y, p9.x, p9.y, p17.x, p17.y)
        rotMatrix.mapPoints(pts)
        val q5x = pts[0]; val q5y = pts[1]
        val q9y = pts[3]
        val q17x = pts[4]; val q17y = pts[5]

        val xc = (q5x + q17x) / 2.0f
        val yc = q9y + ROI_BETA * palmW
        val halfCrop = (ROI_GAMMA * palmW) / 2.0f

        // Map crop back to original image coordinates
        val invMatrix = Matrix()
        rotMatrix.invert(invMatrix)

        val cropSrc = floatArrayOf(
            xc - halfCrop, yc - halfCrop,
            xc + halfCrop, yc - halfCrop,
            xc - halfCrop, yc + halfCrop
        )
        invMatrix.mapPoints(cropSrc)

        // Affine transform to 128x128
        val dst = floatArrayOf(
            0f, 0f,
            SIZE.toFloat(), 0f,
            0f, SIZE.toFloat()
        )
        val transform = Matrix()
        transform.setPolyToPoly(cropSrc, 0, dst, 0, 3)

        val cropBitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(cropBitmap)
        val paint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)

        val invTransform = Matrix()
        transform.invert(invTransform)
        canvas.drawBitmap(bitmap, invTransform, paint)

        // Convert to 128x128 float grayscale [0..255]
        val pixels = IntArray(SIZE * SIZE)
        cropBitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val gray = FloatArray(SIZE * SIZE)
        for (i in pixels.indices) {
            val c = pixels[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }
        cropBitmap.recycle()

        // Sharpness quality check via Laplacian variance
        val sharpness = computeLaplacianVariance(gray, SIZE, SIZE)
        if (sharpness < MIN_SHARPNESS) {
            throw QualityException("Photo too blurry (sharpness: ${sharpness.toInt()}). Hold hand still.")
        }

        return gray
    }

    private fun computeLaplacianVariance(img: FloatArray, width: Int, height: Int): Float {
        var sum = 0.0
        var sumSq = 0.0
        val count = (width - 2) * (height - 2)

        for (y in 1 until height - 1) {
            val row = y * width
            for (x in 1 until width - 1) {
                val lap = img[row + x + 1] + img[row + x - 1] +
                          img[row + width + x] + img[row - width + x] - 4.0f * img[row + x]
                sum += lap
                sumSq += lap * lap
            }
        }
        val mean = sum / count
        return ((sumSq / count) - (mean * mean)).toFloat()
    }

    /**
     * Gaussian illumination bandpass normalization:
     * d = g - GaussianBlur(g, 12)
     * norm = d / (sqrt(GaussianBlur(d^2, 12)) + 1e-3)
     */
    private fun normalise(gray: FloatArray): FloatArray {
        val blurred = gaussianBlur(gray, SIZE, SIZE, sigma = 4.0f)
        val d = FloatArray(gray.size)
        val dSq = FloatArray(gray.size)
        for (i in gray.indices) {
            val diff = gray[i] - blurred[i]
            d[i] = diff
            dSq[i] = diff * diff
        }

        val blurredVar = gaussianBlur(dSq, SIZE, SIZE, sigma = 4.0f)
        val norm = FloatArray(gray.size)
        for (i in gray.indices) {
            norm[i] = (d[i] / (sqrt(blurredVar[i]) + 1e-3f))
        }
        return norm
    }

    /**
     * Fast separable Gaussian blur on 128x128.
     */
    private fun gaussianBlur(src: FloatArray, width: Int, height: Int, sigma: Float): FloatArray {
        val radius = (3 * sigma).toInt().coerceIn(3, 15)
        val kernelSize = 2 * radius + 1
        val kernel = FloatArray(kernelSize)
        var sum = 0f
        val twoSigmaSq = 2f * sigma * sigma
        for (i in -radius..radius) {
            val v = exp(-(i * i) / twoSigmaSq)
            kernel[i + radius] = v
            sum += v
        }
        for (i in kernel.indices) kernel[i] /= sum

        val temp = FloatArray(src.size)
        val dest = FloatArray(src.size)

        // Horizontal pass
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                var acc = 0f
                for (k in -radius..radius) {
                    val px = (x + k).coerceIn(0, width - 1)
                    acc += src[row + px] * kernel[k + radius]
                }
                temp[row + x] = acc
            }
        }

        // Vertical pass
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                var acc = 0f
                for (k in -radius..radius) {
                    val py = (y + k).coerceIn(0, height - 1)
                    acc += temp[py * width + x] * kernel[k + radius]
                }
                dest[row + x] = acc
            }
        }
        return dest
    }

    /**
     * Encodes 128x128 grayscale palm patch into Competitive Code (code, weight).
     */
    fun encodeRoi(roi: FloatArray, rotationDeg: Float = 0f): Template {
        val rotated = if (rotationDeg != 0f) rotatePatch(roi, rotationDeg) else roi
        val norm = normalise(rotated)

        val halfK = KERNEL_SIZE / 2
        val codes = ByteArray(SIZE * SIZE)
        val strengths = FloatArray(SIZE * SIZE)

        // Convolve with 6 Gabor filters and find the minimum response (darkest line direction)
        for (y in 0 until SIZE) {
            val row = y * SIZE
            for (x in 0 until SIZE) {
                var minVal = Float.MAX_VALUE
                var minOri: Byte = 0

                for (k in 0 until N_ORI) {
                    val filter = gaborBank[k]
                    var resp = 0f

                    for (ky in -halfK..halfK) {
                        val py = (y + ky).coerceIn(0, SIZE - 1)
                        val kRow = (ky + halfK) * KERNEL_SIZE
                        val pRow = py * SIZE
                        for (kx in -halfK..halfK) {
                            val px = (x + kx).coerceIn(0, SIZE - 1)
                            resp += norm[pRow + px] * filter[kRow + (kx + halfK)]
                        }
                    }

                    if (resp < minVal) {
                        minVal = resp
                        minOri = k.toByte()
                    }
                }
                val idx = row + x
                codes[idx] = minOri
                strengths[idx] = -minVal
            }
        }

        // Threshold top KEEP_FRACTION (40%) most line-like pixels
        val sortedStrengths = strengths.clone()
        sortedStrengths.sort()
        val cutoffIdx = ((1.0f - KEEP_FRACTION) * (SIZE * SIZE)).toInt().coerceIn(0, SIZE * SIZE - 1)
        val cutoff = sortedStrengths[cutoffIdx]

        val weights = FloatArray(SIZE * SIZE)
        for (i in strengths.indices) {
            weights[i] = if (strengths[i] > cutoff) 1.0f else 0.0f
        }

        return Template(codes, weights)
    }

    private fun rotatePatch(src: FloatArray, degrees: Float): FloatArray {
        val rad = Math.toRadians(degrees.toDouble()).toFloat()
        val cosA = cos(rad)
        val sinA = sin(rad)
        val center = (SIZE - 1) / 2f
        val out = FloatArray(src.size)

        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                val dx = x - center
                val dy = y - center
                val srcX = (center + dx * cosA + dy * sinA).roundToInt().coerceIn(0, SIZE - 1)
                val srcY = (center - dx * sinA + dy * cosA).roundToInt().coerceIn(0, SIZE - 1)
                out[y * SIZE + x] = src[srcY * SIZE + srcX]
            }
        }
        return out
    }

    /**
     * Circular angular agreement between orientations a and b (0..5):
     * diff = min(|a - b|, 6 - |a - b|)
     * agreement = 1.0 - diff / 3.0
     */
    private fun agreement(a: Byte, b: Byte): Float {
        val diff = abs(a.toInt() - b.toInt())
        val circ = min(diff, N_ORI - diff)
        return 1.0f - circ / (N_ORI / 2.0f)
    }

    /**
     * Distance between single template and single probe over residual shifts [-4..+4].
     */
    private fun distanceOne(t: Template, p: Template): Float {
        var bestDist = 1.0f

        for (dy in -MAX_SHIFT..MAX_SHIFT) {
            val yStartA = max(0, dy)
            val yEndA = SIZE + min(0, dy)
            val yStartB = max(0, -dy)

            for (dx in -MAX_SHIFT..MAX_SHIFT) {
                val xStartA = max(0, dx)
                val xEndA = SIZE + min(0, dx)
                val xStartB = max(0, -dx)

                var numerator = 0f
                var sumWa = 0f
                var sumWb = 0f

                for (ya in yStartA until yEndA) {
                    val yb = yStartB + (ya - yStartA)
                    val rowA = ya * SIZE
                    val rowB = yb * SIZE

                    for (xa in xStartA until xEndA) {
                        val xb = xStartB + (xa - xStartA)
                        val idxA = rowA + xa
                        val idxB = rowB + xb

                        val wa = t.weights[idxA]
                        val wb = p.weights[idxB]

                        sumWa += wa
                        sumWb += wb
                        if (wa > 0f && wb > 0f) {
                            numerator += wa * wb * agreement(t.codes[idxA], p.codes[idxB])
                        }
                    }
                }

                val den = sqrt(sumWa * sumWb) + 1e-6f
                val dist = 1.0f - (numerator / den)
                if (dist < bestDist) {
                    bestDist = dist
                }
            }
        }
        return bestDist
    }

    /**
     * Computes verification distance against multi-rotation probe and returns (isMatch, distance).
     */
    fun match(enrolledTemplates: List<Template>, probeRoi: FloatArray, threshold: Float = DEFAULT_THRESHOLD): Pair<Boolean, Float> {
        // Multi-rotation probe
        val probes = ROTATIONS.map { rot -> encodeRoi(probeRoi, rot) }

        var minDistance = Float.MAX_VALUE
        for (tpl in enrolledTemplates) {
            for (prb in probes) {
                val d = distanceOne(tpl, prb)
                if (d < minDistance) {
                    minDistance = d
                }
            }
        }
        return Pair(minDistance <= threshold, minDistance)
    }
}


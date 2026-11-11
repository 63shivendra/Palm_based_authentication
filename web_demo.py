#!/usr/bin/env python3
"""
web_demo.py - Interactive live palm verification web application.
Runs locally on http://localhost:5000
Supports:
  1. Real-time webcam capture via HTML5 getUserMedia
  2. One-click preset palm samples from the dataset (Female F101 genuine pair, Male M101 impostor)
  3. Live palm landmark detection, 128x128 ROI alignment & sharpness check
  4. 6-orientation Competitive Code visualization
  5. Live 1:1 Enrollment and Verification with exact distance scoring
"""
import base64
import os
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

import palm_match

app = Flask(__name__)

# In-memory storage for enrolled templates: list of (code, weight)
ENROLLED_TEMPLATES = []

# Known sample presets from the repository
PRESET_SAMPLES = {
    "f101_sample1": "data/PalmMatchDB_ROI/PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F101(15-24)/F12MB/left/ei_1608938994605-removebg-preview.png",
    "f101_sample2": "data/PalmMatchDB_ROI/PalmMatchDB/FEMALE PALMPRINTS/female Age(15-24)/F101(15-24)/F12MB/left/ei_1608939044925-removebg-preview (1).png",
    "m101_sample1": "data/PalmMatchDB_ROI/PalmMatchDB/MALE PALMPRINTS/m101 age 60/ml101 12m/ei_1611580528741-removebg-preview (10).png",
}

def decode_image(b64_string):
    """Decodes a data-URL or base64 string to a BGR numpy image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_png_b64(bgr_or_gray):
    """Encodes a BGR or Gray image to a base64 PNG data URL."""
    _, buf = cv2.imencode(".png", bgr_or_gray)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def render_compcode(code, weight):
    """Renders Competitive Code (0..5) as an HSV color map where weight masks brightness."""
    h, w = code.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # 6 orientations mapped to 0..150 degrees in OpenCV HSV (0..180)
    hsv[:, :, 0] = (code * 30).astype(np.uint8)
    hsv[:, :, 1] = 220
    hsv[:, :, 2] = (weight * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PalmPay Matcher - Live Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  </style>
</head>
<body class="min-h-screen p-4 md:p-8">
  <div class="max-w-5xl mx-auto">
    <!-- Header -->
    <header class="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 border-b border-gray-800 gap-4">
      <div>
        <h1 class="text-2xl font-bold text-white flex items-center gap-2">
          <span class="text-emerald-400">✋ PalmPay</span> Live Matcher
        </h1>
        <p class="text-sm text-gray-400 mt-1">1:1 Contactless Palm Verification via MediaPipe & Competitive Code</p>
      </div>
      <div class="flex items-center gap-3">
        <span id="enrollBadge" class="px-3 py-1 rounded-full text-xs font-semibold bg-gray-800 text-gray-300 border border-gray-700">
          Enrolled: 0 Samples
        </span>
        <button id="btnReset" class="px-3 py-1 bg-red-900/40 hover:bg-red-800/60 text-red-300 text-xs rounded border border-red-700 transition">
          Reset Template
        </button>
      </div>
    </header>

    <!-- Main Content -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
      
      <!-- Left Column: Video / Input Stream (7 cols) -->
      <div class="lg:col-span-7 space-y-4">
        <div class="bg-gray-900 rounded-xl p-4 border border-gray-800 shadow-xl">
          <div class="flex items-center justify-between mb-3">
            <span class="text-xs uppercase tracking-wider font-semibold text-gray-400">Live Camera / Input Source</span>
            <div class="flex gap-2">
              <button id="btnStartCam" class="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded transition">
                Start Webcam
              </button>
              <button id="btnStopCam" class="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium rounded transition hidden">
                Stop Webcam
              </button>
            </div>
          </div>

          <!-- Video / Canvas Container -->
          <div class="relative aspect-[4/3] bg-black rounded-lg overflow-hidden border border-gray-800 flex items-center justify-center">
            <video id="videoElement" autoplay playsinline muted class="absolute inset-0 w-full h-full object-cover hidden"></video>
            <canvas id="displayCanvas" class="absolute inset-0 w-full h-full object-cover"></canvas>
            <div id="noCamPlaceholder" class="text-center p-6 text-gray-500">
              <div class="text-4xl mb-2">📷</div>
              <p class="text-sm">Webcam idle</p>
              <p class="text-xs text-gray-600 mt-1">Click "Start Webcam" above or choose a preset test image below</p>
            </div>
            
            <!-- Live Status HUD Pill -->
            <div id="hudPill" class="absolute top-3 left-3 px-3 py-1 bg-black/70 backdrop-blur rounded-full text-xs font-mono text-cyan-400 border border-cyan-500/30 hidden">
              READY
            </div>
          </div>

          <!-- Controls: Action Buttons -->
          <div class="grid grid-cols-2 gap-3 mt-4">
            <button id="btnEnroll" class="py-2.5 px-4 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-semibold text-sm rounded-lg transition shadow-lg shadow-emerald-900/30 flex items-center justify-center gap-2">
              <span>📝 Enroll This Palm</span>
            </button>
            <button id="btnVerify" class="py-2.5 px-4 bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-white font-semibold text-sm rounded-lg transition shadow-lg shadow-cyan-900/30 flex items-center justify-center gap-2">
              <span>🔍 Verify Live Match</span>
            </button>
          </div>

          <!-- Presets for easy testing without webcam -->
          <div class="mt-4 pt-4 border-t border-gray-800">
            <div class="text-xs font-semibold text-gray-400 mb-2">Or test with dataset presets:</div>
            <div class="flex flex-wrap gap-2">
              <button onclick="loadPreset('f101_sample1')" class="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-xs rounded border border-gray-700 text-gray-300">
                Subject A (Photo 1)
              </button>
              <button onclick="loadPreset('f101_sample2')" class="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-xs rounded border border-gray-700 text-gray-300">
                Subject A (Photo 2 - Genuine)
              </button>
              <button onclick="loadPreset('m101_sample1')" class="px-2.5 py-1 bg-gray-800 hover:bg-gray-700 text-xs rounded border border-gray-700 text-red-300">
                Subject B (Impostor)
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Verification Results & Biometric Visuals (5 cols) -->
      <div class="lg:col-span-5 space-y-4">
        
        <!-- Verification Decision Card -->
        <div id="resultCard" class="bg-gray-900 rounded-xl p-5 border border-gray-800 shadow-xl transition-all duration-300">
          <div class="text-xs uppercase tracking-wider font-semibold text-gray-400 mb-2">Verification Decision</div>
          
          <div id="statusBanner" class="py-3 px-4 rounded-lg bg-gray-800/80 border border-gray-700 text-center text-sm font-bold text-gray-400 transition-colors">
            WAITING FOR ENROLLMENT & VERIFY
          </div>

          <div class="grid grid-cols-2 gap-3 mt-4 text-center">
            <div class="bg-gray-950 p-3 rounded-lg border border-gray-800">
              <div class="text-xs text-gray-500">Calculated Distance</div>
              <div id="scoreDistance" class="text-2xl font-black font-mono text-white mt-1">--</div>
            </div>
            <div class="bg-gray-950 p-3 rounded-lg border border-gray-800">
              <div class="text-xs text-gray-500">Decision Threshold</div>
              <div class="text-2xl font-black font-mono text-gray-400 mt-1">0.300</div>
            </div>
          </div>
          
          <p id="statusExplanation" class="text-xs text-gray-400 text-center mt-3">
            Distance &le; 0.300 is accepted as Genuine Match.
          </p>
        </div>

        <!-- Biometric Pipeline Internals -->
        <div class="bg-gray-900 rounded-xl p-5 border border-gray-800 shadow-xl">
          <div class="text-xs uppercase tracking-wider font-semibold text-gray-400 mb-3">Biometric Feature Maps</div>
          
          <div class="grid grid-cols-2 gap-4">
            <!-- 128x128 Palm ROI -->
            <div class="flex flex-col items-center">
              <span class="text-xs text-gray-400 mb-1.5 font-medium">Aligned Palm ROI (128x128)</span>
              <div class="w-32 h-32 bg-black rounded-lg border border-gray-800 overflow-hidden flex items-center justify-center shadow-inner">
                <img id="roiImg" class="w-full h-full object-cover hidden" alt="Palm ROI">
                <span id="roiPlaceholder" class="text-xs text-gray-600">Pending</span>
              </div>
            </div>

            <!-- 128x128 Competitive Code -->
            <div class="flex flex-col items-center">
              <span class="text-xs text-gray-400 mb-1.5 font-medium">Competitive Code Map</span>
              <div class="w-32 h-32 bg-black rounded-lg border border-gray-800 overflow-hidden flex items-center justify-center shadow-inner">
                <img id="compCodeImg" class="w-full h-full object-cover hidden" alt="Competitive Code">
                <span id="compCodePlaceholder" class="text-xs text-gray-600">Pending</span>
              </div>
            </div>
          </div>

          <div class="mt-4 pt-3 border-t border-gray-800 text-[11px] text-gray-500 leading-relaxed">
            <span class="text-emerald-400 font-semibold">Competitive Code:</span> 6 Gabor directional filters convolve with the normalized palm ROI. The winning line angle per pixel is recorded into a compact direction map and compared under ±4px spatial shifts.
          </div>
        </div>

      </div>
    </div>
  </div>

  <script>
    let video = document.getElementById('videoElement');
    let canvas = document.getElementById('displayCanvas');
    let ctx = canvas.getContext('2d');
    let stream = null;
    let enrolledCount = 0;

    // Start webcam
    document.getElementById('btnStartCam').addEventListener('click', async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
        });
        video.srcObject = stream;
        video.classList.remove('hidden');
        document.getElementById('noCamPlaceholder').classList.add('hidden');
        document.getElementById('btnStartCam').classList.add('hidden');
        document.getElementById('btnStopCam').classList.remove('hidden');
        document.getElementById('hudPill').classList.remove('hidden');

        video.onloadedmetadata = () => {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          renderLoop();
        };
      } catch (err) {
        alert("Camera access denied or unavailable: " + err.message + "\\nYou can still use the Dataset Presets below!");
      }
    });

    document.getElementById('btnStopCam').addEventListener('click', () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
      }
      video.classList.add('hidden');
      document.getElementById('noCamPlaceholder').classList.remove('hidden');
      document.getElementById('btnStartCam').classList.remove('hidden');
      document.getElementById('btnStopCam').classList.add('hidden');
      document.getElementById('hudPill').classList.add('hidden');
    });

    function renderLoop() {
      if (!stream) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Draw target guide oval
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const rx = canvas.width * 0.25;
      const ry = canvas.height * 0.32;

      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(0, 229, 255, 0.4)";
      ctx.lineWidth = 3;
      ctx.stroke();

      requestAnimationFrame(renderLoop);
    }

    function captureFrameB64() {
      const snapCanvas = document.createElement('canvas');
      snapCanvas.width = canvas.width || 640;
      snapCanvas.height = canvas.height || 480;
      const snapCtx = snapCanvas.getContext('2d');
      snapCtx.drawImage(canvas, 0, 0, snapCanvas.width, snapCanvas.height);
      return snapCanvas.toDataURL('image/jpeg', 0.9);
    }

    // Enroll Button
    document.getElementById('btnEnroll').addEventListener('click', async () => {
      const imgB64 = captureFrameB64();
      setLoading("Enrolling palm...");
      
      try {
        const resp = await fetch('/api/enroll', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imgB64 })
        });
        const data = await resp.json();
        if (data.ok) {
          enrolledCount = data.enrolled_count;
          updateEnrollBadge();
          showBiometricMaps(data.roi_b64, data.compcode_b64);
          showStatus("ENROLLED SUCCESSFULLY (" + enrolledCount + " sample)", "text-emerald-400", "bg-emerald-950/60", "border-emerald-700");
          document.getElementById('scoreDistance').innerText = "--";
          document.getElementById('statusExplanation').innerText = "Template stored in memory. Now position palm and click 'Verify Live Match'.";
        } else {
          showStatus(data.error || "Enrollment rejected", "text-amber-400", "bg-amber-950/60", "border-amber-700");
        }
      } catch (e) {
        showStatus("Error: " + e.message, "text-red-400", "bg-red-950/60", "border-red-700");
      }
    });

    // Verify Button
    document.getElementById('btnVerify').addEventListener('click', async () => {
      if (enrolledCount === 0) {
        alert("Please enroll at least one palm photo first!");
        return;
      }
      const imgB64 = captureFrameB64();
      setLoading("Matching palm...");

      try {
        const resp = await fetch('/api/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imgB64 })
        });
        const data = await resp.json();
        if (data.ok) {
          showBiometricMaps(data.roi_b64, data.compcode_b64);
          document.getElementById('scoreDistance').innerText = data.distance.toFixed(3);

          if (data.match) {
            showStatus("✅ MATCH VERIFIED (Same Palm)", "text-emerald-300", "bg-emerald-900/70", "border-emerald-500");
            document.getElementById('statusExplanation').innerText = "Distance (" + data.distance.toFixed(3) + ") <= Threshold (0.300). Access Granted.";
          } else {
            showStatus("❌ NO MATCH (Different Palm)", "text-red-300", "bg-red-900/70", "border-red-500");
            document.getElementById('statusExplanation').innerText = "Distance (" + data.distance.toFixed(3) + ") > Threshold (0.300). Access Denied.";
          }
        } else {
          showStatus(data.error || "Quality rejection", "text-amber-400", "bg-amber-950/60", "border-amber-700");
        }
      } catch (e) {
        showStatus("Error: " + e.message, "text-red-400", "bg-red-950/60", "border-red-700");
      }
    });

    // Reset Button
    document.getElementById('btnReset').addEventListener('click', async () => {
      await fetch('/api/reset', { method: 'POST' });
      enrolledCount = 0;
      updateEnrollBadge();
      showStatus("TEMPLATES RESET", "text-gray-400", "bg-gray-800", "border-gray-700");
      document.getElementById('scoreDistance').innerText = "--";
    });

    // Preset Loader
    async function loadPreset(presetKey) {
      setLoading("Loading preset " + presetKey + "...");
      const resp = await fetch('/api/load_preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset: presetKey })
      });
      const data = await resp.json();
      if (data.ok) {
        const img = new Image();
        img.onload = () => {
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);
          document.getElementById('noCamPlaceholder').classList.add('hidden');
          showStatus("LOADED PRESET: " + presetKey, "text-cyan-400", "bg-cyan-950/60", "border-cyan-700");
        };
        img.src = data.image_b64;
      }
    }

    function showBiometricMaps(roiB64, compCodeB64) {
      if (roiB64) {
        const r = document.getElementById('roiImg');
        r.src = roiB64;
        r.classList.remove('hidden');
        document.getElementById('roiPlaceholder').classList.add('hidden');
      }
      if (compCodeB64) {
        const c = document.getElementById('compCodeImg');
        c.src = compCodeB64;
        c.classList.remove('hidden');
        document.getElementById('compCodePlaceholder').classList.add('hidden');
      }
    }

    function showStatus(text, textColor, bgColor, borderColor) {
      const banner = document.getElementById('statusBanner');
      banner.className = "py-3 px-4 rounded-lg border text-center text-sm font-bold transition-all " + textColor + " " + bgColor + " " + borderColor;
      banner.innerText = text;
    }

    function setLoading(msg) {
      showStatus(msg, "text-cyan-300", "bg-cyan-950/40", "border-cyan-800");
    }

    function updateEnrollBadge() {
      const b = document.getElementById('enrollBadge');
      b.innerText = "Enrolled: " + enrolledCount + " Samples";
      b.className = enrolledCount > 0 
        ? "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-700" 
        : "px-3 py-1 rounded-full text-xs font-semibold bg-gray-800 text-gray-300 border border-gray-700";
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/load_preset", methods=["POST"])
def api_load_preset():
    data = request.json or {}
    key = data.get("preset", "")
    path = PRESET_SAMPLES.get(key)
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": f"Preset {key} not found"}), 404
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return jsonify({"ok": True, "image_b64": encode_png_b64(bgr)})

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    global ENROLLED_TEMPLATES
    data = request.json or {}
    raw_img = data.get("image")
    if not raw_img:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        bgr = decode_image(raw_img)
        # Try MediaPipe detection first; if no hand found, fallback to centre crop
        try:
            roi = palm_match.align_palm(bgr, palm_match.detect_landmarks(bgr))
        except palm_match.QualityError:
            roi = palm_match.centre_crop(bgr)

        code, weight = palm_match.encode_roi(roi)
        ENROLLED_TEMPLATES.append((code, weight))

        compcode_vis = render_compcode(code, weight)
        return jsonify({
            "ok": True,
            "enrolled_count": len(ENROLLED_TEMPLATES),
            "roi_b64": encode_png_b64(roi),
            "compcode_b64": encode_png_b64(compcode_vis)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/verify", methods=["POST"])
def api_verify():
    global ENROLLED_TEMPLATES
    if not ENROLLED_TEMPLATES:
        return jsonify({"ok": False, "error": "No template enrolled yet"}), 400

    data = request.json or {}
    raw_img = data.get("image")
    if not raw_img:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        bgr = decode_image(raw_img)
        try:
            roi = palm_match.align_palm(bgr, palm_match.detect_landmarks(bgr))
        except palm_match.QualityError:
            roi = palm_match.centre_crop(bgr)

        code, weight = palm_match.encode_roi(roi)
        probe = [palm_match.encode_roi(roi, r) for r in palm_match.ROTATIONS]
        dist = palm_match.distance(ENROLLED_TEMPLATES, probe)
        is_match = dist <= palm_match.DEFAULT_THRESHOLD

        compcode_vis = render_compcode(code, weight)
        return jsonify({
            "ok": True,
            "match": bool(is_match),
            "distance": float(dist),
            "threshold": float(palm_match.DEFAULT_THRESHOLD),
            "roi_b64": encode_png_b64(roi),
            "compcode_b64": encode_png_b64(compcode_vis)
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global ENROLLED_TEMPLATES
    ENROLLED_TEMPLATES = []
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


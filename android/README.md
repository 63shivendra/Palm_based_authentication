# PalmPay Matcher - Native Android Live App

A real-time, on-device 1:1 contactless palmprint enrollment and verification application written in Kotlin.

## Architecture

```
CameraX (30 FPS Stream)
       │
       ▼
Google MediaPipe Hand Landmarker (21 3D Landmarks via GPU)
       │
       ▼
PalmMatcher (Pure Kotlin Competitive Code Engine)
  ├─ 1. Upright Alignment & Center Crop (128x128)
  ├─ 2. Sharpness & Distance Quality Gates (Laplacian Var >= 15.0)
  ├─ 3. Local Bandpass Illumination Normalization
  ├─ 4. 6-Orientation Gabor Filter Bank (CompCode)
  └─ 5. Multi-Rotation [-6°, +6°] & Shift [±4px] Angular Distance Matching
       │
       ▼
Live Verification HUD (Real-Time Green / Red Feedback in < 30ms)
```

## How to Open & Run in Android Studio

1. Launch **Android Studio** (Hedgehog or newer recommended).
2. Click **Open** and select the `d:\PALM_payments\android` directory.
3. Allow Gradle to sync. All dependencies (`CameraX`, `MediaPipe Tasks Vision`, `Material Components`) will be resolved automatically.
4. Connect an Android device with Developer Mode / USB Debugging enabled (Android 7.0+ / API 24+).
5. Click **Run 'app'** (`Shift + F10`).

## App Usage

1. **Enrollment Mode**:
   - Align your open palm inside the on-screen guide oval.
   - When the skeleton turns cyan, tap **Capture Sample** (you can capture 1 to 3 samples from slightly varied angles).
   - Templates are encrypted and saved locally in `palm_vault`.
2. **Live Verify Mode**:
   - Toggle to **Live Verify**.
   - Show your hand to the camera.
   - The app continuously aligns and matches frames:
     - **MATCH**: Green ring + haptic vibration (distance $\le 0.30$).
     - **NO MATCH**: Red ring (distance $> 0.30$).


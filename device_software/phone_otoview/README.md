# OtoView (phone client)

Android phone client for the digital otoscope. It connects to the Raspberry Pi
otoscope server, shows the live H.264 video, sends LED and shutdown commands,
captures still frames, and runs a 4-class on-device classifier on the captured
images.

## System context

The phone joins the Pi's Wi-Fi hotspot and opens two TCP connections to the Pi:

| Channel | Port | Purpose |
|---|---|---|
| Video | 1234 | raw H.264 stream, decoded on the phone |
| Command | 4321 | `command_light_on`, `command_light_off`, `command_quit` |

## Connecting to your Pi

The server address and ports are set at the top of `MainActivity.kt`
(`defaultIp`, `defaultVideoPort`, `defaultCmdPort`; defaults `10.42.0.1`, 1234,
4321). If your Pi uses a different address or ports, change these to match the
server configuration.

## Screens

- Main: live view with Start, Capture, LED, and Gallery.
- Gallery: saved captures; for each image, the classifier's top class and per-class probabilities.

## Video receive/decode

The H.264 stream is received and decoded with the app's own implementation:
`stream/AnnexBReader.kt` splits the Annex-B bytestream into NAL units,
`stream/H264TcpDecoder.kt` feeds them to `MediaCodec`, and
`stream/VideoStreamManager.kt` manages the TCP connection, retries, and state.

## On-device model

Inference uses PyTorch Mobile (Lite). `app/src/main/assets/model.ptl` is the
traced 4-class classifier and `assets/labels.txt` holds the class names (one per
line). Model input is (1, 3, 256, 256); the resize and mean/std in
`ai/AiAnalyzer.kt` must match the training pipeline.

The bundled `model.ptl` is the paper's highest-accuracy model — the full
S + C + EB configuration (style transfer + ROI crop + edge-blur, 95.5% on the
otoscope test set).

## Build

- Android Studio, or `./gradlew assembleDebug`.
- Kotlin + Jetpack Compose. minSdk 28, targetSdk 34, package `com.example.otoview`.
- Model dependency: `org.pytorch:pytorch_android_lite:1.13.1`.

## Layout

```
app/src/main/
├── java/com/example/otoview/
│   ├── MainActivity.kt        # main screen: stream + controls
│   ├── net/CommandClient.kt   # TCP command channel to the Pi
│   ├── stream/                # H.264 TCP receive + decode (VideoStreamManager, H264TcpDecoder, AnnexBReader)
│   ├── ai/AiAnalyzer.kt       # PyTorch Lite inference on captured frames
│   ├── gallery/               # saved-image gallery, viewer, probability list
│   ├── ui/                    # crop overlay, debug views
│   ├── util/                  # capture / save helpers
│   ├── bench/                 # optional latency/resource instrumentation
│   └── debug/                 # debug toggles
└── assets/                    # model.ptl, labels.txt
```

## Third-party

- PyTorch Mobile / `pytorch_android_lite` (BSD-3-Clause).
- `ui/theme/*` (Color.kt, Theme.kt, Type.kt) is the default Android Studio Jetpack Compose template.

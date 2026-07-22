# Otoscope Glass client

Google Glass client for the digital otoscope. It receives the H.264 video stream
from the Raspberry Pi otoscope server and shows it on the Glass Enterprise
Edition 2 display. Captured frames are classified on-device into 4 classes with
PyTorch Mobile. Control is by Glass touchpad gestures and by the Glass
voice-commands menu.

## System context

The Glass joins the Pi's Wi-Fi hotspot and reads the raw H.264 stream over TCP
(port 1234). The SPS is parsed from the stream to configure the decoder. The app
registers as a Glass app through the `com.google.android.glass.category.DIRECTORY`
intent category.

## Connecting to your Pi

The server address and ports are set in `IpAddress.kt` (`ipAddress`,
`port_stream`, `port_command`; defaults `10.42.0.1`, 1234, 4321). If your Pi uses
a different address or ports, change these to match the server configuration.

## Controls

Touchpad gestures (`GlassGestureDetector`):

| Gesture | Action |
|---|---|
| Swipe forward  | Capture the current frame and run on-device analysis |
| Swipe backward | Return to the live stream view |
| Swipe down     | Exit the app |

On this Glass EE2 the touchpad also reports gestures as key events, so `tap`
(DPAD_CENTER) and swipe forward/back (TAB / Shift+TAB) capture and return to the
video as well.

Voice commands (`res/menu/voice_commands_menu.xml`; needs the `RECORD_AUDIO`
permission, requested on first launch): "capture" / "analyze" / "scan" /
"take (a) picture" / "process" capture and analyze; "video" / "(go) back" /
"go (to) main" return to the live view; "light on" / "light off" toggle the
otoscope LED over the command socket; "quit" / "exit" / "terminate" close the
app. The captured frame is classified by the bundled model and the top class
with its probability is overlaid on the display.

## Layout

```
app/src/main/java/com/example/glasstcptest/
├── MainActivity.kt          # entry, wiring
├── VideoFragment.kt         # video surface + playback
├── StreamingView.kt         # display surface
├── TcpIpReader.kt           # TCP stream reader
├── SpsReader.kt / SpsParser.kt   # H.264 SPS extraction and parsing
├── ModelProcessor.kt        # on-device inference (PyTorch Lite)
├── GlassGestureDetector.kt  # Glass touchpad gestures
├── CameraData.kt, IpAddress.kt
└── (assets/) model.ptl, labels.txt
```

## On-device model

`app/src/main/assets/model.ptl` is the traced 4-class classifier (input
(1, 3, 256, 256); the resize and mean/std in `ModelProcessor.kt` must match the
training pipeline) and `assets/labels.txt` holds the class names. It is the
paper's highest-accuracy model — the full S + C + EB configuration (style
transfer + ROI crop + edge-blur, 95.5% on the otoscope test set).

## Build

- Android Studio, or `./gradlew assembleDebug`.
- Kotlin. minSdk 27, targetSdk 34, package `com.example.glasstcptest`.
- Model dependency: `org.pytorch:pytorch_android_lite:1.13.1`.

## Attribution

`GlassGestureDetector.kt` is derived from `GlassGestureDetector.java` in Google's
[glass-enterprise-samples](https://github.com/googlesamples/glass-enterprise-samples)
(GestureLibrarySample), Copyright 2019 Google LLC, licensed under the Apache
License 2.0 (http://www.apache.org/licenses/LICENSE-2.0). It was ported to Kotlin
and modified for this project; the original copyright and license notice are kept
in the file header. A copy of the Apache 2.0 license is included in this folder as
`LICENSE-Apache-2.0.txt`.

On-device inference uses PyTorch Mobile (`org.pytorch:pytorch_android_lite`),
which is distributed under the BSD-3-Clause license.

# Device software

Software for the digital otoscope hardware. The otoscope streams video from a
Raspberry Pi over Wi-Fi; a phone or Google Glass client displays the stream,
controls the device, captures frames, and runs an on-device classifier.

```
device_software/
├── raspberrypi_server/   # Pi camera server: H.264 stream + command/LED control
├── phone_otoview/        # Android phone client (stream, capture, gallery, on-device AI)
└── glass_otoscope/       # Google Glass client (stream view, gestures + voice commands, on-device AI)
```

## How the pieces connect

- The Pi runs a Wi-Fi hotspot and two TCP servers: video on port 1234 and commands on port 4321.
- A client joins the hotspot, reads the H.264 video stream, and sends LED / shutdown commands.
- The phone and Glass clients bundle the same `model.ptl` classifier for on-device inference.

Each component has its own README with build and run details.

## Setting your own values

The hotspot credentials and the server address/ports are specific to your setup.
Set the hotspot SSID/password in `raspberrypi_server/deploy/hotspot_on.sh`, and
make sure the clients use the same address and ports as
`raspberrypi_server/_03_config.json` (phone: `MainActivity.kt`; Glass:
`IpAddress.kt`). The defaults assume the Pi's NetworkManager hotspot at
`10.42.0.1`.

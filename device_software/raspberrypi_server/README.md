# Otoscope RPi Server

Raspberry Pi server for the digital otoscope. It streams H.264 video from the Pi
camera (picamera2) over TCP, and receives shutdown and LED-control commands on a
separate command port.

## How it works

On boot, `/etc/rc.local` runs `hotspot_on.sh` (brings up the Wi-Fi hotspot) and
then `init.sh` (starts the server), in that order. The server runs two processes
at the same time.

| Process | Port | Protocol | Role |
|---|---|---|---|
| `capture_stream` | 1234 | TCP | picamera2 H.264 stream (1280×720, 1 Mbps) |
| `command_listen` | 4321 | TCP | text commands (`command_quit`, `command_light_on`, `command_light_off`) |

- Video is encoded with picamera2's built-in `H264Encoder` (no ffmpeg).
- The stream is a raw H.264 elementary stream written straight to the TCP socket.
- The server does not send stills back; the client extracts frames from the incoming stream.
- The LED (GPIO 23, output) is driven only by the command port. There is no physical switch handling.

## Files

```
otoscope-rpi-server/
├── _01_main.py          # Entry point. Supervisor loop around main() (restart / exit handling).
├── _02_funcs.py         # Core. The two processes: capture_stream / command_listen.
├── _03_config.json      # Settings (IP, stream/command ports, crop / SAR options).
├── init.sh              # Start the server: sudo python3 _01_main.py _03_config.json
├── hotspot_on.sh        # Bring up the Wi-Fi hotspot (nmcli).
├── hotspot_off.sh       # Tear down the hotspot / restart Wi-Fi.
└── switch_boot_mode.sh  # Swap /etc/rc.local <-> rc.local.switch (toggle boot autostart).
```

> The `_01`/`_02`/`_03` prefixes mark run order and role. `_01_main.py` does
> `import _02_funcs` and `init.sh` refers to `_03_config.json` directly, so do
> not rename them.

## Configuration (`_03_config.json`)

| Key | Value | Description |
|---|---|---|
| `my_ip_address` | `10.42.0.1` | Server bind IP (hotspot gateway address) |
| `streaming_socket` | `1234` | H.264 stream port |
| `command_socket` | `4321` | Command port |
| `use_scaler_crop` | `true` | Use a centered sensor crop |
| `sar_correction` | `1.0` | Aspect-ratio (SAR) correction factor |

## Requirements

- Raspberry Pi OS (libcamera stack)
- Python 3
- `picamera2`, `RPi.GPIO` (see `requirements.txt`; usually installed via apt)
- `nmcli` (NetworkManager) for the hotspot scripts

## Running

```bash
# Manual start
sudo python3 _01_main.py _03_config.json

# or
./init.sh
```

For autostart on boot, configure `/etc/rc.local` to call `hotspot_on.sh` and
`init.sh`.

## Security note

`hotspot_on.sh` sets a throwaway default Wi-Fi hotspot password (`00000000`).
Change it to your own value before any real deployment. `_03_config.json` uses a
fixed hotspot gateway IP (`10.42.0.1`).

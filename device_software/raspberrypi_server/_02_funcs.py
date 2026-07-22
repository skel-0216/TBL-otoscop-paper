import json
from multiprocessing import Process, Value
import socket
import time  # for sleep
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import libcamera

import RPi.GPIO as GPIO
# GPIO Setting
led = 23    # pin 23, pgio 16


def gpio_init():
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(led, GPIO.OUT)


# running state for value_
RUNNING = 0
STOP = -1


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def main(path_json):
    print("************ start")

    try:
        print("try start")
        config = load_json(path_json)
        print("json loaded")
        a = Value('i', RUNNING)
        print("GPIO cleanup()")
        gpio_init()

        q1 = Process(target=capture_stream, args=(a, config))
        q2 = Process(target=command_listen, args=(a, config))

        q1.start()
        q2.start()

        q1.join()
        q2.join()
        
        print("try End")

        return 1

    except KeyboardInterrupt:
        return 2


def capture_stream(value_, config_):
    my_ip_address = config_["my_ip_address"]
    streaming_socket = config_["streaming_socket"]
    
    print("Stream server started on", my_ip_address, streaming_socket)

    picam2 = Picamera2()

    # configs
    video_config = picam2.create_video_configuration()
    video_config["size"] = (1280, 720)
    video_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
    picam2.configure(video_config)
    encoder = H264Encoder(1000000)

    # create the server socket once
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_stream:
        socket_stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_stream.bind((my_ip_address, streaming_socket))
        socket_stream.listen(5)
        
        while value_.value != STOP:
            print("Waiting for connection...")
            try:
                conn, addr = socket_stream.accept()
                print("Client connected:", addr)
                stream = conn.makefile("wb")

                # start streaming
                picam2.encoders = encoder
                encoder.output = FileOutput(stream)
                picam2.start_encoder(encoder)
                picam2.start()

                # keep the connection alive
                while value_.value != STOP:
                    try:
                        # detect whether the client dropped
                        data = conn.recv(1, socket.MSG_PEEK)
                        if not data:
                            break  # connection closed
                    except Exception:
                        break  # socket error -> treat as disconnect

                # stop streaming
                picam2.stop_encoder()
                conn.close()
                print("Client disconnected")

            except Exception as e:
                print("Accept or streaming error:", e)
                time.sleep(1)

    print("Stream server terminated")


def command_listen(value_, config):
    """Accept reconnections: loop on accept and handle EOF."""
    my_ip_address = config["my_ip_address"]
    button_socket = config["command_socket"]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_command:
        socket_command.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_command.bind((my_ip_address, button_socket))
        socket_command.listen(5)
        socket_command.settimeout(1.0)  # accept timeout

        print("Command server started on", my_ip_address, button_socket)

        while value_.value != STOP:
            try:
                try:
                    client_socket, address = socket_command.accept()
                except socket.timeout:
                    continue
                print("Command client connected:", address)

                with client_socket:
                    client_socket.settimeout(1.0)  # recv timeout
                    while value_.value != STOP:
                        try:
                            data = client_socket.recv(1024)
                        except socket.timeout:
                            continue
                        except OSError as e:
                            print("Command recv error:", e)
                            break

                        if not data:
                            print("Command client disconnected")
                            break

                        msg = data.decode(errors="ignore").strip()

                        if msg == "command_quit":
                            print("Socket received :", msg)
                            value_.value = STOP  # go to Exit state
                            # GPIO cleanup happens once, when main exits
                            break
                        
                        if msg == "command_light_on":
                            print("Socket received :", msg)
                            GPIO.output(led, GPIO.HIGH)
                        
                        if msg == "command_light_off":
                            print("Socket received :", msg)
                            GPIO.output(led, GPIO.LOW)

                        # if needed, multiple commands in one buffer could be split here
            except Exception as e:
                print("Command loop error:", e)
                time.sleep(0.5)

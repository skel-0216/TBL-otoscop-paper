#!/bin/bash

sudo nmcli device disconnect wlan0
sudo nmcli device uo wlan0
nmcli radio wifi off
nmcli radio wifi on

#!/bin/bash

sudo mv /etc/rc.local /etc/rc.local.temp
sudo mv /etc/rc.local.switch /etc/rc.local
sudo mv /etc/rc.local.temp /etc/rc.local.switch

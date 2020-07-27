#!/bin/bash

# generate X configuration file 
nvidia-xconfig  -a \
    --no-use-display-device  \
    --virtual=1920x1200 \
    --allow-empty-initial-configuration \
    -o /tmp/xorg.conf

# start xorg
nohup Xorg -noreset \
    +extension GLX  \
    +extension RANDR \
    +extension RENDER \
    -config /tmp/xorg.conf :0 > /var/log/xorg.log 2>&1 &

sleep 1

DISPLAY=:0 xhost +local:root

tail -f /var/log/xorg.log

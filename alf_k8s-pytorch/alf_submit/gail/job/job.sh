#!/bin/bash

# init envrionment, should always keep this
source /opt/set_env.sh

# set PYTHONPATH
export PYTHONPATH=$WORKING_PATH/h-baselines:$WORKING_PATH/h-baselines/SocialRobotCustom/python

# mujoco_py path
export PYTHONPATH=/opt/usr/local/lib/python3.6/dist-packages:${PYTHONPATH}
export MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt
export LD_LIBRARY_PATH=/opt/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

# do not need a xserver to render, just disable it
unset DISPLAY

# run training, headless
# sometimes the network is unstable so we customize timeout
cd $WORKING_PATH/h-baselines; pip install --default-timeout=1000 -e .; pip install --default-timeout=1000 -e ./SocialRobotCustom; source ~/.bashrc
xvfb-run python3 run_grid_search_multi.py $DIR_OUT

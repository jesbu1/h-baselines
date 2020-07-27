#!/bin/bash

# init envrionment, should always keep this
source /opt/set_env.sh

# set PYTHONPATH
export PYTHONPATH=$WORKING_PATH/alf:$WORKING_PATH/SocialRobot/python

# mujoco_py path
export PYTHONPATH=/opt/usr/local/lib/python3.6/dist-packages:${PYTHONPATH}
export MUJOCO_PY_MUJOCO_PATH=/opt/.mujoco/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=/opt/.mujoco/mjkey.txt
export LD_LIBRARY_PATH=/opt/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

# do not need a xserver to render, just disable it
unset DISPLAY

# run training, headless
# sometimes the network is unstable so we customize timeout
cd $WORKING_PATH/alf; pip install --default-timeout=100 -e ./alf/nest/cnest .
cd $WORKING_PATH/alf/alf/examples
xvfb-run python3 -m alf.bin.grid_search --root_dir=$DIR_OUT --gin_file=__gin_file__ --search_config=__search_conf__
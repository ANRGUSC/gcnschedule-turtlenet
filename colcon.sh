#!/bin/bash
#TO RUN ./colcom.sh (args)    || args = {run}
#Change the variables according the robot or packages
declare -a PACKAGENAMES=("gcnsched_demo" "interfaces")
LAUNCH_FILE="sched_launch.py"

PACKAGES=""
for val in ${PACKAGENAMES[@]}; do
   PACKAGES+=$val" "
done
echo $PACKAGES
colcon build --packages-select $PACKAGES --allow-overriding $PACKAGES && . install/setup.bash 
if [ $1 = "run" ]; then
    ros2 launch gcnsched_demo $LAUNCH_FILE
fi
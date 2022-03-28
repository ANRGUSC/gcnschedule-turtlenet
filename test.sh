#!/bin/bash

. ./install/setup.sh

EXECUTOR_NAME=node_1 ros2 run gcnsched_demo executor &
EXECUTOR_NAME=node_2 ros2 run gcnsched_demo executor &
EXECUTOR_NAME=node_3 ros2 run gcnsched_demo executor &

ros2 run gcnsched_demo scheduler &
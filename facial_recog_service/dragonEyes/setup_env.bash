#!/bin/sh
#for dlib
apt-get -y update  
apt-get -y install build-essential cmake  
apt-get -y install libopenblas-dev liblapack-dev  
apt-get -y install libx11-dev libgtk-3-dev  
apt-get -y install python python-dev python-pip  
apt-get -y install python3 python3-dev python3-pip


apt -y auto remove
#etc.
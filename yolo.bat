@echo off
cd "yolov5"
start cmd /k "py detect_3s.py --weights best.pt --source 0"

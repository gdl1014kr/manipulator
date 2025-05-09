https://medium.com/starschema-blog/offline-camera-calibration-in-ros-2-45e81df12555
https://qlalf-smithy.tistory.com/36
[check-108.pdf](https://github.com/user-attachments/files/20120276/check-108.pdf)

[calib.io_checker_200x150_8x10_15.pdf](https://github.com/user-attachments/files/20121511/calib.io_checker_200x150_8x10_15.pdf)



Ros2 camera_calibration package install
  
sudo apt update
sudo apt install ros-humble-camera-calibration

calibration - 15mm Checker Width, 8x10 board size

ros2 run camera_calibration cameracalibrator --size 7x9 --square 0.015 --ros-args -r image:=/camera/camera/color/image_raw


result

**** Calibrating ****
mono pinhole calibration...
D = [0.36577003492985377, -1.41852325849264, -0.013243046628133612, -0.005181817729925357, 0.0]
K = [906.913274619811, 0.0, 331.45828334365774, 0.0, 899.5014985892801, 176.29881514403618, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [932.297203925624, 0.0, 329.79868846790004, 0.0, 0.0, 919.437228995759, 172.92376180584446, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
640

height
480

[narrow_stereo]

camera matrix
906.913275 0.000000 331.458283
0.000000 899.501499 176.298815
0.000000 0.000000 1.000000

distortion
0.365770 -1.418523 -0.013243 -0.005182 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
932.297204 0.000000 329.798688 0.000000
0.000000 919.437229 172.923762 0.000000
0.000000 0.000000 1.000000 0.000000


https://qlalf-smithy.tistory.com/37

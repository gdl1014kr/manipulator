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
D = [2.2396880849473115, -34.79007140454074, 0.019669345494337416, 0.0064741798662663694, 0.0]
K = [1783.3659358707491, 0.0, 346.48051179180726, 0.0, 1894.167551837444, 226.9542708214539, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [1850.542872737183, 0.0, 348.02217466734265, 0.0, 0.0, 1965.2317096140916, 229.17928850493672, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
640

height
480

[narrow_stereo]

camera matrix
1783.365936 0.000000 346.480512
0.000000 1894.167552 226.954271
0.000000 0.000000 1.000000

distortion
2.239688 -34.790071 0.019669 0.006474 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
1850.542873 0.000000 348.022175 0.000000
0.000000 1965.231710 229.179289 0.000000
0.000000 0.000000 1.000000 0.000000

https://qlalf-smithy.tistory.com/37

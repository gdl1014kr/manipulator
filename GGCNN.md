좋습니다. 사용자의 설정 정보를 기반으로 RealSense D435 Camera에서 얻은 RGB + Aligned Depth 데이터를 활용해 마스크 기반 객체 중심 3D 좌표를 추출하고, GGCNN grasping 알고리즘에 연동하여 로봇 매니퓰레이터(Kinova Gen3 Lite)로 실시간 Pick-and-Place를 구현하기 위한 전체 grasping 파이프라인과 구현 방법을 조사하겠습니다.

특히 다음 항목을 포함하겠습니다:

* RealSense D435에서의 3D 포인트 추출 방식 (aligned depth 기반)
* 마스크와 depth를 기반으로 grasp 후보 추출하는 절차
* tf2 기반 camera-to-robot 변환 행렬 설정 및 적용 방식
* GGCNN의 입력 포맷 및 이를 위한 전처리 방식 (cropped depth + binary mask 등)
* ROS2 기반 GGCNN 연동 구현 사례 및 오픈소스
* Kinova 제어 연계 방법 및 grasp 포즈 변환 적용 사례

조사 결과를 바탕으로 전체 grasping 경로를 따라 개념→코드→ROS2 연동 흐름까지 설명하는 기술 문서를 작성해 드릴게요.


# Jetson AGX Orin 기반 RealSense D435 + NanoOWL/NanoSAM + GGCNN 파이프라인 구현

## RealSense D435에서 3D 좌표 추출 및 내부 파라미터 활용

RealSense D435는 깊이 카메라이므로 각 픽셀의 깊이 값을 카메라 내부(intrinsic) 파라미터와 함께 사용하여 3D 좌표로 변환할 수 있다. 예를 들어 Intel RealSense SDK의 `rs2_deproject_pixel_to_point` 함수는 깊이값(m)과 해당 픽셀 좌표, 카메라의 초점거리(fx, fy)와 중심점(cx, cy) 등의 intrinsics를 이용해 3D 점을 계산한다. 즉, 깊이 영상에서 관심 픽셀의 깊이(거리)를 가져오고 카메라의 intrinsic 행렬로 `(X, Y, Z)` 좌표를 얻는다. **librealsense** API를 사용하면 `pipeline.get_active_profile().get_stream(RS2_STREAM_DEPTH).as_video_stream_profile().get_intrinsics()`처럼 내부 파라미터를 얻어 `rs2_deproject_pixel_to_point`에 넘겨주면 된다. 또한 RGB 영상을 depth에 맞춰 정렬(align)하여 물체 마스크와 동일 좌표계를 맞춘 뒤, 해당 영역의 depth 값을 추출할 수 있다.

## 체커보드 기반 외부 캘리브레이션 및 tf2 변환 적용

카메라와 로봇 좌표계의 관계(외부 파라미터)를 구하기 위해 체커보드 패턴을 사용하여 캘리브레이션을 수행한다. 일반적으로 OpenCV `findChessboardCorners`와 `solvePnP`를 통해 월드 좌표의 체커보드 점과 이미지상의 픽셀 좌표를 이용해 카메라 위치와 자세(Rt)를 얻는다. ROS 환경에서는 **camera\_calibration** 또는 **industrial\_extrinsic\_cal** 패키지를 사용해 체커보드를 촬영하고 캘리브레이션 데이터를 만들 수 있다. 이렇게 얻은 카메라->체커보드(또는 로봇기저) 간 변환을 ROS2의 `tf2`로 브로드캐스트하면, 다양한 좌표계 간 변환이 가능해진다. 예를 들어 카메라 프레임에서 추정된 grasp 좌표를 로봇 베이스나 End-effector 프레임으로 변환하려면 `tf2_geometry_msgs::do_transform_pose` (혹은 Python의 `tf2_geometry_msgs.do_transform_pose`) 등을 사용한다. 캐리브레이션 결과로 `camera_link`와 `base_link` 사이의 고정된 트랜스폼이 설정되면, 이후 실시간으로 `tf2`를 통해 손쉽게 좌표 변환이 가능해진다.

## GGCNN 입력 형식 및 전처리

GGCNN은 깊이 이미지(단일 채널)를 입력으로 사용하며, 출력으로 각 픽셀별 grasp quality, grasp 각도, 그립 너비 맵을 생성한다. 네트워크는 학습 시 깊이 영상을 미터 단위로 정규화하여 사용하도록 설계되었으므로, 입력 깊이 영상도 \*\*카메라로부터 거리(미터)\*\*로 변환해야 한다. 일반적으로 입력 이미지는 고정 크기로 리사이즈하거나 크롭한다. 예를 들어 한 연구에서는 입력 크기를 **360×360** 픽셀로 설정했다. 객체를 찾은 후 해당 영역을 중심으로 깊이 영상을 자르고, 필요한 경우 패딩하여 정사각형 이미지를 만든 뒤 GGCNN에 입력할 수 있다. 이때 전처리로 배경을 제거하거나 필터링(가우시안 블러 등)하여 노이즈를 줄이고, 깊이 값의 단위를 네트워크 훈련 환경과 맞추는 것이 중요하다. 또한 Jetson 환경에서는 모델을 **TensorRT**로 최적화하고 FP16 모드로 실행하여 추론 속도를 높일 수 있다.

## NanoOWL/NanoSAM 마스크와 깊이 영상 결합 파이프라인

&#x20;아래 그림은 RGB-D 센서를 이용한 객체 검출과 grasp 예측 파이프라인의 예시이다. RGB 영상에서 객체(예: 오렌지)를 검출하면(bounding box) 해당 위치의 깊이 패치가 GGCNN에 입력되어 grasp를 예측한다. 본 과제에서는 YOLO 대신 텍스트 기반 검출 모델인 **NanoOWL**을 사용하여 개방형 어휘(object names)로 객체를 찾고, **NanoSAM**을 이용해 해당 객체의 인스턴스 마스크를 생성한다. 파이프라인은 다음과 같다.

* **1단계: 텍스트 기반 검출** – NanoOWL(OWL-ViT 최적화 버전)을 이용하여 사용자가 지정한 객체(예: “cup”)를 영상에서 실시간 검출한다. 이때 결과로 bounding box 혹은 객체 마스크(예: id\:object confidence)가 나온다.
* **2단계: 인스턴스 분할** – NanoSAM(경량화된 SAM)으로 검출된 객체의 정확한 영역을 추출한다. NanoOWL이 bbox를 주면 NanoSAM이 그 영역 안의 픽셀을 분할하여 마스크를 얻는다.
* **3단계: 깊이 정렬 및 크롭** – RealSense의 **align** 기능으로 color-깊이 축을 맞춘 뒤, 얻은 마스크를 깊이 영상에 적용하여 객체 영역만 남긴다. 이후 마스크의 최소 bounding box를 계산하여 깊이 이미지를 크롭/리사이즈한다.
* **4단계: GGCNN 입력 생성** – 크롭된 깊이 패치(단일 채널)를 네트워크 입력 크기에 맞게 재배치(예: 360×360)하고, 단위(m)로 정규화하여 GGCNN에 공급한다.

이처럼 NanoOWL/NanoSAM 기반 파이프라인은 YOLO-GGCNN 예시(그림)와 유사하게 작동하며, ROS2 노드로 구현하면 실시간으로 파이프라인을 처리할 수 있다.

## GGCNN 출력(Grasp) → 로봇 End-Effector 포즈 변환

GGCNN은 입력 깊이 이미지의 각 픽셀별로 grasp \*\*품질(quality map)\*\*과 **각도(orientation map)**, **그립 너비** 등을 출력한다. 최종적으로 픽셀 `(u,v)`와 대응하는 출력 맵의 최대값 위치를 선택하여 grasp 픽셀 좌표 `(u*,v*)`와 각도 θ를 얻는다. 이어서 해당 픽셀의 깊이 값을 이용해 3D 좌표 `(X,Y,Z)`를 계산한다. 예를 들어 `X = (u*-cx)*Z/fx`, `Y = (v*-cy)*Z/fy`, `Z = depth(u*,v*)`와 같이 단순 핀홀 카메라 모델을 적용한다. 이렇게 얻은 3D 좌표는 카메라 좌표계 기준이며, Θ는 카메라 축을 기준으로 한 회전 각이다. 이 정보를 바탕으로 로봇 기준의 End-effector 목표 포즈를 생성한다. 구체적으로, 예측된 `(X,Y,Z)`를 카메라-로봇 변환 (`tf2`)을 통해 로봇 베이스/EE 프레임으로 변환하고, grasp 각도 θ에 대응하도록 그립퍼의 요(Yaw)축을 회전시킨다. 예를 들어 카메라의 광축을 기준으로 상향 그립을 가정하면, 로봇 상에서 그립퍼가 카메라로부터 `(X,Y,Z)` 방향으로 내려가도록 포즈를 설정할 수 있다. ROS2에서는 `geometry_msgs::Pose` 메시지에 카메라 프레임의 포즈를 담아 `tf2::doTransform` 등을 통해 베이스/ee\_link 좌표로 변환할 수 있다. 이 과정으로 GGCNN의 픽셀 단위 예측이 로봇 동작으로 이어진다.

## ROS2 기반 GGCNN 구현 사례 및 코드

GGCNN은 여러 로봇 시스템에 통합된 예가 있다. 예를 들어 Douglas Morrison 등은 GGCNN을 Kinova Mico 로봇에 적용한 ROS 패키지를 공개했다. 해당 코드에서는 RealSense로부터 깊이 영상을 받고 GGCNN 네트워크를 돌려 grasp 맵을 계산한 뒤, Kinova Mico를 제어하는 워크플로를 담고 있다. 최신 ROS2 환경에서는 ChatterArm 프로젝트 등에서 LangSAM(텍스트→SAM)과 GGCNN을 통합해 사용한다. 이들 코드는 ROS2 노드(서비스/액션)를 이용하여, 객체명으로 지정된 영역(ROI)의 깊이를 GGCNN에 입력해 grasp 포즈를 얻고, MoveIt2를 통해 로봇 이동을 수행한다. 일반적으로 GGCNN 추론은 PyTorch 기반이므로 **TensorRT**로 엔진 변환 후 C++/Python 노드로 연동하거나, Isaac ROS DNN Inference 노드를 활용할 수 있다. 구현 시에는 CUDA 가속, 멀티스레딩 `rclcpp::MultiThreadedExecutor`, zero-copy 공유 메모리 등을 사용해 처리 지연을 최소화하는 것이 권장된다. (코드 예: `dougsm/ggcnn_kinova_grasping`, `llm-grasp-capstone-docs/ros2_ggcnn` 등 참조)

## 전체 파이프라인 통합(ROS2) 및 실시간 처리

최종 파이프라인은 ROS2 런치 파일이나 구성 노드로 엮는다. 예를 들어 `realsense2_camera` 노드가 RGB-D 토픽을 발행하고, NanoOWL/NanoSAM 노드가 검출·분할을 수행하며, GGCNN 노드가 그립 정보를 계산한다. tf2로 카메라와 로봇 프레임을 계속 브로드캐스트하고, GGCNN 결과를 MoveIt2의 목표 포즈로 전달하여 Manipulator를 제어한다. Jetson AGX Orin 환경에서 실시간 성능을 내기 위해 각 단계의 가속화가 필수이다. NanoOWL(OWL-ViT)은 TensorRT 최적화 후 Orin AGX 기준 약 **28 FPS**까지 처리 가능하며, NanoSAM(ResNet18 기반)은 **\~8.1ms**(≈120 FPS)로 실행된다. GGCNN 추론 역시 경량화된 모델로 실시간(수십 ms) 예측이 가능하다. 이외에도 입력 해상도 조절, 처리 스레드 분리, GPU Direct 카메라 입력, 불필요한 토픽 복사 회피 등으로 프레임률을 확보할 수 있다. 최적화된 전체 시스템에서는 실시간 피드백 제어가 가능하며, 동적 환경에서도 안정적인 pick-and-place 작업이 수행된다.

**참고자료:** Intel RealSense SDK Projection 문서, ROS 캘리브레이션 가이드, GGCNN 논문 및 구현, NVIDIA NanoOWL/NanoSAM GitHub, Kinova 및 GGCNN ROS 예제 등.

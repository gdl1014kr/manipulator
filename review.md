# Learning robust, real-time, reactive robotic grasping

## Abstract

 **GG-CNN(Generative Grasping Convolutional Neural Network)**: 
 
 deep learning을 통해 input된 depth image의 각 pixel(카메라로부터 해당 위치의 물체까지의 거리)에 대해 grasp 품질(Quality)(성공 확률), grasp 각도(Angle), grasp 너비(Width)를 동시에 예측하고 object-independent grasp 수행.
 (Cornell Grasping Dataset을 통해 label된 grasp rectangle을 pixel 단위의 map으로 변환하여 grasp 성공 확률이 높은 위치, 각도, 너비를 예측하도록 학습, 각도는 주기성을 고려하여 sine 및 cosine으로 표기)

 => 특징:
 - grasp 후보를 미리 정해두고 평가하는 것이 아니라, image 전체에 대한 grasp information을 한 번에 생성하는 방식
 - 가장 높은 품질의 pixel을 찾아 최적의 grasp 자세 결정. Static 환경 뿐만 아니라 낯선 물체 및 다양한 특성을 가진 물체, Dynamic 환경(주변 환경이 실시간으로 변하거나 물체가 움직이는 환경), 센서 노이즈 및 제어 오차, 물체가 무질서하게 밀집되어 있거나 가려진 clutter 환경에서도 높은 grasp 성공률(실시간)

 => 장점:
 - grasp 대상 물체가 움직이거나 주변 환경이 예측 불가능하게 변화하는 Dynamic 환경에서도 실시간 grasp 예측 가능.(경량화된 network 및 속도 향상(초당 50Hz으로 depth image 받아 실시간으로 gripper 위치 update)
 - 정확도 요구 사항 완화(Camera 및 로봇 간의 정밀한 Calibration 이나 로봇 position control의 정확도에 덜 의존적, 오차에도 작동)
 - grasp 동작 중에 센서로부터 실시간으로 주변 환경 정보를 받아들이고 이를 바탕으로 gripper의 움직임을 수정하여 goal grasping position 으로 전환(close-loop grasping)
 
 => 단점:
 - input 값이 depth image 이기 때문에 투명하거나 반사율이 높은 물체는 인식 성능 저하

 => 성능:
 - Static Object, Dynamic Object, clutter 환경에서 모두 높은 grasp 성공률

## Contribution

1. GG-CNN: pixel 단위의 실시간 grasp prediction
   
    기존 grasp method(Open-loop grasping)은 물체의 image or point cloud에서 grasp position 및 angle을 선정해 여러 개 sampling. 
각각에 대한 grasp quality 평가 및 순위 매겨 가장 좋은 후보 선택. -> 계산 시간 오래걸림.(실시간에 부적합) 
또한, 처음에 한 번 grasp 계획을 세우고 나면 고정된 goal position으로만 이동하기 때문에 Dynamic 환경 및 정확하지 않은 센서, 제어 값에서 사용 부적합.

    반면, GG-CNN(Generative Grasping Convolutional Neural Network)은 input된 depth image의 각 pixel에 대해 grasp 품질(quality), 각도(angle), 너비(width)를 동시에 예측하여, 전체 image에 대한 grasp 가능성을 실시간으로 평가함으로써 복잡한 후보 sampling 과정을 거치지 않고도, 빠른 추론이 가능(실시간)

2. 경량화된 network 구조로 실시간 closed-loop 구현
  - GG-CNN은 약 62,000개의 파라미터를 가진 경량화된 완전 합성곱 신경망(Fully Convolutional Network)으로, 약 19ms의 추론 시간
  - 최대 50Hz의 실시간 depth image를 받아 지속적으로 grasp position update & control(closed-loop) 가능.

3. 다양한 환경에서의 강인한 grasp 성능 입증
   
    GG-CNN은 Static Object, Dynamic Object, clutter한 환경에서 모두 높은 grasp 성공률

4. 다중 시점(multi-view) 기반의 grasp prediction으로 혼잡(clutter) 환경에서의 성능 향상
GG-CNN의 실시간 성능을 활용하여, multi-view에서의 grasp prediction.

    => 가려진 영역이나 복잡한 배치의 물체에 대한 grasp 성공률을 최대 10% 까지 향상

## Method

최적 grasp position(g):
- gripper가 물체를 잡기 위해 도달해야 하는 3차원 공간에서의 위치(x,y,z) => p
- z축을 중심으로 한 gripper 회전 각도(angle) => f
- gripper의 너비(width) => w     
=> g = (q, p , f, w)

1. Input & Output


    input: 300×300 pixel의 normalization된 depth image.

    output: 각 pixel에 대해 다음 세 가지 prediction :

    Grasp Quality Map (Q): grasp 성공 확률(품질, quality) (0~1 사이의 값).

    Grasp Angle Map (Φ): grasp angle (−π/2 ~ π/2 범위).

    Grasp Width Map (W): grasp width (pixel 단위).


2. Network Architecture

    구조: Fully Convolutional Network (FCN) 형태로, 약 62,000개의 파라미터를 가짐.

    속도: 전체 pipeline(전처리 포함) 추론 시간은 약 19ms로, 최대 50Hz의 실시간 제어 가능.


3. dataset 및 전처리

    dataset: Cornell Grasping Dataset을 사용하여 학습.

    전처리: 각 grasp rectangle을 중심으로 하는 mask를 생성하여 해당 영역의 Q, Φ, W 값을 설정.

    각도 Φ는 주기성을 고려하여 sin(2Φ)와 cos(2Φ)로 변환하여 학습.


4. grasp prediction 및 실행

    그립 선택: Q map에서 가장 높은 값을 갖는 pixel을 선택하여 해당 위치의 Φ와 W 값을 사용.

    좌표 변환: 선택된 pixel 좌표를 카메라와 로봇 간의 변환 행렬을 통해 월드 좌표로 변환.

    실행 방식:

    Open-loop: 단일 프레임에서 예측한대로 grasp 수행.

    Closed-loop: 실시간으로 dapth image를 받아 지속적으로 grasp position update & Control

## Conclusion

1. 성과:

- 경량화된 network로 인한 속도 향상: 다른 grasp network에 비해 크기가 작아 매우 빠른 계산 시간(최대 50Hz) 확보.
  이로 인한 closed-loop control 가능

- 실시간성 확보:
  1. depth image로부터 pixel 단위로 grasp pose 정보 생성하는 방식을 통해 기존의 grasp 후보들을 일일이 Sampling 및 Classification 방식의 한계 극복.
  2. closed-loop grasping을 통해 동적 환경, 복잡한 환경에서 robust. 각각 88%, 87%의 높은 grasp 성공률
 
2. 한계 및 해결 방안:

- 특정 재질(검정색, 반사되는 물체, 투명한 물체)에 대한 정확한 depth information 추출 어려움.
 
  => multi-view fusion method:
  로봇이 여러 다른 위치나 경로를 따라 이동하면서 여러 장의 depth image 촬영 -> 각 image에서 GG-CNN으로 생성된 grasp map information 수집 -> 수집된 information에서 각 격자 셀에 대해 여러 시점에서 관측된 grasp quality(품질), 각도(angle), 너비(width) 정보들의 평균을 계산하여 최종 grasp 후보 결정(grasp 성공률 최대 10% 향상)

여러 시점에서 얻은 정보를 조합함으로써, 한 시점에서는 가려져 보이지 않았던 좋은 grasp 지점 발견 가능.
다양한 각도에서 물체를 관찰하며 깊이 카메라의 측정 오류나 노이즈로 인한 부정확한 예측을 보완하고, 더 정확하고 신뢰할 수 있는 grasp information을 얻을 수 있음.

- gripper의 구조로 물리적인 한계 발생: 얇은 물체, 미끄러지는 물체, 깨지기 쉬운 물체 등 잡기 어려움.
  
  => 촉각 센서와 같은 다중 Sensor fusion.
  
- 물체들이 밀집되어 있는 clutter 환경에서 gripper가 주변 물체와 충돌 및 grasp 실패 현상 발생.

  => grasp 뿐만 아니라 pushing와 같은 다른 조작 action 학습.

# KGNv2: Separating Scale and Pose Prediction for Keypoint-based 6-DoF Grasp Synthesis on RGB-D input

KGNv2 - 6-DoF(위치 x,y,z 회전 roll,pitch, yaw) grasp method. GraspNet, KGN 개선 버전. grasp pose 예측 정확도 향상 목표. 
3차원 공간에서의 완전한 grasp pose 생성 목표

grasp pose 추정=> image 상의 keypoint(특정 특징점) 활용

input: RGB-D(2D RGB image + 3D Depth image) => image 공간에서 추출된 특정 지점을 나타내는 keypoint를 통해 grasp pose & 카메라로부터의 거리(카메라를 향하는 scale) - 카메라의 시점에서 파지하려는 물체까지의 3차원 공간 상의 절대적인 거리를 의미. 좀 더 정확히는, 파지 자세의 기준점(예: 논문에서 언급된 파지 중심)이 카메라 원점으로부터 3차원 공간 상에서 얼마나 떨어져 있는지를 나타내는 값
별도 추정
output: 파지 중심, keypoint 위치, scale 예측을 통한 6-DoF grasp pose 추정 & gripper open width

기존 방식: keypoint(2d)들의 상대적인 위치로부터 grasp pose(3d)의 scale( 6자유도(6-DoF) 파지 자세의 위치(translation) 벡터 크기, 
즉 카메라 원점부터 해당 파지 자세의 원점까지의 3차원 공간 거리)와 회전 동시 추정
=> keypoint 예측의 작은 오차에도 스케일 추정이 불안정 해지는 문제, image 공간에서 keypoint position을 얼마나 정확하게 예측하는지에 크게 의존하게 됨. 특히 keypoint prediction 시 발생하는 sensor noise가 Perspective-n-Point(PnP) algorithm을 사용한 3D grasp pose 추정에 악영향.


기존 방식과의 차이점: 4-dof의 평면 파지가 아닌 6-dof 파지 가능. point cloud 기반 파지 방식이 아닌 image 기반 keypoint 방식(RGB-D에서 2D keypoint 예측하여 찾음. -> 예측된 2D keypoint와 gripper에 미리 정의된 3D keypoint를 사용하여 pnp 알고리즘으로 카메라 좌표계 상에서의 3D grasp pose(위치 및 회전 정보) 추론. -> 카메라 좌표계를 로봇 좌표계로 변환-> 네트워크가 별도로 예측한 scale(카메라-파지 자세 간의 거리)을 회귀적으로 예측하여 pnp 추론 결과에 곱해 최종 위치 보정 수행 -> 최종적인 6-dof 파지 자세 결정, gripper open width 예측

=>but, KGNv2는 이를 별도의 네트워크로 분리하여 grasp pose 추정의 정확도를 높임. => Keypoint의 의존성 낮춤

+ 기존 방식과 달리 input data를 point cloud와 같은 명시적인 3D 표현으로 변환하는 과정을 거치지 않고 직접 파지 자세 추정
=> but, 2d image date에서 3d 공간의 자세 정보 추정 - 어려움

keypoint - image 또는 3d 모델(공간정보)에서 특정 지점을 나타내는 특징점. 특정 파지 자세에 있을 때 그리퍼의 주요 부분(또는 그리퍼에 정의된 가상점)이 카메라 이미지 상에 어디에 나타날지를 예측하는 지점.

introduction

- 6-dof pose method
1. point- cloud 기반 => 소규모 객체에 대한 파지 성능 좋지 않음, 센서 노이즈에 취약, 실시간에 적합하지 x(계산 시간 오래 걸림- 계산 비용 증가)
2. RGB-D 기반 => 소규모 객체 더 정확한 구별, 센서 노이즈에 대한 robustness(강건함), 빠른 처리 속도
RGB-D image를 input으로, 로봇이 물체를 안정적으로 잡을 수 있는 6-DoF grasp pose(3D 공간에서의 물체의 위치(x,y,z)와 방향(roll,pitch, yaw)와 gripper open width 찾음.

우리 프로젝트에는 rgb-d 기반이 좋음. => point cloud 방식은 point cloud에서 기하학적 정보를 추출하는데 처리 시간이 오래걸림. 
but, rgb-d 기반은 빠른 추론(실시간)에 적합.(처리 속도 빠름) 소규모 객체나 시각적 구별이 중요한 객체에 적합. 밝은 환경(실험실)에서 진행하기 때문.

-----------------------------------------------------------------------------------------------------
## Abstract

KGNv2는 RGB-D 이미지를 입력으로 받아 6-DoF(3D 위치와 3D 회전) grasp pose를 예측하는 네트워크로, 기존 KGN 대비 grasp 성공률이 약 5% 향상되는 등 성능이 크게 개선됨.  
RGB 컬러 정보는 깊이 센서 노이즈에 강건성을 제공하며, 합성 데이터만으로도 실제 환경에서 competitive한 sim-to-real 전이 능력을 입증.  
파지 자세 추정에서 스케일과 포즈 예측을 분리하는 새로운 전략과 키포인트 표현 방식 개선을 통해 PnP 알고리즘 기반 자세 복원의 안정성을 높임.

Abstract
KGNv2- KGN 개선 버전. Keypoint 기반 grasp pose(회전 및 방향) 추정. grasp pose의 scale(거리)은 별도의 네트워크를 통해 독립적으로 예측
keypoint - 객체나 로봇 그리퍼의 특정 지점(키포인트)을 이미지 상에서 먼저 찾고, 이 키포인트 정보를 바탕으로 3차원 파지 자세를 계산하는 방식
6-DoF - 3D 공간에서의 물체의 위치(x상

-한계: 실제 실험에서 단일 객체 파지 시 불안정한 자세 예측이나 네트워크의 부적절한 외삽으로 인한 가려진 영역 파지 시도 등 실패 사례가 발생했습니다. 
또한, 학습에 사용된 단순한 합성 객체들의 균일한 색상 특성이 실제 세계의 다양하고 복잡한 시각적 외형을 
인식하는 모델의 능력에 한계를 줄 수 있다고 언급합니다.

-향후 연구 방향: 향후 연구로는 생성 모델(diffusion model, TEXTure 등)을 활용하여 
학습 데이터셋에 실제와 유사한 다양한 질감을 추가하는 방안을 모색 가능

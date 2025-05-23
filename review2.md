# KGNv2: Separating Scale and Pose Prediction for Keypoint-based 6-DoF Grasp Synthesis on RGB-D input

기존 방식(KGN): 2D keypoint(input image를 통해 grasp 하려는 지점을 표시) 들의 상대적인 위치로부터 3D grasp pose와 scale(카메라 원점부터 해당 grasp pose의 원점까지의 3차원 공간 거리) 동시 추정

=> 
- keypoint 예측의 작은 오차에도 Scale 추정이 불안정.
- image 공간에서 keypoint position을 얼마나 정확하게 예측하는지에 크게 의존. 특히 keypoint 예측시 발생하는 sensor noise가 Perspective-n-Point(PnP) algorithm을 사용한 3D grasp pose 추정에 악영향.

KGNv2: 
(RGB-D에서 2D keypoint 예측하여 찾음. -> 예측된 2D keypoint와 gripper에 미리 정의된 3D keypoint를 사용하여 pnp Algorithm 적용. 카메라 내부 파라미터를 활용하여 카메라 좌표계 기준의 3D grasp pose(위치 및 회전 정보) 추론. -> 카메라 좌표계를 로봇 좌표계로 변환-> 네트워크가 별도로 예측한 scale(카메라-파지 자세 간의 거리)을 회귀적으로 예측하여 pnp 추론 결과에 곱해 최종 위치 보정 수행 -> 최종적인 6-dof 파지 자세 결정, gripper open width 예측

=> 
- KGNv2는 grasp pose와 Scale을 별도의 네트워크로 분리하여 grasp pose 추정의 정확도를 높임. => Keypoint의 의존성 낮춤
- RGB-D를 input data로 사용함으로써 depth image data의 sensor noise를 rgb로 보완.(robustness)

6-dof pose method
1. point- cloud 기반(GraspNet 등) => 소규모 객체에 대한 grasp 성능 저하, Sensor noise에 취약, 실시간에 적합하지 x(계산 시간 오래 걸림- 계산 비용 증가)
2. RGB-D 기반(KGNv2 등) => 소규모 객체 더 정확한 구별, Sensor noise에 대한 robustness(강건함), 빠른 처리 속도
RGB-D image를 input으로, 로봇이 물체를 안정적으로 잡을 수 있는 6-DoF grasp pose(3D 공간에서의 물체의 위치(x,y,z)와 방향(roll,pitch, yaw)와 gripper open width 찾음.

우리 프로젝트에는 RGB-D 기반이 적합. => point cloud 방식은 point cloud에서 기하학적 정보를 추출하는데 처리 시간이 오래걸림. 
RGB-D 기반은 처리 속도가 빨라 실시간에 적합. On-device에서 자연어 기반 실시간 object detection & grasp 하려는 contribution에 일치.  

=> contribution을 입증하기 위해 point-cloud 방식, RGB-D 방식 모두 사용하여 성능 비교해볼 예정.

-------------------------------------------------------------------------------
## Abstract

KGNv2- GraspNet, KGN 개선 버전. 6-DoF grasp method. Keypoint로 부터 grasp pose(회전 및 방향)의 scale(거리)은 별도의 네트워크를 통해 독립적으로 예측. 3차원 공간에서의 완전한 grasp pose 생성 목표, keypoint 표현 방식 개선을 통해 PnP Algorithm 기반 자세 복원의 안정성을 높임.

input: RGB-D(2D RGB image + 3D Depth image) => image 공간에서 추출된 특정 지점을 나타내는 keypoint를 통해 grasp pose & Scale(카메라의 시점에서 grasp 하려는 물체까지의 3D 공간 상의 절대적인 거리. 즉, grasp pose의 기준점이 카메라 원점으로부터 3차원 공간 상에서 얼마나 떨어져 있는지를 나타내는 값 별도 추정)
output: grasp 중심, keypoint 위치, scale 예측을 통한 6-DoF grasp pose(위치 x,y,z 회전 roll,pitch, yaw) 추정 & gripper open width
=> keypoint 검출을 통한 6-DoF grasp pose & gripper open width prediction(객체나 로봇 그리퍼의 특정 지점(keypoint)을 input된 image에서 먼저 찾고, keypoint 정보를 바탕으로 3D grasp pose를 계산하는 방식)

--------------------
## Method

- Scale-normalized keypoint 설계로 keypoint offset을 Scale로 나누어, noise가 Scale에 반비례해 감소하는 수학적 특성으로 자세 추정 오차를 줄임.  
- 별도의 네트워크를 통해 Scale 회귀 예측. 이를 PnP Algorithm으로 얻은 pose에 곱해 카메라 좌표계 상에서의 6-DoF grasp pose(이후 카메라 좌표계 -> 로봇 좌표계 변환), gripper open width를 예측.
- PnP 결과에 대한 noise 민감도를 줄이기 위해 Scale-normalized keypoint 설계 도입. => keypoint offset을 Scale로 나누어, noise의 영향이 거리에 따라 감소.

  => 최종적으로 예측된 6-DoF grasp pose와 gripper open width로 물체를 grasp. 
- 학습은 keypoint heatmap, offset, scale, open width 예측에 대한 손실 함수(focal loss, L1 regression loss 등)를 결합하여 수행.

## Contribution 
  
1. 네트워크 분리를 통한 Scale & grasp pose 별도 예측 => 기존 KGN 대비 grasp 성공률 약 5% 향상, Keypoint의 의존성 낮춤

기존 keypoint 기반 접근 방식의 한계(grasp pose의 scale과 회전/위치를 동시에 추정하며 발생하는 불안정성과 grasp 정확도 저하)를 극복하기 위해 grasp pose 추정 문제를 Scale과 자세 예측으로 분리.

2. Scale-normalized keypoint 설계

PnP Algorithm을 사용할 때 keypoint 예측 noise가 자세 추정에 미치는 악영향을 줄이기 위해 Scale-normalized keypoint 표현 방식 사용. Scale이 클수록 noise에 민감
keypoint 출력 공간을 추정된 Scale로 normalization하도록 재설계하여 keypoint 오류에 대한 민감성을 줄이고 추정된 pose의 정밀도를 향상. 

3. 단순 합성 데이터만으로도 실제 환경에 일반화 가능한 sim-to-real 성능을 입증

4. 기존 Point cloud 방식을 사용하지 않고 RGB-D image로부터 직접 6-DoF grasp pose와 gripper open width를 예측해 효율성과 계산 속도 개선

## Conclusion

KGNv2: 
- 개선점 및 효과 : grasp pose, Scale을 별도의 네트워크로 분리하고 PnP Algorithm에 대한 분석을 바탕으로 Scale-normalized keypoint 디자인 도입 => grasp pose 추정 정확도 높임.
- 검증: 합성 데이터셋을 사용한 실험에서 KGN2가 기존 KGN 방식보다 label로부터 grasp 분포 더 잘 학습. sim-to real 검증, 높은 grasp 성공률
- 한계: 실제 실험에서 단일 객체 파지 시 불안정한 자세 예측이나 네트워크의 부적절한 외삽으로 인한 가려진 영역 파지 시도 등 실패 사례 발생.이는 특정 객체의 형태, 미끄러운 표면, 또는 복잡한 시각적 환경에 대한 네트워크의 한계를 시사.
-향후 연구 방향: 향후 연구로는 생성 모델(diffusion model, TEXTure 등)을 활용하여 학습 데이터셋에 실제와 유사한 다양한 질감을 추가하는 방안을 모색.
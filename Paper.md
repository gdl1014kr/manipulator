좋습니다. 영어 논문 중심으로 아래 세 가지 주제에 해당하는 최신 연구들을 모두 조사하겠습니다:

1. NanoOWL + NanoSAM 조합을 사용한 논문
2. Depth Image 또는 Depth PointCloud를 Upsampling하여 Grasping에 활용한 논문
3. Jetson AGX Orin 등 Edge Device에서 Grasping을 수행한 논문

가능한 최신(2020년 이후) 학회/저널 기반 논문을 중심으로 정리하고, 각 논문에 대해 제목, 연도, 사용 기술, 실험 플랫폼, 성능 지표 및 핵심 내용을 분석하여 보고서 형식으로 제공하겠습니다. 잠시만 기다려주세요.


# NanoOWL과 NanoSAM 기반 연구

* **“Haptics-based, higher-order Sensory Substitution …: Virtual Whiskers” (Feng 외, 2024, *Disability and Rehabilitation: Assistive Technology*)** – 초시각 장애인용 전자 지팡이 시스템을 제안. NVIDIA Jetson Orin NX와 RGB 카메라를 사용하며, NanoOWL(객체 탐지)과 NanoSAM(영상 분할)을 조합하여 바닥과 장애물을 식별. “오픈 패스 모드”에서는 NanoOWL이 바닥을 감지하고 NanoSAM으로 도보 가능 영역을 분할하여 햅틱 벨트로 전달. “딥스 모드”는 MiDaS 단일 뷰 심도 추정으로 장애물 거리를 측정해 햅틱 피드백을 제공. 10명의 시각장애인 실험에서 캐닝(지팡이) 단독 사용 대비 두 모드 모두 경로 탐색 시간 증가(인지 부하 상승)가 관찰되었지만, **오픈 패스 모드**는 경로 탐색 시간과 지팡이 접촉 횟수를 유의미하게 줄이고 안전 거리를 높여 주행 안정성을 개선하는 것으로 나타났다. 다만 시스템 학습 곡선 및 인지 부하 문제로 숙련도에 따른 성능 편차가 한계로 지적됨.

# 깊이 영상/포인트클라우드 업샘플링 연구

* **“ClueDepth Grasp: Leveraging positional clues of depth…” (Hong 외, 2022, *Frontiers in Neurorobotics*)** – 투명 물체의 심도(depth) 결측을 보완하는 딥러닝 기반 기법(CDGrasp)을 제시. DenseFormer + U-Net 등 멀티모달 네트워크를 통해 투명 표면의 반사∙굴절로 인한 오류 점(depth)을 제거하고 정밀한 심도 맵을 복원. ClearGrasp 데이터셋 실험에서 기존 기법 대비 깊이 보완 정확도에서 최고 성능을 달성했으며, Baxter 휴머노이드 로봇으로 8종의 실제 투명 객체를 총 80회 시험(각 객체 10회)한 결과 높은 안정성으로 성공적 그립을 확보함으로써 제안 방법의 유효성을 입증했다. 특히 실제 투명장면의 深度 예측에서 δ1.05 정확도를 5.47% 향상시키며 성능 개선을 보였다. 제한점으로는 복잡한 네트워크로 인한 계산량과, 특정 환경(정형된 테이블 배경)에서의 실험으로 일반화 검증이 필요하다.
* **“Rethinking scene representation: A saliency-driven hierarchical multi-scale resampling…” (Yi 외, 2024, *Expert Systems with Applications*)** – RGB-D 포인트클라우드에서 객체 중심 특징과 살리언시 정보를 활용한 계층적 재샘플링 기법(SHSMR)을 제안. 장면을 균질하지 않은 분포에서도 주요 피처를 보존하는 3D 재구성으로 변환하며, 로봇 조작에서 중요 객체를 인식·추적한다. 제안 방법은 실험상 피처 지속성(feature persistence)을 높이고 불균형 포인트 분포에 강인함을 보여주었으며, 실제 로봇 실험에서 **일반 객체 및 산업 부품 그리핑** 작업 시 더 정확한 그립 추정 결과를 나타냈다. 즉, 종래의 균일 샘플링 대비 잡힌 개체의 특징을 고려한 샘플링이 그립 성능을 개선한다. 한계로는 추가적인 샘플링 단계에 따른 계산 비용 증가 가능성이 있으며, 다양한 환경·물체에 대한 범용성 추가 검증이 필요하다.

# 엣지 디바이스 기반 실시간 그립 예측

* **“A YOLO-GGCNN based grasping framework…” (Jhang 외, 2023, *Expert Systems with Applications*)** – SLAM 기반 맵핑 후 모바일 암이 움직이는 환경에서 **YOLOv4 + GGCNN**을 이중 스테이지로 사용하는 그립 추정 프레임워크. Nano 기반 Jetson 언급은 없으나, 시스템 구현을 위해 ROS와 모바일 로봇(TIAGo++)을 활용. 제안 기법은 단일 프레임당 0.11초(≈9.1 FPS)의 추론 속도를 보였고, \*\*캡처 정확도(capture accuracy)\*\*는 86.0%를 달성하여 기존 방법 대비 안정적 그립 성능을 나타냈다. 다양한 실험 환경에서 목표 객체를 선택적으로 정확히 잡아내는 것이 확인되었으며, 다만 시스템은 약 9 FPS로 엣지 임베디드 사용에 적합한 반면 비교적 고성능 하드웨어도 요구된다.
* **“Efficient End-to-End 6-DoF Grasp Detection … for Edge Devices” (Huo 외, 2024, *arXiv*)** – RGB-D 입력으로 6자유도 그립(병렬 집게) 후보를 고속 검출하는 경량 네트워크(E3GNet) 설계. **NVIDIA Jetson TX2/Xavier NX**에서 평가되었으며, 대표적인 PointNet 기반 HGGD 대비 훨씬 빠른 연산 속도를 보였다. 예를 들어 Jetson TX2에서 E3GNet의 추론시간은 약 157.9 ms(약 6.3 FPS)로, HGGD(649.4 ms)보다 약 4배 빨랐다. 22개 다양한 물체로 구성된 6개 혼합 장면에서 실험한 결과, E3GNet은 그립 성공률 94%(51회 시도 중 48회 성공)를 기록해, HGGD(80%)보다 크게 우수했다. 제한점은 정확도와 속도를 동시에 개선하였으나 여전히 일부 복잡 장면에서는 그립 안정성 감소 여지가 있다는 점이다.
* **“Fast GraspNeXt: … for Robotic Grasping on the Edge” (Wong 외, 2023, *arXiv*)** – 다양한 시각 과제를 동시에 수행하는 멀티태스크 네트워크를 제안. MetaGraspNet 벤치마크에서 박스ㆍ마스크 검출과 흡착 그립 히트맵 예측 등을 동시에 수행하며 우수한 성능을 보였다. 17.8M 파라미터로 경량화하여 Jetson TX2(8GB)에서 평가한 결과, 다른 효율적 백본 대비 최대 3.15배 빠른 연산을 달성했다. (예: Fast-GraspNeXt의 TX2 추론 시간은 ≈1106 ms로 0.9 FPS, ResNet 기반 대비 3배 이상 빠름). 이처럼 임베디드 환경에서 실시간 대응력 제고에 기여했으나, 여전히 1 FPS 이하 속도로 동작하여 진정한 실시간 적용에는 추가 최적화가 필요하다.

**출처:** 각 논문 및 학회/저널 발표자료 등.


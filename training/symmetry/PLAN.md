# v4 대칭 키포인트 추가 학습 계획

현재 v4 가중치에서 출발해 대칭 손실을 적용한 모델과 번호 고정 비교군을 학습한다.
전면은 raw 키포인트 번호로 결정하지 않고, 추론 뒤 PnP와 투영 면적 비교로 선택한다.

| 구분 | 설정 |
|---|---|
| 데이터 | 제공된 두 ZIP, 총 245장 |
| 분할 | 녹화 영상 단위로 학습 162 / 검증 38 / 최종 평가 45 |
| 대칭 | 정사각형 3D 박스의 수직축 0/90/180/270도 회전(C4) |
| 점 가중치 | 직접 클릭 1.0 / PnP·중심 생성점 0.25 |
| 학습 | AdamW, 학습률 0.0003, 배치 8, 입력 640, 최대 40에폭 |
| 조기 종료 | 검증 fitness가 12에폭 연속 개선되지 않을 때 |
| 최종 모델 선택 | 검증 검출률 우선, 같으면 직접 클릭한 점의 평균 픽셀 오차 최소 |
| 비교 결과 | 기존 v4 / C4 추가 학습 / 번호 고정 추가 학습 |
| 후처리 | 기존 largest_projected_vertical_face 재사용, C4 번호 변경 불변성 테스트 |
| 실행 모델 반영 | 평가용 모델을 별도 저장하고 현재 FSM 기본 모델은 자동 교체하지 않음 |

현재 장비의 GPU에 전력·온도 제한이 확인되어 실제 본 학습은 더 빠른 CPU로 진행한다.
라벨과 녹화 메타데이터의 카메라 내부 파라미터가 다르므로 `DATA_QUALITY.md`를 함께 참고한다.
아래는 재현을 위한 상세 실험 조건이다.

Date: 2026-09-05. Source weights: `extracted/pallet_yolo26n_pose_livegt_v4.pt`.

1. Audit both ZIP datasets without modifying the originals. Validate image/annotation dimensions,
   9-point topology, visibility/source fields, finite coordinates, duplicate image hashes and physical dimensions.
2. Convert JSON annotations to YOLO pose labels. Keep manually clicked coordinates unchanged.
   Encode manual points as v=2, PnP/centroid targets as v=1 and unavailable/out-of-frame points as v=0.
   Use the clipped envelope of all eight cuboid corners as the detection box.
3. Split by recording (not adjacent frames): train 162, validation 38, test 45 before audit exclusions.
   Validation: 20260903_190743, 20260904_142318, 20260904_142958.
   Test: 20260903_192254, 20260904_144614, 20260904_150335, 20260904_150944.
   All other recordings train. These are held out from this fine-tune only; original v4 exposure is unknown.
4. Use C4 rotations about the vertical Y axis of the 1.1 x 0.15 x 1.1 cuboid.
   This is a geometric box equivalence, not a claim that every physical pallet face has identical appearance.
   Select ONE complete permutation per positive prediction using the gain-weighted coordinate,
   keypoint-presence and RLE losses; never select independently per corner.
   Weight manual coordinates 1.0 and generated coordinates 0.25. Keep existence targets v>0,
   consistent with runtime use of the keypoint confidence; do not reinterpret confidence as physical visibility.
5. Run a GPU/gradient smoke test, then fine-tune from v4 with AdamW, lr0=0.0003,
   batch=8, imgsz=640, up to 40 epochs, patience=12, seed=42, cosine schedule.
   No flip, mosaic, perspective or rotation augmentation initially; mild scale/translation/color augmentation.
   Retain end-to-end one-to-many and one-to-one branches and RLE training.
   Set warmup bias LR to 0.0003 (not the stock SGD-oriented 0.1).
   Hardware audit: local GPU was restricted to 210 MHz / 405 MHz memory with a 10W
   power cap and software thermal throttling. Batch-8 GPU training ~9.1 seconds;
   CPU forward/backward benchmark with four threads ~2.38 seconds. CPU is an available
   fallback while this external hardware restriction persists; no system power settings changed.
6. Use symmetry-aware validation OKS/mAP for best checkpoint/early stopping. Compare original v4,
   symmetry fine-tune and (resources permitting) an otherwise identical fixed-index fine-tune control.
   Final held-out metrics: matched detections, manual-point pixel error/PCK, generated-point error,
   group-level results; supplement with saved raw-keypoint overlays and video stability inspection.
7. Export the selected model as a standard Ultralytics PoseModel checkpoint loadable without custom classes.
   Runtime already chooses the largest camera-facing vertical face AFTER PnP in
   `depth_cam/calib/geometry.py::largest_projected_vertical_face`.
   Do not automatically replace the live FSM model until the experiment is evaluated.

Limitations: only 245 positive frames, 13 recordings, four manually clicked points per image;
remaining points are derived rather than independent 3D ground truth. No background-only labels.
Absolute signed axis is unconfirmed in the JSON, compatible with the intended symmetry-invariant target.
Do not use PnP-derived pose error as independent physical ground truth.

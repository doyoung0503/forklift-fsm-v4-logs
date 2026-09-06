# 최종 선택: Cleanlabel + 정지시간/선형 endpoint 회귀

사용자 결정에 따라 48개 측정점으로 적합한 함수를 FSM 기본 회전 보정 함수로 선택했다.
앞선 선택 15개 영상의 31점 비교 함수가 아니라 전체 품질 통과 48점 함수다.

`theta(T) = 12.10246966536945 * max(T - 1.276363332869698, 0)`

양의 목표 회전량 `theta`의 명령시간은 `T = 1.276363332869698 + theta / 12.10246966536945`.
좌우 통합, joystick deflection 30, 단위 초/도. 역함수의 큰 목표는 기존 단일 명령 상한 2.5초에서 잘라 정지·재관측·재계획한다. 아주 작은 목표는 기존 최소 명령각 조건으로 차단한다.

## 시작

작업공간 루트에서 평소와 같이 실행한다. 이 명령은 실제 FSM 실행 명령이므로 장비 준비 후 사용한다.

```powershell
python extracted/depth_cam/main_rec_v4.py
```

- 가중치: `extracted/pallet_yolo26n_pose_cleanlabel.pt` (SHA256 `4ee578e02810caae56b786c23a084121f4dfb33941e11d4d3fb216b9bfbbe60e`). 기존 가중치 파일은 삭제하지 않았다.
- 활성 함수: `extracted/depth_cam/calib/fsm_v4/rotation_endpoint.selected.json`.
- 모델/함수 해시와 라이브 추론 계약을 시작 시 확인한다. 불일치·누락 시 구형 물리 함수로 대체하지 않고 시작을 차단한다.
- 라이브 추론은 배치 평가와 동일하게 100px 반사 패딩, imgsz 640, conf 0.4, 최고 box 신뢰도 선택, 좌표 복원 후 PnP를 사용한다. 공통 perception 설정이므로 다른 FSM 실행기에도 이 모델/전처리가 적용된다.
- 함수에 관성이 이미 포함되어 있어 예전 관성 보정 1.07도, 구형 실시간 coast 예측, 적응 시간 배율을 중복 적용하지 않는다.
- 계획한 유지시간 또는 실제 목표각 통과 시 STOP. 시야/입력 유효성/타임아웃 등 기존 안전 정지와 3초 정지 대기 후 안정성 검사를 유지한다.
- heading↔bearing 변환에는 현재 실측 config의 회전축 오프셋을 그대로 사용한다. 실측값은 변경하지 않았다.
- 기존 `--calibrate`는 다른 물리 모델 적합 파이프라인이므로 최종 선택 함수를 덮어쓰지 않도록 차단한다. 모델을 바꾸려면 새로 추론·검증하고 대응하는 endpoint 함수를 명시적으로 선택해야 한다.

## 해석과 검증

RMSE 1.694도, R² 0.889, 녹화 동일가중 교차검증 RMSE 1.982도. 실제 각도 센서 정답에 대한 오차가 아니라 영상 추정에 대한 일치도다.
지연 1.2764초는 유효 절편이며 실제 운동 시작시간과 동일하다고 보장하지 않는다. endpoint만으로 STOP 직전각·속도·관성을 분해하지 않으며 해당 로그 필드는 null로 표시한다.
관측 시간 범위는 약 0.093–2.531초이며 범위 밖 일반화는 검증되지 않았다. 장비를 실제로 실행하여 검증한 것은 아니다.

[48개 기준점 그래프](out/cleanlabel_command_clips_20260906/final_selected/reference_points_48.png) · [기준점 CSV](out/cleanlabel_command_clips_20260906/final_selected/reference_points_48.csv)

오프라인 소프트웨어 검증:

```powershell
python -m unittest discover -s extracted/depth_cam/tests -p 'test_*.py'
python -m rotation_fit.verify_final_endpoint_runtime
```

두 번째 명령은 저장된 영상의 첫 프레임 23장을 사용해 배치/라이브 키포인트 일치를 검사한다. 카메라/CAN을 열지 않는다.

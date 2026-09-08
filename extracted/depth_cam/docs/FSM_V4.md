# FSM v4: 시각 기반 정지-관측 Macro-Action 제어

## 실행과 설정

- 실행: `cd depth_cam && python3 main_rec_v4.py`
- 단일 설정 파일: `calib/fsm_v4/config.py`
- 모니터 테스트 기본값은 `CAN_ENABLED=False`이다. 이 모드에서는 Kvaser
  라이브러리/장치 없이 카메라·추론·FSM·녹화를 실행하고 실제 CAN 프레임은
  한 건도 송신하지 않는다. 실차 구동 때만 명시적으로 `True`로 바꾼다.
- 자동 삽입은 외부파라미터가 임시값인 동안 기본적으로 비활성화되어 있다.
  `AUTO_INSERT_ENABLED=False`이면 최종 정렬 후 `READY_TO_INSERT`에서 STOP을 유지한다.

설정 파일을 수정한 뒤 프로세스를 다시 시작하면 CAN joystick template,
운동 모델, planner, 허용오차와 timeout에 새 값이 함께 적용된다. v1-v3의
설정은 수정하지 않는다.

## 외부파라미터

| 변수 | 의미 |
|---|---|
| `CAMERA_TO_ROT_CENTER_X_M` | 카메라에서 차량 회전중심까지 X, 우측 양수 |
| `CAMERA_TO_ROT_CENTER_Z_M` | 카메라에서 차량 회전중심까지 Z, 전방 양수 |
| `CAMERA_YAW_IN_VEHICLE_DEG` | 카메라 광축의 차량 전방축 대비 yaw, 우측 양수 |
| `CAMERA_HORIZONTAL_FOV_DEG` | 활성 카메라 스트림의 수평 시야각 |
| `ROT_CENTER_TO_FORK_TIP_M` | 두 Z 실측값으로 계산되는 회전중심-포크 끝 길이 |
| `FORK_TIP_CLEARANCE_M` | 삽입 전 포크 끝 여유거리 |
| `CAMERA_TO_FORK_TIP_X_M` | 카메라에서 포크 중심까지 X |
| `CAMERA_TO_FORK_TIP_Z_M` | 카메라에서 포크 끝까지 Z |

수치의 단일 기준은 `calib/fsm_v4/config.py`이다. 문서에는 값을 복사하지 않는다.
확정된 Z 실측값에서 `ROT_CENTER_TO_FORK_TIP_M`, staging 거리와 회전 bearing
offset을 계산하므로 파생 길이를 별도로 수정하지 않는다. 자동 삽입을 켜려면
`EXTRINSICS_MEASURED=True`여야 한다.
직진시 한쪽으로 흐르는 평균 경로각은 `FORWARD_PATH_BIAS_DEG`에 기록하며,
planner가 각 macro-step의 목표 방위를 반대 방향으로 선보상한다.

## 조정 영역

`config.py`는 다음 여덟 영역으로 나뉜다.

1. 카메라·회전중심·포크 외부파라미터
2. 회전/전진/후진 joystick deflection
3. 회전 startup, 시각 각속도, 선제 STOP, 최대 회전시간
4. 전진거리 적합식, 신뢰 가능한 최소거리, macro-step, 적합시간 이후
   timeout 여유 및 후진 임시 모델
5. PnP filter, frame age, 지연시간과 이상치 gate
6. SEARCH, 정지 안정화 및 화면 가시성
7. 3 m 안전거리, staging pose와 최종 허용오차
8. 자동 삽입 활성화와 분할 삽입 거리

## 자동 로그 기반 회전 응답

`main_rec_v4.py`는 카메라/CAN 초기화 전에 현재 `.pt` 모델의 SHA-256에 맞는 회전
artifact를 확인한다. 없거나 모델이 바뀌었으면 raw 영상 재추론, 로그 검증,
`yaw_deg`/`heading` 적합, recording 단위 LOO 검증과 원자 배포를 자동 수행한다.
실패하면 기존 artifact를 보존하고 FSM을 시작하지 않는다.

응답은 조이스틱 deflection 30에서 좌우를 하나의 크기 모델로 합쳐 다음 네 구간을
표현한다.

- 명령 write-return부터 실제 움직임까지의 정지시간
- 움직임 시작 뒤 등가속 구간
- 최대 각속도의 등속 구간
- STOP 직전 각속도와 약 2초 후 회전량의 선형 관성 함수

artifact는 현재 Pose 모델 hash, 실제 CAN 시각, 좌우 부호, 식별 가능성과 LOO RMSE가
모두 통과할 때만 활성화된다. 적합 도메인은 차량 `heading`이며, FACE/RECENTER의
`bearing`에는 현재 거리와 실측 카메라-회전중심 오프셋 0.68 m를 사용한 gain을
적용한다. 새 관성 함수가 활성화되면 기존 감속/coast 보정은 다시 더하지 않는다.

## 단계별 파이프라인

1. `PRECHECK`: 설정값 일관성을 검사하고 STOP을 보낸다.
2. `SEARCH_SWEEP`: 오른쪽 제자리 회전만 유지한다. 첫 탐지에서 즉시 STOP하며 30초 동안 탐지되지 않으면 STOP 후 `FAILED`로 종료한다.
3. `ACQUIRE_VERIFY`: 정지 상태의 연속 PnP를 모아 동일 pose가 안정적으로 유지되는지 확인한다.
4. `FACE_ROTATE/SETTLE`: PnP 전면 평면 중심이 카메라 광축을 향하도록 회전한다. 로그 구간속도/실시간 PnP 예측 또는 전면 중심의 실시간 오차 `+/-0.5 deg` 중 먼저 만족한 조건으로 STOP한다.
5. `STANDOFF_*`: 전진·후진 공통 거리-시간 적합식을 사용한다. 원시 PnP camera-Z가 `3.0 +/- 0.1 m`에 들어오거나 필터 pose의 예상 정지점이 밴드에 도달하면 즉시 STOP하고 안정화 후 다시 검증한다.
6. `STAGING_PLAN`: 팔레트 좌표계에서 차량 회전중심의 목표 staging pose를 계산한다.
7. `WAYPOINT_TURN`: hard cap은 20도이며 팔레트 모서리와 8% 화면 여유를 이용해 전면부가 계속 보이는 실제 허용각까지만 실행한다.
8. `WAYPOINT_DRIVE`: 전진 적합식이 신뢰되는 거리 범위에서 한 번의 macro-step을 실행하고 PnP 이동량으로 선제 STOP한다.
9. `WAYPOINT_*_SETTLE`: 관성 이동이 끝난 안정 pose를 얻은 뒤 진행량을 평가하고 6단계부터 재계획한다.
10. `FINAL_POSE_LOCK`: pallet-frame 위치, lateral, 전면 yaw와 거리 조건을 동시에 검증한다.
정지 관측 재사용: SETTLE에서 승인한 pose/중앙각/시야 여유는 이어지는 `STANDOFF_VERIFY`, `STAGING_PLAN`, `FINAL_POSE_LOCK`에 전달한다. 승인 관측과 현재 관측 모두 `MAX_MEASUREMENT_AGE_SEC`(현재 0.30초) 이내이며, 현재 자세와 중앙각이 기존 안정성 허용오차 안이고 시야 여유가 감소하지 않았을 때 10개 표본을 다시 모으지 않는다. 원래 측정 시각은 갱신하지 않는다. 새 동작, 관측 누락, 복구, 리셋, 디버그 재개 또는 유효기간 경과 시 재사용을 해제하고 새 표본을 수집한다. 사용 여부는 HUD의 `[OBSERVE REUSE]`로 확인한다. 제동 대기와 최초 안정성 확인은 유지한다.

11. `READY_TO_INSERT`: 삽입 승인 시 확정한 카메라 Z에서 `INSERT_CAMERA_Z_REMAINDER_M`를 뺀 거리를 유지한다. 이 상태부터 추론과 PnP 검사를 중단한다. 자동 삽입이 꺼져 있으면 STOP 상태로 대기한다.
12. `INSERT_DRIVE/SETTLE`: 확정 거리를 `forward_seconds()`로 변환한 시간만큼 한 번 전진하고 STOP 후 `STOP_MIN_SETTLE_SEC`를 기다려 `DONE`으로 전환한다. PnP 복구/자세/정지 안정성 검사를 하지 않으며 잔여 거리 재시도도 없다. 이동 거리는 실측하지 않으므로 화면에는 삽입 시간 진행률을 표시한다. 명령 시간 한계와 전체 삽입 타임아웃은 유지하며, 실행 루프는 새 카메라 프레임 없이도 타이머를 처리한다.
13. `DONE/FAILED`: 명시적인 종료 상태에서 STOP을 유지한다.

## 제어 원칙

- IMU는 endpoint 결정에 사용하지 않는다.
- 움직임이 확인되기 전에 짧은 회전 명령을 종료하지 않는다.
- 명령 중 관측값이 목표에 도착할 때가 아니라, 예상 정지 pose가 목표에
  도착할 때 STOP한다.
- FACE 전면 중심 및 STANDOFF 원시 3 m 관측은 예측식보다 우선하는 즉시
  STOP 조건이다.
- 방향을 즉시 반전하지 않고 항상 STOP/settle/verify를 거친다.
- 가시성, 누적거리, 보정횟수, 무진전, 각 동작시간과 전체시간을 제한한다.
- 원시 PnP와 host monotonic 프레임 수신시각을 v4 filter에 전달한다.
- 동작 중 한 프레임 미탐은 즉시 STOP하지 않는다. 연속 미탐 시간이
  `VISION_LOSS_CONFIRM_SEC=1.0 s` 미만이면 마지막 유효 pose와 원래의 오래된
  timestamp를 사용해 이미 시작한 bounded command만 유지한다. 유효 프레임이
  하나라도 들어오면 연속 미탐 타이머는 0으로 초기화된다.
- 연속 미탐이 1.0초에 도달하면 STOP 후 `RECOVER_VISUAL`로 전환한다. 그 뒤
  `MAX_PNP_LOSS_SEC=1.0 s` 안에 복구되면 재검증하고, 복구되지 않으면
  `FAILED`로 종료한다. 정지/계획 단계에서는 미탐 유예 중에도 새 동작을
  계획하지 않고 STOP을 유지한다.

## 70도 HFOV와 adaptive waypoint 회전

HFOV 70도에 좌우 8% image-edge margin을 적용한 사용 가능 반시야각은 약
30.46도다. 물리 화면 끝까지 약 4.54도의 여유가 있어 회전 STOP margin 최대
4도를 포함한다. 폭 1.1 m 팔레트가 중앙에 있고 정면이라는 단순 조건에서 거리별
가시 회전각은 다음과 같다.

| camera-Z | 전면부 유지 최대각 | 0.8 m 전진시 lateral |
|---:|---:|---:|
| 3.0 m | 20.07도, hard cap 20도 적용 | 27.36 cm |
| 2.5 m | 18.06도 | 24.80 cm |
| 2.3 m | 17.01도 | 23.41 cm |
| 2.0 m | 15.09도 | 20.82 cm |
| 1.5 m | 10.33도 | 14.34 cm |

따라서 고정 20도로 항상 회전하지 않는다. planner hard cap만 20도로 높이고,
매 계획시 실제 네 모서리 픽셀 bearing과 8% margin으로 좌/우 허용각을 다시
계산한다. 가까울수록 자동으로 12도 이하까지 줄어 전면부 가시성을 유지한다.
회전 후 직진거리 `d`의 횡방향 이동량은 `d * sin(turn)`이다.

## 선제 STOP 기준

- 회전: 적합 함수가 목표각을 만족하는 명령 유지시간과 정지 후 관성각을 함께
  역산한다. 실시간 PnP 속도가 더 빠르면 측정 지연 동안의 회전량과 관성 margin으로
  더 일찍 STOP하며, 목표를 통과하거나 계획 시간이 끝나도 STOP한다.
- 거리: 필터 camera-Z와 속도로 같은 lookahead 뒤의 Z를 예측하고 전진 관성
  `0.04 m`를 더한다. 예상 Z가 3 m 밴드에 들거나 목표를 통과하면 STOP한다.
- 위 예측과 별도로 FACE `+/-0.5 deg`, STANDOFF 원시 PnP `3.0 +/-0.1 m`는
  실시간 즉시 STOP 조건이다.

## 회전·전진 통합 가시성 계획 (2026-09-04)

`PREDICTIVE_VISIBILITY_PLANNER_ENABLED=True`이면 실행 가능한 회전각과 그 뒤의
전진 거리를 하나의 후보로 평가한다.

- 녹화 intrinsics에서 확인한 약 55.5~55.7도보다 보수적인 55도 fallback과
  좌우 8% 안전 여유를 적용한다. 실행 중에는 실제 stream intrinsics를 우선한다.
- 카메라 광학중심이 아닌 실제 차량 회전중심을 기준으로 전면 외곽 1.10 m를
  변환한다.
- 회전·전진의 끝점뿐 아니라 동작 중간 경로도 검사한다.
- PnP `rvec/tvec`에서 만든 전면 3D 네 코너가 있으면 상하 잘림도 함께 검사한다.
- lateral 오차가 남았을 때 staging 평면을 넘어가지 않도록 조향 여유를 남긴다.

회전을 실행한 뒤에는 기존 FSM 방식대로 STOP, settle, PnP 재관측을 거쳐 실제

## Planning-only visibility margins (2026-09-07)

FACE centering remains in the initial 4 m approach. After staging starts,
use PLAN -> turn -> STOP/settle/observe -> forward -> STOP/settle/observe -> PLAN.
There is no staging RECENTER action. Visual reacquisition resumes staging,
without restarting FACE or the 4 m approach. Final fork alignment is retained.

Image-edge and center-bearing margin violations no longer interrupt an executing
turn or forward command. IMAGE_EDGE_VISIBILITY_GUARD_ENABLED is now a legacy
trace flag set to False; IMAGE_EDGE_MARGIN_NORM=0.08 still constrains planning.
The current 55 degree HFOV gives a usable half-angle of approximately 23.62 degrees.

For a start inside the safe ROI, the entire turn and forward path must preserve
that ROI. A visible start outside the ROI may turn within the physical viewport,
but must restore the reserved ROI before driving forward. Measured 3D corners
are transformed with the offset camera, including vertical viewport checks.
If no feasible joint action exists, stop with a planner failure, without centering.
If the settled turn cannot continue straight safely, return to PLAN.

Persistent detection/pose loss, command target completion, timeouts, staging
limits and STOP settling remain active. These are geometric predictions;
vehicle inertia and perception errors still require field validation.

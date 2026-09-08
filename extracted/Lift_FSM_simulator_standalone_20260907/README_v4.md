# 현재 FSM · CAN/모델 결과 시뮬레이터

`run_v4_simulator.bat` 실행 후 **http://127.0.0.1:8766/** 에 접속합니다. 터미널에서는 `python serve_v4.py --no-browser`로 실행합니다. Python 3.10 이상과 형제 폴더 `../depth_cam`의 현재 FSM 의존성이 필요합니다. 원본 설정의 모델 파일·회전 보정 아티팩트 검증은 유지합니다. 시뮬레이션 위치 기반 3D 카메라 영상과 가상 결과 오버레이를 표시하며 실제 카메라와 YOLO/PnP 추론은 실행하지 않습니다. 실시간 연결 및 세 조건 영상은 README_fsm_camera.md를 참고하세요.

## 초기 배치와 화면 배율

**2026-09-08 변경:** Run/Step과 Python Session은 이제 FSM 요청 기반 추론을 기본으로 사용합니다. 첫 결과부터 실제 가상 지연을 기다리며 영상 비사용 구간에서는 요청하지 않습니다. 상세 요구사항과 기존 연속 출력 모드와의 차이는 [요청 기반 실행 설명](README_requested_inference.md)을 참고하세요. 아래 주기적 모델 출력/과거 GT 설명은 `inference_mode="continuous"` 비교 모드에 해당합니다.

**초기 배치 · 팔레트 / 지게차**에서 두 물체의 월드 X/Z와 방향을 각각 지정하고 **초기 배치 적용**을 누릅니다. 지게차 위치는 회전 중심, 팔레트 위치는 삽입 전면 중심입니다. 월드 +X는 화면 오른쪽, +Z는 위쪽이며 방향은 0°=위, 90°=오른쪽, ±180°=아래입니다. 팔레트 방향은 입구가 바라보는 방향입니다. 초기 배치는 지게차 (0,-0.68), 방향 0° / 팔레트 (0,2), 방향 180°로 기존 카메라 상대 거리 2m를 유지합니다.

API에서는 `placement_mode="world"`와 `pallet_x`, `pallet_z`, `pallet_heading`, `forklift_x`, `forklift_z`, `forklift_heading`을 사용합니다. FSM에는 월드 배치를 카메라 상대 포즈로 변환해서 전달하며 이후 운동은 계속 CAN 프레임만으로 발생합니다. `truth.world_x/world_z/heading`은 지게차 회전 중심의 월드 위치·방향입니다. 이동·회전해도 팔레트의 월드 위치는 고정됩니다.

기존 API/저장 기록은 `placement_mode="relative"`가 기본이며 x/z/yaw를 카메라 상대 포즈로 해석합니다. 화면의 **카메라 상대 좌표 · 기존 조건**으로도 사용할 수 있습니다. 27조건 실험·threshold 탐색의 X/Z/yaw는 기존과 같이 상대 좌표입니다.

평면 뷰는 자동 줌·자동 추적을 사용하지 않습니다. **100%=10×10m, 200%=5×5m, 50%=20×20m**이며 25~400%를 설정할 수 있습니다. 가로·세로 축척을 같게 유지하기 위해 캔버스 안에 정사각형 영역을 표시합니다. 화면 중심 X/Z는 수동으로 옮길 수 있고 배율은 실행·재생 중에도 바꿀 수 있습니다. 배율 변경은 차량 상태나 FSM에 영향을 주지 않습니다. **100% 복원**은 배율만 복원합니다.

## 실물 팔레트와 차량 치수

좌측 상단 좌표 캔버스는 브라우저 표시 영역 높이의 2/3 크기인 정사각형입니다. 1920×1080 전체화면에서는 720×720 CSS px이며, 일반 창에서는 브라우저 도구 모음을 제외한 표시 높이를 기준으로 줄어듭니다. 좁은 창에서는 우측 조작부 공간도 확보합니다. 캔버스 전체에 100%=10×10m가 표시되고 눈금은 안쪽에 배치됩니다. 넓은 화면의 우측 조작부는 두 열로 표시합니다.

`assets/pallet_top.png`는 사용자가 제공한 투명 상면 사진 원본입니다. 원본은 편집하지 않고 Canvas에서 투명 여백을 제외한 영역 `(81,75,1887,1898)`을 팔레트 외곽 **1.10 × 1.10 m**에 매핑합니다. 사진 아래쪽이 지정 삽입 전면이며 월드 위치는 전면 중심입니다. 배율·방향·차량 움직임에 관계없이 같은 미터 좌표를 사용합니다. 팔레트 깊이를 실험용으로 바꾸면 사진과 충돌 영역도 해당 깊이로 변합니다.

현재 기본 삽입 통로의 양쪽 외측 간격은 **0.71 m**, 두 포크의 외측 간격은 **0.60 m**로 정렬 시 좌우 **0.055 m**씩 여유가 있습니다. 이 폭은 사진 픽셀로 추정하지 않고 기존 차량 물리·충돌 계산과 같은 `vehicle_profile.json`을 사용합니다. 상판 사진 아래의 삽입 통로는 점선과 입구 실선으로 표시하고, 포크는 삽입 위치를 볼 수 있도록 사진 위에 겹쳐 그립니다. 중앙 지지대와 개별 포켓 형상은 상면 사진과 총 폭만으로 확정할 수 없어 기본 모델은 기존 연속 개구부를 유지합니다. 분리 포켓 옵션은 별도의 대조 실험입니다.

FSM `calib/fsm_v4/config.py`의 실측 카메라→포크 끝 **1.18 m**, 회전 중심 Z **−0.68 m**를 차량 프로필에 반영했습니다. 회전 중심→포크 끝은 **1.86 m**입니다. 포크 뿌리의 카메라 기준 위치와 개별 포크 폭은 별도 실측값이 없어 각각 Z=0과 조정 가능한 0.10 m를 사용합니다. 따라서 1.18 m는 실측된 카메라→끝 거리이며 포크 자체 전체 길이로 단정하지 않습니다. 화면 아래 치수표와 175% 이상 배율의 치수선에서 확인할 수 있습니다.

## 매 프레임 절대 오차

화면의 **모델 포즈 오차**에서 X/Y/Z/yaw 평균과 표준편차를 지정합니다. **초기 배치 적용** 후 Run/Step 또는 **끝까지 계산**을 누릅니다.

```text
error_axis[k] ~ Normal(mean_axis, std_axis²)
model_pose_axis[k] = ground_truth_axis(capture_time[k]) + error_axis[k]
```

평균은 **추정값−정답**의 편향입니다. 이전 추정값에 오차를 누적하지 않습니다. 축과 프레임 사이 샘플은 독립이며 yaw는 [-180°,180°)로 정규화합니다. 지연이 있으면 과거 촬영 시점의 정답에 오차를 더합니다.

| 설정 | API 옵션 | 단위 | 화면 초기값 |
|---|---|---|---|
| X 평균 / 표준편차 | `x_bias` / `x_noise` | m | 0 / 0.01 |
| Y 평균 / 표준편차 | `y_bias` / `y_noise` | m | 0 / 0.01 |
| Z 평균 / 표준편차 | `z_bias` / `z_noise` | m | 0 / 0.02 |
| yaw 평균 / 표준편차 | `yaw_bias` / `yaw_noise` | ° | 0 / 1 |
| 모델 출력 FPS | `model_hz` | Hz | 10 |
| 출력 간격 표준편차 | `model_interval_sd` | s | 0 |
| 지연 / 표준편차 | `latency` / `latency_sd` | s | 0.12 / 0 |
| 시야 내 미탐 확률 | `dropout` | 0~1 | 0.05 |
| 수평 시야각 | `camera_hfov_deg` | ° | 55 |
| 최소 전면 가시 비율 | `min_visible_fraction` | 0~1 | 0.5 |

위 값은 **임의 실험값**이며 로그에서 추정한 절대 정확도·미탐률이 아닙니다. 평균·표준편차 0이면 해당 축 오차가 없어집니다. `seed`가 같으면 같은 조건을 재현합니다. API의 생략 기본값은 비교 실험을 위해 포즈 오차 0, 미탐 확률 0, 30 FPS입니다. 화면 초기값과 구분하세요. `example_gaussian_options.json`은 화면의 인식 실험값을 사용하는 예제입니다.

출력 간격은 `max(0.001, 1/model_hz + Normal(0, model_interval_sd²))`, 지연은 `max(0, Normal(latency, latency_sd²))`입니다. 잘린 분포이므로 큰 표준편차에서 실제 평균 FPS/지연은 입력 중심값과 달라집니다. 첫 결과의 FPS는 `null`이고 이후 실제 출력 간격의 역수를 기록합니다. 지연 변동 때문에 늦게 나온 결과가 더 과거의 촬영 시각을 가리킬 수도 있습니다.

모델 결과 사이에는 같은 패킷을 유지합니다. 조회할 때마다 오차를 다시 샘플링하거나 이전 프레임을 새로운 FSM 관측으로 넣지 않습니다. CAN 유지 송신은 모델 FPS와 독립적으로 진행합니다. 미탐 시 `det_ok=false`와 포즈 `null`을 전달합니다. 최대 탐지 거리 제한은 없으며 시야·가시 비율은 정답 기하로 판단합니다. 원거리라는 이유로 미탐 처리하지 않습니다. 화면의 시야 영역은 무한한 시야각 영역을 현재 좌표 뷰 경계에서 잘라 표시합니다. 예전 실행 옵션의 camera_range는 호환을 위해 읽되 무시하고 새 옵션에서 제외합니다. `loss_start`, `loss_duration`으로 연속 미탐 구간을 추가할 수 있습니다. `oracle`은 기하 시야 제한을 해제하는 대조 모드이며 설정한 확률 미탐은 여전히 적용합니다.

## 제어 경계

```text
독립 World의 정답 차량 상태
  → 과거 촬영 포즈 + Gaussian 오차 + 시야/미탐 + 출력 주기
  → lift.model.v1 결과 패킷 → JSON IPC → 별도 FSM 프로세스
  → 현재 CalibrationFSMV4.step (원본 코드, 카메라/추론 런처 미실행)
  → 원본 control.py 프레임 생성 / _write
  → 가상 채널의 lift.can.v1 ID·바이트·시각 → JSON IPC
  → 독립 World의 CAN 수신기 → 차량 운동
```

차량은 FSM 상태 이름, FWD 같은 명령 문자열, 목표값 또는 완료 판정을 읽지 않습니다. `serve_v4.py`, `process_runtime.py`, `protocol_world.py`, `serve_world.py`는 `calib`를 import하지 않습니다. 현재 FSM을 아는 코드는 자식 프로세스의 `current_fsm_driver.py`와 그 코드가 사용하는 `v4_runtime.py`의 가상 시간/설정 연결 및 `current_can_adapter.py`입니다. `v4_runtime.Session`은 비교 검사 기준으로만 남겨 두며 화면 서버는 사용하지 않습니다.

자식 어댑터가 원본 클래스를 직접 생성합니다. PRECHECK, 안정화, 정지 대기, timeout, 실패 분기를 생략하지 않습니다. 가상 시간 참조와 설정은 자식 안에서만 적용·복원합니다. 물리 CAN 초기화·송신 스레드는 실행하지 않고 원본 프레임 생성·write의 수신 채널을 메모리로 바꿉니다. CAN 주기와 명령 burst는 가상 시계로 스케줄링하며 실제 OS 지터·버스 중재·장비 응답을 재현하지 않습니다.

차량 보정은 `vehicle_profile.json`에 독립적으로 저장됩니다. FSM 보정값 변경에 따라 환경이 자동으로 바뀌지 않습니다. **FSM 수정·저장 후 Reset / 초기 배치 적용 / 끝까지 계산을 누르면 새 프로세스가 최신 코드를 읽습니다. FSM 변경 때문에 화면 서버를 재시작할 필요가 없습니다.** 실행 중인 프로세스에는 코드를 섞어 넣지 않으며 기존 실행·기록은 생성 당시 버전을 유지합니다. 시뮬레이터 서버 자체나 차량 프로필 변경에는 서버 재시작이 필요합니다. 자식마다 별도 빈 bytecode 캐시 위치를 사용해 짧은 간격의 같은 크기 파일 수정도 이전 pyc를 재사용하지 않습니다.

## 프로세스 실행과 기존 인터페이스

### main_rec_v4.py의 실제/시뮬레이션 모드

시뮬레이터의 Run·Step·끝까지 계산은 실제 `../depth_cam/main_rec_v4.py --mode simulation --ipc` 프로세스를 실행합니다. 초기 배치가 표시될 때 `[SIMULATION] main_rec_v4.py | FSM PID ... | Virtual CAN` 창이 열리고, 모델 결과·탐/미탐·FPS·가상 시간·CAN과 원본 `ui/diagram_v4.py` 다이어그램을 표시합니다. 합성 카메라 미리보기가 있어도 실제 카메라나 추론은 실행하지 않습니다. 창의 모델 결과는 생성 패킷이며 FSM의 실제 포즈 사용 여부는 `FSM input: model/held/clock`으로 구분합니다.

**화면 구성과 동기화:** 브라우저의 시각화는 좌표평면만 표시하며 배치 설정·Run/Pause/Step·배속·로그 저장 도구를 유지합니다. 카메라 영상·모델 결과·CAN·FSM 다이어그램은 별도 `main_rec_v4.py` 창에 표시합니다. 같은 가상 시각 `t`의 카메라 JPEG와 상태를 FSM 창에 보내고 표시 응답을 받은 뒤 브라우저 좌표평면·시계·녹화를 갱신합니다. 브라우저에 카메라 캔버스를 만들거나 이미지를 디코딩하지 않습니다. Pause·Step·배속·슬라이더 역재생에 모두 적용됩니다. 렌더링이 느리면 실제 재생 속도가 지정 배속보다 낮아질 수 있지만 가상 시간과 FSM 결과는 바뀌지 않습니다. 모델 지연으로 포즈의 촬영 시각이 현재 시각보다 과거인 현상은 설정된 모델 특성이므로 유지됩니다. 브라우저 화면 녹화도 좌표평면을 정사각형 비율로 저장합니다.

끝까지 계산은 계산 중간 상태를 창에 재생하지 않고 완료 프레임을 두 화면에 함께 표시합니다. 저장된 JSON도 표시용 세션을 열어 두 화면에서 재생하며 FSM을 다시 실행하거나 새 런 로그를 만들지 않습니다. `/api/present` 및 `Session.present()`는 표시 전용이며 실제 FSM 상태·CAN·월드·난수·로그를 변경하지 않습니다. API로 직접 `Session(show_window=True)`를 사용할 때도 `present(frame, presentation_id, jpeg)`를 호출해야 창이 표시됩니다. JPEG가 없거나 카메라 렌더링에 실패하면 이전 영상을 재사용하지 않고 해당 프레임의 빈 카메라 영역을 표시합니다.

`depth_cam` 폴더에서 다음과 같이 실행할 수 있습니다.

```powershell
python main_rec_v4.py --mode simulation
python main_rec_v4.py --mode real
```

`simulation`을 직접 실행하면 시뮬레이터 서버와 브라우저를 열고, 브라우저에서 Run/Step을 누르면 같은 런처의 별도 FSM 프로세스와 창이 시작됩니다. `--port 8767`로 서버 포트를 바꿀 수 있습니다. `--ipc`는 서버가 사용하는 모델/CAN JSON 표준입출력 모드이며 단독 실행하면 요청을 기다립니다. 기존 인자 없는 실행은 `real`과 동일합니다. 실제 모드는 기존 `main_rec.py` 경로와 `CAMERA_ENABLED`·`CAN_ENABLED` 설정 및 보정 검증을 유지하고 창 제목에 `[REAL]`을 표시합니다. 시뮬레이션 모드에서는 실제 런처/초기화/물리 CAN 경로를 호출하지 않으며 `--calibrate`와 혼용할 수 없습니다.

완료 시 FSM 계산과 로그 저장은 마감하고 결과 창은 유지합니다. Reset/새 실행/페이지 종료/서버 종료 시 창 프로세스를 정리합니다. 창을 닫거나 Esc/Q를 누르면 실행 중인 런은 다음 tick에 원본 STOP CAN 경로로 취소됩니다(브라우저 Pause 중에는 가상 시간도 정지하므로 다음 진행 때 처리). 완료 후 창 닫기는 프로세스도 종료합니다. 일괄 조건 실험·검증은 같은 `main_rec_v4.py` 진입점을 창 없이 사용합니다. API의 `show_window` 및 Python `Session(show_window=True/False)`로 구분하며 `/api/run`의 `window_session_id`는 완료 창 정리용 세션 ID입니다.

### 런별 자동 로그

Run/Step으로 첫 프레임을 진행하거나 끝까지 계산·조건 실험·그리드 탐색을 실행하면 `run_logs/YYYYMMDD_HHMMSS_마이크로초_고유번호/`에 자동 저장합니다(폴더 날짜는 UTC). 초기 배치 미리보기만으로는 폴더를 만들지 않고, 저장된 궤적 재생도 새 런으로 기록하지 않습니다. `capture=false`인 고속 실험도 자동 파일 기록은 유지됩니다. Run 일시정지는 같은 폴더에 이어서 기록하며 Reset/배치 교체/취소는 기존 런을 `cancelled`로 마감합니다. 화면의 **자동 저장 로그**에 실제 폴더와 저장 상태·건수를 표시합니다.

| 파일 | 내용 |
|---|---|
| `run_meta.json` | 옵션·seed·차량 치수·FSM 설정·소스 해시·프로세스·시간 기준 |
| `run_can_tx.jsonl` | World가 수신한 모든 CAN ID·바이트·HEX·송신 시각. 유지 송신과 STOP 포함 |
| `run_control_seq.jsonl` | 초기 상태, 관측된 FSM 상태 전이(`phase=state`), 명령 변화(`cmd`), 최종 종료 사유(`lifecycle`) |
| `run_model_results.jsonl` | 모델 결과 패킷 전체, 포즈·탐/미탐·FPS·지연·정답·샘플 오차. 생성 sequence별 한 번 |
| `run_pallet_state.csv` | 모델 결과 프레임별 포즈·탐/미탐·정답·오차를 표로 기록 |
| `run_relative_pose.csv` | 지연·모델 오차 없는 현재 정답 위치 관계. 초기·각 FSM 주기·최종 시점 기록 |
| `run_inference_timing.csv` | 가상 결과 주기·처리 지연. `inference_ran=0`, `pose_src=simulation`이며 실제 추론 시간 측정이 아님 |
| `run_fsm_steps.jsonl` | FSM 각 호출 주기의 실제 입력, 모델 갱신/step 여부, 상태·명령·CAN·가상 시간·화면 진단 |
| `run_summary.json` | 완료·실패·시간 제한·취소·프로세스 오류, 충돌·잔여 거리 및 최종 로그 건수 |

실차 `calib/tracelog.py`의 `t_iso`, `t_mono`, `frame_i`, `det_ok`, `phase` 등 공통 이름을 사용하지만 파일 스키마가 실차와 완전히 동일하지는 않습니다. `control_seq`는 프로세스 응답의 상태·명령 변화를 기록하며 실차 tracer의 내부 `begin/end` 이벤트를 추정해서 만들지 않습니다. 모델 생성 시점의 상태와 실제 FSM 입력 시점은 `model_results`와 `fsm_steps`를 구분해 비교하세요. 비전 독립 삽입 구간에서도 모델 생성 로그는 남지만, FSM 실제 입력은 null 포즈일 수 있습니다.

`sim_time_s`는 런 시작부터의 가상 초, `t_mono=1000+sim_time_s`, `t_iso`는 런 시작 UTC에 가상 시간을 더한 비교용 시각입니다. 배속 실행의 실제 벽시계 시각과 다릅니다. 파일은 각 실행 주기마다 flush하므로 Pause 중에도 읽을 수 있습니다. 정상 종료·취소·FSM 장애는 종료 요약을 마감합니다. 서버 강제 종료/전원 차단은 종료 처리가 보장되지 않으며, 이 경우 기존 행과 `running` 요약이 미완료 런으로 남습니다. 디스크 기록 실패는 화면과 결과의 `logging.error`에 표시하고 FSM/CAN 판단은 계속 유지합니다. 자동 삭제는 하지 않습니다. Python 검사에서 기록이 필요 없으면 `Session(..., log_root=None)`으로 끌 수 있습니다.

`run_relative_pose.csv`의 리프터 월드 위치는 회전 중심, 팔레트 월드 위치는 삽입 전면 중심입니다. 카메라 기준 팔레트 X/Y/Z와 yaw, 회전 중심 기준 상대 X/Z·직선 거리, 포크 중앙 끝의 월드 좌표, 중앙/좌우 외측 포크 끝의 팔레트 좌표를 저장합니다. 팔레트 로컬 +Z는 전면에서 내부로 향하는 방향입니다. `tip_insertion_depth_m`은 중앙 끝의 로컬 Z를 0 이상으로 자른 종방향 진입 깊이이며, 옆으로 벗어나 있더라도 양수가 될 수 있어 성공 판정은 충돌·정렬과 함께 확인해야 합니다. `insertion_remaining_m`은 기존 검증기의 카메라 Z 잔여 거리이고 `minimum_clearance_m`은 런 전체의 최소 여유입니다. 모델 결과의 촬영 시점 정답은 별도로 `run_model_results.jsonl`에 있습니다.

### 실행 구조

`controller_process.py`가 숨김 자식 프로세스 `fsm_worker.py`를 시작합니다. 부모→자식은 모델 결과와 가상 시간, 자식→부모는 CAN 프레임입니다. 실행 관리용 `lift.worker.v1`은 `info`, `init`, `tick`, `stop`, `close` 요청과 request_id를 사용합니다. 제어 데이터와 별도로 `lifecycle {finished,outcome,reason}` 및 선택적 `telemetry`를 전달합니다. 화면 서버는 DONE/FAILED라는 내부 상태명으로 실행 종료를 판단하지 않습니다. 다른 상태명을 쓰거나 `_samples` 같은 내부 속성이 없는 제어기도 이 관리 규약을 구현하면 연결할 수 있습니다.

stdout은 UTF-8 JSON 전용이고 FSM 출력·예외는 stderr로 분리됩니다. 응답 제한은 기본 10초, 응답 4MB입니다. 응답 누락, 프로세스 종료, 잘못된 JSON/요청 번호는 실행 장애로 기록하고 프로세스를 종료·회수합니다. 장애 시 가짜 STOP을 만들지 않고 CAN 수신기의 watchdog과 설정한 관성이 정지할 때까지 차량 시간을 진행합니다. 시간 제한이나 정상 취소는 자식의 원본 STOP 경로를 통해 CAN을 받고 종료합니다. Reset 교체, 세션 퇴출, 페이지 종료, 서버 정상 종료에도 자식을 정리합니다. 실행 완료 시에도 남은 자식 프로세스를 유지하지 않습니다.

브라우저는 좌표평면과 조작·설정 도구를 표시합니다. 별도 FSM 창은 원본 `ui/diagram_v4.py` OpenCV 렌더러로 **Acquire Pallet → Set Standoff → Plan Approach → Execute Approach → Insert Forks** 단계와 동작 목표를 표시하고 모델 결과·CAN·합성 카메라 영상을 함께 보여줍니다. 방문 상태 이력도 로그로 기록하며 표시용 다이어그램은 제어기를 대신하지 않습니다.

최신 FSM은 `vision_independent=True`인 삽입 단계에서 실차 런처처럼 포즈 없이 step을 계속 호출합니다. 이때 모델 FPS와 별도로 최대 10ms 간격의 가상 타이머로 진행합니다(실차 waitKey(10) 루프에 대응하는 스케줄 가정). 화면에 **비전 독립 단계 · 포즈 없이 타이머로 진행**을 표시하고 trace의 실제 inputs는 null 포즈로 기록합니다. 일반 비전 단계에서는 새 모델 sequence만 관측으로 공급합니다. 환경에서 생성한 모델 결과가 있다고 해서 비전 독립 단계에 강제로 사용하지 않습니다.

남는 제약: 현재 FSM의 클래스/step, 설정 검증, 가상 시계와 `control.py` 내부 함수 연결은 여전히 **자식 어댑터 내부의 의존성**입니다. 해당 구조가 바뀌면 어댑터 조정이 필요합니다. 프로세스 분리가 실제 CAN 송신 스레드·실차 물리와 100% 일치한다는 의미는 아닙니다. 배속 실행을 위해 공통 가상 시간과 lifecycle 관리 인터페이스도 필요하므로 CAN/모델 데이터 두 종류만으로 모든 실행 관리가 끝나는 구조는 아닙니다.

## FSM 입력

현재 `main_rec.py`의 `fsm_step_kwargs`와 동일한 필드명을 사용합니다. 자동 검사는 실제 런처의 AST와 필드 집합을 대조합니다. 전체 예제는 `GET /api/protocol`로 확인할 수 있습니다.

| 입력 | 공급 방식 |
|---|---|
| `det_ok` | 시야/가시 비율 + 미탐 확률 (거리 제한 없음) |
| `detected_length` | 선택한 전면 폭 |
| `dist_z` | 정답 Z + Z 오차 |
| `yaw_smooth` | 선택한 면의 정답 yaw + 각도 오차 |
| `offset_smooth` | [X,Y,Z] 결과, 미탐이면 null |
| `target_bearing_deg` | 관측 모델 중심의 방위각 |
| `vision_meta` | 투영 기하·원시 포즈·촬영/결과 시간 |

`vision_meta` 기하 필드:
`center_bearing_deg`, `front_face_center_bearing_deg`, `model_center_bearing_deg`, `bbox_margin_norm`, `frame_width`, `frame_height`, `fx`, `fy`, `ppx`, `ppy`, `face_corners_px`, `face_corners_camera_m`, `selected_front_face`, `selected_face_area_px2`, `projected_face_areas_px2`, `yaw_raw_deg`, `pos_x_m`, `pos_z_m`.

640×480, 중심 (320,240), 설정한 수평 시야각의 핀홀 투영을 사용합니다. 기본은 관측 가능한 수직 면 중 최대 투영 면 선택이며 지정 전면 유지도 가능합니다. 오차가 있는 포즈에서 모서리·방위각을 함께 계산합니다. 영상 추론으로 얻은 실제 bbox가 아니라 투영 모형입니다.

시간 필드:
`measurement_mono`, `fsm_step_mono`, `inference_latency_sec`, `inference_start_mono`, `inference_end_mono`, `model_inference_latency_sec`, `pose_result_mono`, `camera_frame_number`, `sensor_timestamp_ms`, `sensor_timestamp_domain`.

추론 관련 이름은 프로토콜 호환용이며 여기서는 가상 처리 지연을 뜻합니다. 정답·샘플링 오차·충돌은 진단 API/결과에만 담고 FSM 모델 패킷에 넣지 않습니다. 현재 FSM은 CAN 수신 피드백을 입력으로 사용하지 않습니다.

## CAN 프로토콜

```json
{"protocol":"lift.can.v1","t":0.0,"id":483,"data":[127,127,67,127,127,127,127,127],"dlc":8,"extended":false,"remote":false}
```

JSON ID/바이트는 10진수이며 t는 시작부터의 초입니다.

| ID | 해석 |
|---|---|
| 0x1E3 | 8바이트 이동. 0부터 세는 [1] 조향, [2] 전후진, 중립 127 |
| 0x2E3 | 주행 모드: HEX 42 00 00 0A SS 40 69 93. SS는 4비트 카운터. emergency는 첫 바이트 0x80 |
| 0x764 | heartbeat [0] |

현재 전진/후진은 [2]=67/187, 강도 30 제자리 좌/우회전은 [1]=157/97입니다. 다른 강도와 동시 조향·전후진도 처리합니다. 강도 비례 배율은 가정입니다. 미지원 ID, 확장/remote 프레임은 운동에 사용하지 않습니다. 카운터는 기록하지만 순서 검사로 차단하지 않습니다.

어댑터는 원본 설정의 control 5ms / movement 10ms / heartbeat 200ms, 초기 burst, 정지→이동 burst를 생성합니다. 수신기의 heartbeat 0.6s / control·movement 0.1s watchdog과 주행 모드 검사는 차량 프로필의 **시뮬레이션 가정**입니다.

## 외부 FSM 연결

독립 환경만 실행하려면 `python serve_world.py --port 8767`을 사용합니다. 이 서버는 Python 표준 라이브러리만 필요하고 depth_cam 없이도 동작합니다. 화면의 외부 world 관찰 기능을 쓰려면 클라이언트를 화면 서버의 8766 API에 연결합니다.

```powershell
python protocol_client.py --controller demo --url http://127.0.0.1:8766 --realtime
python protocol_client.py --controller current --url http://127.0.0.1:8766 --options example_gaussian_options.json --seconds 60 --realtime --output external_run.json
```

시작 시 출력되는 World ID를 화면의 **외부 world ID**에 넣고 **외부 world 관찰**을 누릅니다. 화면은 진단만 조회하며 제어·시간 진행은 클라이언트가 담당합니다.

1. `POST /api/world/create`: `{"options":{...}}` → id, model, sim_time_s, next_model_time_s.
2. model.step을 FSM에 전달합니다. 모델의 단조 시계는 `1000 + sim_time_s`입니다. 제어기 timeout에도 이 시계를 사용합니다.
3. `POST /api/world/advance`: id, dt, can_frames. CAN t는 현재~현재+dt 사이 비감소 순서입니다. dt 0~1s, 최대 1,000프레임입니다. **모든 결과를 받으려면 next_model_time_s를 넘지 않도록 진행**합니다. 큰 간격으로 진행하면 마지막 모델 결과만 반환됩니다.
4. sequence가 바뀔 때만 새 FSM 관측으로 처리합니다. CAN 유지 송신은 계속합니다. `POST /api/world/model`은 현재 패킷을 조회할 뿐 시간·오차를 갱신하지 않습니다.
5. `POST /api/world/diagnostics`, `/api/world/can-log`: id를 보내 정답·충돌·수신 CAN을 검사합니다.

모델 transport는 여기서 정의한 JSON 결과 인터페이스입니다. 실차 런처는 프로세스 안에서 Python kwargs를 전달하므로, 기존 실차 실행파일을 수정 없이 네트워크에 연결하는 범용 드라이버까지 제공한다는 의미는 아닙니다. CAN 수신 피드백, 유압 승강·접기·리치 동작, 런처 키보드 debug-step은 현재 지원 범위 밖입니다.

## 삽입 검증과 저장

단일 실행, X/Z/yaw 조건 조합·seed 반복, threshold 탐색, 조건 재현, 궤적 재생·CSV/JSON·영상 저장을 제공합니다. **CAN 프레임 JSON**은 가상 채널의 ID·데이터·시각을 저장합니다. 전체 실행 JSON에는 모델 입력과 샘플링 오차가 들어갑니다. **원본 FSM 재입력 검증**은 같은 FSM·시뮬레이터 버전에서 상태·명령·실패 원인과 전체 CAN 프레임을 대조합니다.

FSM DONE과 외부 삽입 판정은 별도입니다. 포크/개구부 충돌, DONE 이후 잔여거리, 인식 상실 복구 실패, timeout을 구분합니다. 충돌은 기하 관찰이며 접촉력·충돌 후 정지 반응은 재현하지 않습니다. 평면 물리이므로 Y 오차는 FSM 입력에 공급하지만 수직 충돌 판정에는 사용하지 않습니다.

```powershell
python -m unittest discover -s tests -p "test*.py"
python serve_v4.py --validate
node --test tests/*.test.js
```

54조건의 잡음 없는 기준 결과는 `validation_v4/FINDINGS.md`, `conditions.csv`, `validation.json`에 있습니다. 명시한 환경 가정에 대한 검사이며 실차 성공률이 아닙니다. 새 오차 분포로 조건 실험을 돌려 비교할 수 있습니다.

나중에 정답 대비 실측 로그가 준비되면 `model_result_model.py`의 sampler를 교체하여 경험 분포·상관관계·연속 미탐 패턴을 반영할 수 있습니다. 현재는 요청한 임의 Gaussian μ/σ 방식입니다. 기존 example_estimation_error_logs.csv도 실제 정확도 보정 자료로 사용하지 않습니다.

기존 index.html/sim.js는 압축파일의 이전 FSM 비교용입니다. 현재 FSM 검증은 v4.html을 사용합니다.


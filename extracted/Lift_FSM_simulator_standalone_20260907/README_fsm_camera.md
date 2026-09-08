# FSM 실시간 가상 카메라와 세 배치 실행 영상

`main_rec_v4.py --mode simulation` 및 `run_v4_simulator.bat`에서 실행하는 시뮬레이터에 3D 카메라를 연결했다. 브라우저는 좌표평면을 표시하고, 별도 네이티브 FSM 창에서 시뮬레이션 좌표로 계산한 RGB·기존 FSM 오버레이·다이어그램을 표시한다. 두 창은 같은 가상 시각으로 동기화된다. 실제 카메라·신경망 추론·물리 CAN은 실행하지 않는다.

## 실행

현재 UI 기본 카메라 높이는 **0.65m**로, 이전 예시 영상의 0.25m보다 0.40m 높다. 대응하는 `vertical_offset`은 **0.575m**다. 페이지를 새로고침한 뒤 **초기 배치 적용**으로 새 실행에 반영한다. 방향/시야각은 기존과 동일하다. 높인 카메라의 세 조건 영상은 `fsm_camera_runs_h65/three_conditions.mp4`, 개별 조건·로그는 같은 폴더의 `index.html`/`cases.json`에서 확인한다. 아래 기존 결과 표와 `fsm_camera_runs/` 영상은 0.25m 기준 기록이다.

```powershell
python render_simulation_runs.py --height .65 --output fsm_camera_runs_h65
```

실행 중인 기존 서버는 종료 후 재시작해야 Python 변경이 반영된다.

```powershell
# 작업공간 루트에서 실행
python extracted/depth_cam/main_rec_v4.py --mode simulation
```

브라우저의 **초기 배치 적용 → Run**으로 진행한다. Step·궤적 재생에도 같은 카메라가 연결된다. 첫 메시/텍스처 로딩에는 수 초 이상 걸릴 수 있다. 렌더링은 FSM 계산과 분리되어 초기 로딩 때문에 제어기의 가상 시간이 진행되지 않는다. 카메라 JPEG 하나와 같은 시각의 위치·CAN·모델 결과·FSM 상태를 네이티브 창에 보낸 뒤, 표시 응답을 받아 브라우저 좌표평면·시계·녹화를 갱신한다. 브라우저에는 카메라 영상과 FSM 다이어그램을 중복 표시하지 않는다. 다음 Run/Step 계산은 표시 완료를 기다린다. Pause와 역재생도 두 화면에 같은 가상 시각을 표시한다. 렌더링 중 밀린 탐색 요청은 최신 것 하나만 남기며 로그/FSM 관측을 버리지 않는다. 두 OS 창의 실제 화면 갱신에는 짧은 IPC/그리기 지연이 있지만 서로 다른 시뮬레이션 프레임을 독립적으로 재생하지 않는다.

추가 패키지: `moderngl`, `trimesh`, `imageio-ffmpeg`와 기존 numpy/OpenCV/Pillow. OpenGL 3.3 이상이 필요하다. 현재 PC에서 설치·실행 확인했다.

## 좌표와 출력 경계

`truth.world_x/world_z`는 리프터 회전 중심이다. 회전 중심–카메라 오프셋은 `vehicle_profile.json`의 `CAMERA_TO_ROT_CENTER_X_M/Z_M`을 사용한다. 현재 Z=-0.68m이므로 카메라는 회전 중심보다 전방 0.68m다.

```
R(h) = [[cos(h), sin(h)], [-sin(h), cos(h)]]
camera_world = pivot_world - R(heading) * camera_to_pivot
pallet_front_world = camera_world + R(heading) * [truth.x, truth.z]
pallet_inward_heading = heading + truth.yaw
```

OBJ와 바닥은 월드 좌표로 렌더링한다. 팔레트는 고정되고 리프터·카메라가 움직인다. 카메라 높이는 바닥에 놓인 명목 높이 0.15m 팔레트를 기준으로 `0.075 + vertical_offset`m로 해석한다. 실제 센서 높이 실측값을 새로 추정한 것은 아니다. 수평 시야각은 시뮬레이션의 `camera_hfov_deg`, 해상도는 640×480, 중심점은 (320,240), 렌즈 왜곡은 0이다. 이 버전은 팔레트·바닥을 렌더링하며 포크/차체에 의한 RGB 가림, 센서 depth, 노출·렌즈 효과는 추가하지 않았다.

`frame.model_packet`은 그 시각 FSM에 전달한 실제 가상 결과 패킷이다. 렌더러는 오차를 다시 샘플링하거나 새 추정값을 만들지 않는다. `calib.camera_overlay.draw_camera_overlay`를 함께 사용하며, `vision_independent`/`skip_detection`을 전달해 실화면의 표시 생략 규칙을 유지한다. 현재 RGB 위에 지연된 가상 결과를 올리는 방식이므로 이동 중 레이블이 뒤처지는 현상도 나타난다.

웹 요청 `/api/camera`는 `{frame, options}`를 받아 JPEG·시각·모델 번호·처리시간을 반환한다. 렌더링 전용 프로세스가 GPU 컨텍스트를 유지한다. World/UI 부모 프로세스는 calib/추론 모듈을 import하지 않는다. 이어서 `/api/present`에 `{id, presentation_id, frame, options, jpeg}`를 보내면 같은 JPEG와 기록된 상태를 네이티브 창에 표시하고 시각·표시 번호를 확인한다. 이 경로는 FSM `.step()`, CAN 송신, 월드 시간 진행, 모델 오차 샘플링, 런 로그 기록을 호출하지 않는다. FSM `.step()` 입력은 기존 모델 결과 프로토콜 그대로다. 단독 worker의 기존 비동기 카메라 경로도 유지하지만, 브라우저 Session은 `sync_display=true`로 같은 이미지와 상태를 공유한다.

## 세 배치 결과

공통 리프터 회전 중심 (0,-0.68)m, 방향 0°. 카메라 높이 예시값 0.25m(`vertical_offset=.175`), RGB 30FPS, 가상 결과 10Hz, 지연 120ms, 미탐 5%, Gaussian 표준편차 XYZ=1/1/2cm·yaw=1°, 평균 0. seed는 순서대로 20260907~20260909다.

| 조건 | 팔레트 전면 월드 X/Z | 팔레트 입구 방향 | 결과 | 시뮬레이션 시간 |
|---|---|---|---|---|
| 01_front | 0 / 2.0m | 180° | DONE, 삽입 검증 통과 | 12.56초 |
| 02_right | +0.30 / 2.2m | 165° | FAILED | 10.10초 |
| 03_left | -0.30 / 2.2m | -165° | FAILED | 4.20초 |

실패 사유는 두 조건 모두 `no commandable visibility-safe insertion rotation (minimum 0.50deg)`다. 실패를 성공처럼 보이도록 수정하거나 조건을 바꾸지 않았다. 성공률 추정 실험이 아니라 요청한 세 배치의 실행 결과다.

`fsm_camera_runs/three_conditions.mp4`는 세 조건을 연속으로 보여준다(약 29.87초). 개별 영상과 `index.html`도 같은 폴더에 있다. 1배속이며 각 조건 마지막 1초는 최종 상태를 유지한다. 좌측 2D와 우측 RGB는 같은 로그 행을 사용한다. 30FPS 출력 시각보다 늦지 않은 최신 로그 행을 선택하며, 임의 경로나 새로운 오차를 생성하지 않는다.

재생성:

```powershell
cd extracted/Lift_FSM_simulator_standalone_20260907
python render_simulation_runs.py
# 기존 실행 로그만 다시 렌더링
python render_simulation_runs.py --render-only
```

`cases.json`은 조건·결과·원본 로그 경로·해시·렌더링 성능을 기록한다. `*_report.json`은 전체 FSM/CAN trace이고, `logs/<run>/run_fsm_steps.jsonl` 및 `run_meta.json`이 영상 생성의 실제 입력이다. 각 `*_video_frames.csv`에는 영상 프레임 → 로그 행/시각/모델 번호/카메라·팔레트 좌표의 대응이 있다.

## 검증

- 세 실행 모두 동일 기록을 새로운 FSM 프로세스에 재입력해 상태·명령·CAN 재현 검사 통과.
- 개별/합본 영상 전체 디코딩 완료, 합본 896프레임. 두 뷰 동일 로그 시각 및 팔레트 월드 위치 고정 확인.
- 렌더링+오버레이 중앙값 약 5.1~5.5ms, 95백분위 약 7.2ms. 파일 인코딩·네트워크·브라우저 표시 시간은 별도다.
- 실시간 HTTP 요청의 반환 시각·모델 sequence 일치 확인. 네이티브 창을 켠 실행에서도 초기 3D 로딩이 FSM tick을 막지 않는 것 확인.
- 기존 프로세스 경계 테스트 11개, 로그 테스트 5개, 공통 오버레이 테스트 3개, 좌표 변환 테스트 1개 통과. JS에서 요청 병합·Reset 시 이전 응답 폐기와 기존 2D 좌표 테스트 통과.
- 연결된 브라우저가 없어 브라우저의 실제 클릭/배치 검증은 수행하지 못했다. HTTP·JS 단위 검사와 생성 이미지/영상을 확인했다.

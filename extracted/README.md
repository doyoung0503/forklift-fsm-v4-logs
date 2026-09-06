# FSM v4 독립 실행 묶음

이 폴더는 FSM v4 런타임에 필요한 공통 `calib` 코드, UI 코드, YOLO Pose 모델을
함께 포함합니다. 현재 v4 시작기는 새 모델의 회전 응답을 자동 생성하므로 작업공간의
형제 폴더 `rotation_fit/`도 함께 있어야 합니다.

단, Python 패키지와 RealSense 장치 드라이버는 실행 PC에 별도로 설치되어
있어야 합니다. Kvaser 패키지/장치는 CAN ON일 때만 필요합니다. 녹화 결과는
실행 후 `depth_cam/rec/`에 생성됩니다.

## 포함 항목

- 프로젝트 루트의 유일한 `.pt` 파일: 팔레트 YOLO26 Pose 모델(파일명 자유)
- `depth_cam/main_rec_v4.py`: FSM v4 실행 진입점
- `depth_cam/main_rec.py`: 카메라, 추론, 녹화, FSM 공통 실행부
- `depth_cam/calib/`: 공통 코드와 FSM v4 전체 코드
- `depth_cam/ui/`: HUD 상태도 코드
- `depth_cam/docs/FSM_V4.md`: FSM v4 설정 및 단계 설명
- `preflight_v4.py`: 실행 전 파일·패키지·import 검사
- `run_v4_windows.bat`, `run_v4_gitbash.sh`: 검사와 시작 확인을 포함한 실행 파일
- `../rotation_fit/auto_calibrate_rotation.py`: 재추론부터 검증·원자 적용까지 수행

## Windows 설치

Python 3가 설치되어 있는지 확인합니다.

```text
py -3 --version
```

프로젝트 최상위 폴더에서 Python 패키지를 설치합니다.

```text
py -3 -m pip install -r requirements.txt
py -3 -m pip install -r ../rotation_fit/requirements.txt
```

Intel RealSense 드라이버도 설치해야 합니다. 모니터 테스트 기본 설정인
`CAN_ENABLED=False`에서는 Kvaser가 없어도 실행됩니다. 실차 구동을 위해
`CAN_ENABLED=True`로 바꿀 때만 Kvaser CANlib SDK/드라이버와 아래 패키지를
추가 설치합니다.

```text
py -3 -m pip install -r requirements-can.txt
```

GPU를 사용할 경우 실행 PC의 CUDA 환경에 맞는 PyTorch를 먼저 설치하는 것이
좋습니다.

## 실행 전 검사

프로젝트 최상위 폴더에서 다음 명령을 실행합니다.

```text
py -3 -u preflight_v4.py
```

모든 항목이 `OK`여야 합니다.

## 권장 실행 방법

Windows 탐색기 또는 CMD에서는 다음 파일을 실행합니다.

```text
run_v4_windows.bat
```

Git Bash에서는 다음 명령을 사용합니다.

```text
bash run_v4_gitbash.sh
```

모델 경로는 프로젝트 루트의 유일한 `.pt` 파일을 자동 탐색하므로 파일명이나
현재 작업 디렉터리에 의존하지 않습니다. 루트에 `.pt`가 없거나 두 개 이상이면
잘못된 모델 선택을 막기 위해 시작 단계에서 중단합니다.

`main_rec_v4.py`는 카메라/CAN 초기화 전에 모델 SHA-256을 확인합니다. 같은 모델로
검증된 `rotation_model.generated.json`이 없으면 raw 녹화 재추론 → 로그 검증 →
`yaw_deg`/`heading` 함수 적합 → LOO/안전 검증 → 원자 적용을 자동 수행합니다.
한 단계라도 실패하면 FSM을 시작하지 않고 기존 artifact를 보존합니다. 실행 중
hot-swap은 하지 않습니다.

직접 실행할 때는 모듈 import와 녹화 출력 위치를 일정하게 유지하기 위해
`depth_cam`으로 이동하는 방식을 권장합니다.

```text
cd depth_cam
py -3 -u main_rec_v4.py
```

## 중요 안전 주의사항

- 기본값 `CAN_ENABLED=False`에서는 SEARCH의 논리 명령만 진행되고 차량은
  움직이지 않습니다. `CAN_ENABLED=True`로 바꾸면 준비 직후 우회전 SEARCH가
  실제로 시작되므로 시작 확인이 있는 제공 실행 파일을 권장합니다.
- 현재 코드는 카메라/추론 처리 자체가 완전히 멈췄을 때 마지막 CAN 명령을
  독립적으로 만료시키는 하위 watchdog이 없습니다. 실제 바닥 주행 전에 별도
  watchdog을 구현하고 검증해야 합니다.
- `depth_cam/calib/fsm_v4/config.py`의 위치 계측값은 아직 임시값입니다.
- `AUTO_INSERT_ENABLED`의 기본값은 `False`이며, 계측과 검증 전에는 그대로
  유지해야 합니다.
- 비상정지 수단과 충분한 안전 공간 없이 실차를 구동하지 마십시오.

## 주요 설정 파일

FSM v4의 조정값은 아래 파일 한 곳에서 변경합니다.

```text
depth_cam/calib/fsm_v4/config.py
```

모니터/실차 모드는 같은 파일의 `CAN_ENABLED` 한 항목으로 바꿉니다.
연속 미탐 판정시간은 `VISION_LOSS_CONFIRM_SEC`이며 기본값은 1.0초입니다.

모델 경로, 스트림 설정 등 공통 설정은 다음 파일에 있습니다.

```text
depth_cam/calib/config.py
```

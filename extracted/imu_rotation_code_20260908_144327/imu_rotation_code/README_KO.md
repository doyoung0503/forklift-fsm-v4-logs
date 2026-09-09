# IMU 기반 회전 코드 묶음

기존 소스를 수정하지 않고 폴더 구조를 유지해 묶었습니다.

## 주요 코드
- depth_cam/rotation_return_test.py:110 — RealSense accel/gyro 스트림 시작
- depth_cam/rotation_return_test.py:282 — next_frame(): IMU 입력 수신
- depth_cam/rotation_return_test.py:71 — 자이로 Y축 각속도 적분으로 상대 회전각 계산
- depth_cam/rotation_return_test.py:523 — rotate(): 시작 각도 대비 회전량 비교
- depth_cam/rotation_return_test.py:544 — self.executor.exec("STOP")
- depth_cam/calib/fsm/commands.py:115 — STOP을 issue_command_stop()으로 연결
- depth_cam/calib/control.py:486 — CAN 송신 상태를 stop으로 변경
- CAN/control_forklift_v2.py — 참고용 수동 CAN 제어 파일. 이 회전 테스트에서 직접 사용하지 않음.

## 현재 동작
90도 왕복 회전 진단용 코드이며 기본 정지 임계값은 88도입니다.
정지 후 IMU 안정값을 측정하며 추가 회전 보정은 비활성화되어 있습니다.
정지 전후 팔레트 PnP 위치 비교를 수행하므로 IMU만 사용하는 독립 스크립트는 아닙니다.
calib/fsm/__init__.py의 간접 import에 필요한 FSM 파일도 포함했습니다.

## 환경
- IMU 지원 Intel RealSense 카메라
- Kvaser CAN 장치, CANlib 드라이버/SDK 및 지게차 연결
- Python 패키지: numpy, opencv-python, pyrealsense2, canlib, torch, ultralytics, scikit-learn
- depth_cam/requirements.txt는 기존 전체 환경 목록입니다. CUDA 버전 지정 PyTorch가 포함되어 있으므로 환경에 맞게 설치해야 합니다.
- pallet_yolo26n_pose_livegt.pt를 포함했습니다. calib/config.py가 프로젝트 루트의 .pt 파일을 찾으므로 압축 내부 폴더 구조를 유지하십시오.

## 실행 위치와 명령
압축 해제 후 depth_cam 폴더에서 실행:
    python rotation_return_test.py

IMU 정지 임계값을 90도로 지정하려면:
    python rotation_return_test.py --imu-stop-deg 90

위 명령은 실제 CAN 회전을 수행합니다. 이번 압축 작업에서는 장비 실행이나 동작 검증을 하지 않았습니다.
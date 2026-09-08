# 3D 카메라 영상 + 오차 분포 기반 가상 레이블

`camera_preview.py`는 정해진 카메라 경로를 사용하는 예시 생성기다. **신경망 모델은 로드하거나 실행하지 않는다.** 기존 `GaussianResultModel.sample()`과 `protocol_world.observation()`을 그대로 사용한다. 실제 FSM 모드의 실시간 연결과 로그 기반 세 조건 영상은 [README_fsm_camera.md](README_fsm_camera.md)를 참고한다.

## 결과 열기

- `camera_preview_output/index.html`: 영상 재생 페이지
- `camera_preview_output/pallet_camera_demo.mp4`: 좌측 RGB / 우측 정답·가상 레이블 비교, 18초, 30FPS
- `camera_preview_output/camera_raw.mp4`: 레이블 없는 640×480 RGB
- `camera_preview_output/measurements.json`: 프레임별 카메라 좌표, 정답 상대 포즈, 가상 결과 패킷, 오차, 처리시간

실화면 `main_rec.py`에서 그리기 코드를 `calib/camera_overlay.py`로 분리하여 실화면과 예시 영상이 동일 함수를 호출한다. 자홍색 전면 0~3(두께 2), 주황색 후면 4~7(두께 1), 회색 몸체 중심 8, 선택 전면 자홍색 윤곽(두께 3)과 `FRONT(area): 면 이름`, 0.35m XYZ 축 및 yaw/pitch/roll을 실화면 규칙 그대로 표시한다. 표시용 Y는 위, Z는 카메라 방향이며 포즈 좌표 규약은 변경하지 않는다. 화면 중앙 회색 십자도 유지한다.

모든 팔레트 표시는 오차를 적용한 같은 결과 패킷에서 유도한다. 선택된 면의 포즈에서 원래 키포인트 좌표계를 복원해 전면 선택이 바뀌어도 번호 0~8을 유지한다. 별도 픽셀 잡음이나 신경망 추론은 없다. 평면 운동 모델이므로 pitch/roll은 0이다. 모델 치수 1.1×0.15m와 실제 메시의 세부 외곽은 다를 수 있다.

미탐/검출 생략에서는 팔레트 표시 없이 화면 중앙 십자만 남는다. 비전 독립 단계는 실화면처럼 십자까지 없는 원본 RGB다(`draw_packet_overlay(..., vision_independent=True)`). 예시 영상은 FSM 상태를 실행하지 않으므로 비전 독립 단계로 자동 전환하지 않는다. 정답 비교선은 기본으로 끄며 `--show-truth`를 명시한 진단 영상에서만 청록색으로 추가한다. 실제 화면처럼 정보 패널은 RGB와 분리하며, 예시의 하단에는 가상 출력임을 명시한다.

## 예시 조건

팔레트 전면 중심의 월드 좌표는 (0, 0.075, 2)m로 고정한다. 0~6초는 카메라가 전면 거리 3m에서 1m까지 접근, 6~12초는 방향을 유지하면서 좌우 0.55m 이동, 12~18초는 팔레트 몸체 중심 주위를 ±50° 이동하면서 카메라 방향을 바꾼다. 접근 종료와 이후 경로는 위치가 이어진다. 운동은 CAN/FSM으로 만든 경로가 아니다.

카메라 높이 0.25m는 **예시 가정**이며 실측값이 아니다. 수평 시야각 55°, 640×480, fx=fy=320/tan(27.5°), 주점 (320,240), 렌즈 왜곡 0을 사용한다. 기본 시뮬레이터의 vertical_offset=0은 이 렌더러에서 카메라 높이 0.075m에 해당한다. 높이 0.25m는 vertical_offset=0.175m다.

가상 출력은 10Hz, 지연 0.12초, 독립 Gaussian 표준편차 X/Y/Z=0.01/0.01/0.02m, yaw=1°, 평균 0, 미탐 확률 0.05, seed=20260907이다. 실제 오차를 추정한 값이 아니라 UI 예시값이다. 결과 생성 시각마다 한 번 샘플링하고 출력 사이에는 동일 패킷을 유지한다. 과거 촬영 시점의 정답에 오차를 더하므로 현재 영상 위의 레이블에는 지연에 의한 어긋남도 보인다. 시작 직후 과거 시각은 0초로 제한한다. 범위·면 가시성 판정도 기존 프로토콜을 사용한다.

OBJ는 Y-up, 미터 단위이고 메시 경계 크기는 약 1.104122×0.156890×1.110510m다. 축척을 변경하지 않고 X 중심과 전면 Z 원점만 맞춘다. 원래 사진 텍스처와 하부 녹색 재질을 읽는다. 메시의 미세한 외곽 요철·치수와 시뮬레이터의 명목상 전면 사각형은 약간 다르다. 바닥과 조명은 합성 환경이고 실제 창고의 재질, 노출, 센서 잡음, 렌즈 왜곡을 재현하지 않는다.

## 재생성

```powershell
python -m pip install moderngl trimesh imageio-ffmpeg
python camera_preview.py
python camera_preview.py --height .075 --output camera_preview_low
python camera_preview.py --options example_gaussian_options.json --height .25
```

기존 numpy, OpenCV, Pillow도 필요하다. ModernGL은 OpenGL 3.3 이상이 필요하다. 영상 생성기는 매 프레임 GPU 렌더링과 레이블 투영을 수행한 뒤 H.264 MP4로 저장한다. 출력 파일의 30FPS는 재생 속도이며 실시간 서버의 보장 FPS는 아니다. 로컬 GPU에서 측정한 렌더링 및 가상 결과 생성 시간은 `measurements.json`에 저장하며 인코딩·네트워크·브라우저 표시 시간은 제외한다.

## FSM 화면에 연결하는 방법

1. 현재 `frame.truth.x/z/yaw`와 카메라 설정을 `PalletCamera.render_relative(truth, vertical_offset)`에 전달해 RGB를 만든다. 팔레트의 월드 이동 없이 같은 상대 시점을 계산한다. 바닥까지 정확한 월드 배치를 표현하려면 팔레트와 카메라의 월드 변환 행렬을 함께 전달한다.
2. **FSM에 전달한 것과 동일한** `lift.model.v1` 패킷을 `draw_packet_overlay(rgb, packet, options)`에 전달한다. 화면용 오차를 별도로 뽑지 않아야 화면과 FSM 입력이 일치한다. 현재 화면 진단에는 이 모서리가 없으므로 기존 결과 패킷을 화면 응답에도 실어야 한다. 실제 비전 독립 상태와 검출 생략 상태도 `vision_independent`/`skip_detection`으로 전달한다.
3. RGB 30Hz, 가상 출력 10Hz, CAN 주기는 독립적으로 유지한다. `det_ok=false`이면 레이블을 지우고 결과 sequence·촬영 시각·age를 표시한다. 현재 RGB 위에 지연된 결과를 올릴지, 촬영시각에 맞는 RGB를 버퍼에서 찾아 표시할지 명시적으로 선택한다. 이 예시는 전자다.
4. OpenGL 컨텍스트는 생성한 렌더링 스레드에서 사용한다. 화면 요청마다 OBJ를 재로드하지 않고 메시·텍스처·GPU 버퍼를 유지한다. 브라우저 WebGL 방식도 가능하다.

이 방식은 제어기가 받아들이는 오차·지연·미탐과 화면 표현을 검증하기에 적합하다. 실제 인식기의 조명·질감별 성공률을 측정하는 실험은 아니다.

API 참고: [ModernGL Context](https://moderngl.readthedocs.io/en/5.8.2/reference/context.html).

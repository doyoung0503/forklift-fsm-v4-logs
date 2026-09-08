Pallet Pickup FSM Simulator - Standalone Project
================================================

이 폴더는 팔레트 정렬/삽입 FSM 브라우저 시뮬레이터만 분리한 독립 실행 패키지입니다.
원본 Blender 프로젝트, RealSense, YOLO, CAN 장비 또는 외부 서버가 필요하지 않습니다.

주요 기능
---------
- 팔레트 전방 거리, 좌우 위치, yaw 조건 시뮬레이션
- 원본 FSM 상태 전이와 실시간 다이어그램
- 카메라 FOV 및 팔레트 전면 가시성 계산
- 포크/팔레트 포켓 충돌 판정
- 27조건 고속 실험과 결과 재현
- 구동오차/모델 추정오차 분포 적합 및 seeded sampling
- FSM threshold 그리드 서치
- 3D trajectory CSV/JSON 저장과 화면 녹화

macOS 실행
-----------
권장 방법:
1. `run_local_server.command`를 더블클릭합니다.
2. 브라우저에서 http://127.0.0.1:8765/ 가 열립니다.
3. 종료하려면 열린 터미널에서 Ctrl+C를 누릅니다.

터미널에서 직접 실행:

  cd Lift_FSM_simulator_standalone_20260907
  ./run_local_server.command

간단 실행은 `run_simulator.command`를 더블클릭하면 됩니다.

Windows 실행
------------
`run_local_server.bat`를 더블클릭합니다. PowerShell 기반 로컬 서버가 시작되고
http://127.0.0.1:8765/ 가 열립니다. 간단 실행은 `run_simulator.bat`를 사용합니다.

자동 검증
---------
Node.js가 설치되어 있으면 다음 명령으로 독립 패키지 검증을 실행할 수 있습니다.

  ./run_tests.command

또는 `tests/` 안의 다섯 JavaScript 테스트를 각각 `node`로 실행할 수 있습니다.

포함 범위
---------
- 실행에 필요한 HTML, JavaScript, CSS
- 예제 구동오차/추정오차 CSV
- macOS/Windows 실행기
- 독립 실행 가능한 JavaScript 자동 테스트

제외 범위
---------
- 상위 Blender 연구 저장소 및 데이터셋
- 원본 Python/CAN/RealSense/YOLO 하드웨어 코드
- 미리보기 이미지, macOS 메타데이터, 이전 UI 백업
- 상위 저장소의 원본 Python FSM이 필요한 differential trace 테스트

주의
----
이 시뮬레이터는 FSM 결정, 타이머, 명령 순서, 확률적 오차와 구현된 충돌 기하를
검증합니다. 완전한 강체 물리 또는 실제 장비 안전성을 검증하는 도구는 아닙니다.

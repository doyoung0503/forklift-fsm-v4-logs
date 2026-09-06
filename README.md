# forklift-fsm-v4-logs

지게차 FSM v4 실측 로그 수집·분석 작업공간입니다. RealSense 깊이 카메라 녹화 로그로부터
팔레트 YOLO26 Pose 모델의 회전 응답을 적합(fit)하고, 대칭성 학습과 FSM v4 실행 가능성
분석을 수행한 결과물을 모아둔 저장소입니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `tools/` | 녹화 로그 추출·분할, FSM v4 실행 가능성 분석/시각화 스크립트 |
| `rotation_fit/` | 로그 기반 회전 적합 모델. 분석 스크립트, 적합 결과, 리포트 문서 |
| `extracted/` | FSM v4 독립 실행 묶음(`depth_cam/`, `calib/`, `ui/`)과 녹화 로그 |
| `training/symmetry/` | 팔레트 대칭성 학습 코드, 학습 런(runs), 평가 결과 |
| `datasets/` | 학습·평가용 데이터셋 (`pallet_symmetry_v1`) |
| `third_party/`, `model_archive/`, `weights/` | 배포·보관용 Pose 모델 가중치 |
| `_docs/` | 작업 히스토리(`history/`)와 분석 리포트(`analysis/`) |

## 주요 문서

- `_docs/history/2026-09-03.md` — 회전 오차 보정용 로그 기반 적합함수 작업 기록
- `_docs/analysis/fsm_v4_feasibility_20260906/README.md` — FSM v4 실행 가능성 분석
- `rotation_fit/ROTATION_MODEL_REPORT.md` — 회전 모델 리포트
- `rotation_fit/LOG_BASED_ROTATION_MODEL.md` — 로그 기반 회전 모델 설명
- `extracted/README.md` — FSM v4 독립 실행 묶음 실행 방법

## 저장소에서 제외된 항목

용량 문제로 다음 항목은 커밋하지 않았습니다 (`.gitignore` 참고).

- `*.zip` — 원본 배포 묶음 및 데이터셋 아카이브 (약 2.0GB)
- `*.mp4`, `*.avi` — 녹화 원본 및 시각화 렌더 영상 (약 7.7GB)
- `__pycache__/`, `.pytest_cache/` — 파이썬 캐시

## 환경

`extracted/README.md`에 정리된 대로 실행 PC에 Python 3, RealSense 드라이버가 필요하며
CAN 사용 시 Kvaser 패키지가 추가로 필요합니다. 재추론은 `ultralytics`를 설치한 격리
venv에서 수행합니다.

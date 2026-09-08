# calib/config.py
# 공통 설정 / 상수 / 색상 팔레트 (FSM 다이어그램 및 main_rec.py 실행 흐름 기준)

from pathlib import Path

# ===== 모델 & 감지 =====
# FSM v4에서 사용할 pose 모델을 명시적으로 고정한다. 다른 .pt 파일이
# 프로젝트 루트에 추가되어도 런타임 추론 모델은 바뀌지 않는다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILENAME = "pallet_yolo26n_pose_cleanlabel.pt"
_MODEL_FILE = PROJECT_ROOT / MODEL_FILENAME
if not _MODEL_FILE.is_file():
    raise RuntimeError(f"Configured pose model not found: {_MODEL_FILE}")
if _MODEL_FILE.stat().st_size <= 0:
    raise RuntimeError(f"Configured pose model is empty: {_MODEL_FILE}")
MODEL_PATH = str(_MODEL_FILE.resolve())
FRONT_CLASS_NAME = "item"
CONF_THR = 0.40  # Cleanlabel inference_config.yaml; same as the 48-point fit
POSE_REFLECT_PADDING_PX = 100
POSE_INFERENCE_IMGSZ = 640
POSE_INSTANCE_SELECTION = "max_box_conf"

# ===== POSE 키포인트 =====
# 전면부 외곽 4모서리 (좌상, 우상, 우하, 좌하) — 시계방향
POSE_FACE_KPTS = (0, 1, 2, 3)
POSE_BACK_KPTS = (4, 5, 6, 7)   # 후면 외곽 4모서리 — 전면과 같은 순환순서
POSE_REAR_KPTS = POSE_BACK_KPTS  # livegt 프로젝트 명칭과 호환
POSE_CENTER_KPT = 8       # 모델이 정의한 팔레트 중심점 (= 8모서리 3D 중심)
POSE_KPT_VIS_THR = 0.50   # 이 값 이상인 전면/후면/중심점만 visible PnP에 사용
PNP_RANSAC_REPROJ_ERROR_PX = 8.0
PNP_RANSAC_ITERATIONS = 100
PNP_RANSAC_CONFIDENCE = 0.99

# 6D pose 를 전면 4점만으로 풀지, 보이는 모든 키포인트로 풀지.
# 전면은 1.1 x 0.14 m (가로세로비 ~8:1) 라 평면 PnP 가 나쁘게 조건화되어
# yaw 부호가 뒤집히는 프레임이 나온다. 후면 키포인트를 넣으면 깊이 방향
# 지지를 얻어 회전 구간 yaw 상관이 0.02 -> 0.99 로 회복된다.
# 자세한 근거: _docs/history/2026-09-03.md 8절
POSE_USE_MULTI_KPT = True  # 메타데이터/구버전 호환; main_rec은 visible PnP 사용
POSE_MULTI_MIN_FRONT = 0   # visible PnP는 전면 4점 고정 조건을 두지 않는다
POSE_POLY_SHRINK = 0.12    # 평면적합용 폴리곤을 중심 방향으로 축소하는 비율
                           # (모서리 바깥 배경 depth 유입 차단)
POSE_WIDTH_MIN = 0.70      # 3D 폭 sanity 하한(m) — 벗어나면 탐지 무효
POSE_WIDTH_MAX = 1.50      # 3D 폭 sanity 상한(m)

# 파렛트 전면부 실치수(m) — depth 없이 키포인트만으로 자세를 풀 때(solvePnP) 쓰는 모델 크기
PALLET_WIDTH_M = 1.100
PALLET_LENGTH_M = 1.100
PALLET_HEIGHT_M = 0.150

# livegt 학습 라벨에 대응하는 PnP 3D 모델. 0~3은 전면, 4~7은 후면,
# 8은 몸체 중심이며, 원점은 전면 네 모서리의 중심이다.
# 전체 팔레트 실폭은 1.10 m지만 학습 라벨의 유효 전면 키포인트 폭은 1.00 m다.
# Keep the PnP keypoint model on the same physical-width source of truth.
PALLET_FACE_W = PALLET_WIDTH_M
PALLET_FACE_H = PALLET_HEIGHT_M
PALLET_BODY_D = PALLET_LENGTH_M

# ===== 카메라 스트림 =====
# True: 실제 RealSense 사용 / False: 카메라 없이 UI 테스트 (FSM 대기, CAN OFF).
# main_rec.py와 main_rec_v4.py 공통 설정. 저장 후 프로그램을 재시작하면 반영됩니다.
CAMERA_ENABLED = True
STREAM_W, STREAM_H = 640, 480
# 0 = 장치가 지원하는 최대 fps 를 자동 선택(내림차순으로 시도).
# 특정 값으로 고정하려면 그 숫자를 넣는다 (예: 30).
STREAM_FPS = 0

# ===== 성능 (추론 fps 상한 제거용) =====
# 왜곡계수가 전부 0이면 depth 역투영을 numpy 로 벡터화한다.
# (기존 경로는 프레임당 pybind11 왕복이 약 1만 회 발생)
FAST_DEPROJECT = True
# RANSAC 평면적합에 넣을 점 상한. 초과분은 균등 서브샘플.
# 측정: 2770점 20.9ms -> 1500점 14.4ms (프레임당 최대 병목)
RANSAC_MAX_POINTS = 1500
# 프레임마다 추론 로그를 찍으면 고 fps 에서 터미널 I/O 가 병목이 된다.
VERBOSE_PERCEPTION = False
# 녹화 fps 를 실측할 프레임 수 (이 구간이 지난 뒤 VideoWriter 를 연다)
REC_FPS_PROBE_FRAMES = 30

# ===== 정렬 시퀀스 로깅 (digital twin 인코딩용) =====
TRACE_ENABLE = True

# ===== FSM 시작 전 포크 초기화 =====
# 위치 센서가 없으므로 물리적 끝점의 리미트 보호를 기준으로 홈을 잡은 뒤,
# 시간 기반으로 초기 높이/리치 위치를 만든다. 모든 시간 단위는 초(s).
INITIALISATION_ENABLED = True
INITIALISATION_UPRIGHT_COMMAND = "unfold"  # "unfold" 또는 "fold"
INITIALISATION_UPRIGHT_HOME_SEC = 1.0
INITIALISATION_FOLD_SEC = 0.0
INITIALISATION_LIFT_DOWN_HOME_SEC = 6.0
INITIALISATION_LIFT_UP_SEC = 1.0
INITIALISATION_REACH_BACKWARD_HOME_SEC = 3.5
INITIALISATION_REACH_FORWARD_SEC = 2.0
INITIALISATION_SETTLE_SEC = 1.0
TRACE_STATE_HZ = 10.0     # pallet_state.csv 기록 주기(Hz). 명령 전환은 주기와 무관하게 항상 기록

# ===== 연산 디바이스 / 정밀도 =====
# CUDA가 사용 가능하고 USE_GPU=True면 'cuda:{CUDA_DEVICE}'로 동작.
# USE_HALF=True이면 FP16 추론을 시도합니다(지원되는 GPU에서 속도/메모리 이점).
USE_GPU = True
CUDA_DEVICE = 0
USE_HALF = True

# ===== 깊이 샘플링 / 강건화 =====
SAMPLE_STRIDE = 3          # 마스크 내 픽셀 stride 샘플링 간격
Z_INLIER_THRESH = 0.06     # z(inlier) 허용 오차(m)
MIN_POINTS = 120           # RANSAC에 투입할 최소 3D 포인트 개수

# ===== 평면 적합 (RANSAC) =====
PLANE_INLIER_THRESH = 0.01 # 평면 거리 허용(m)
PLANE_MAX_TRIALS = 200
# 평면적합 전용 z-클립(m). Z_INLIER_THRESH(0.06)를 그대로 쓰면 yaw가 클 때
# 전면부의 깊이 스프레드(폭 1.1m × sin(yaw))가 잘려 기울기가 사라지고 yaw≈0 으로 붕괴한다.
PLANE_Z_CLIP = 0.50
# ===== HUD 색상 (BGR for OpenCV) =====
COLOR_STATUS_OK   = (0, 220, 0)
COLOR_STATUS_TRK  = (0, 165, 255)
COLOR_ALERT       = (0, 0, 255)
COLOR_META        = (180, 180, 0)
COLOR_YAW         = (200, 100, 255)
COLOR_OFFSET      = (50, 200, 255)
COLOR_WIDTH       = (255, 170, 0)
COLOR_ZERR        = (0, 220, 255)
COLOR_BOX         = (0, 180, 255)
COLOR_CNT         = (255, 0, 255)
COLOR_CENTER      = (180, 180, 180)
COLOR_PANEL_BG    = (30, 30, 30)
COLOR_PANEL_EDGE  = (70, 70, 70)

# ===== 정렬 허용치 =====
YAW_TOL_DEG   = 2.00   # ±2 deg
OFF_TOL_M     = 0.12   # ±12 cm

# ===== 폭(가로 길이) 기준 =====
# detected_length >= WIDTH_MIN_FULL -> ALIGN
# detected_length <  WIDTH_MIN_FULL -> RECOVER
WIDTH_MIN_FULL = 0.00  # m (탐지된 파렛트 최소 길이 임계값)

# ===== ALIGN 단계: 거리 밴드 제어 =====
ALIGN_DIST_M = 2.20    # 정렬 목표 거리 (m)
ALIGN_BAND_M = 0.30    # 허용 밴드 (±m)

# ===== 정렬 완료 판정 안정화 =====
CMD_STABLE_THR = 5     # 같은 판정이 연속 n프레임 유지돼야 상태 전이

# ===== 정지 추론 스냅샷 안정화 =====
# DIST_CHECK와 POST_LATERAL_INFER에서는 아래 개수만큼 연속된 유효 추론값이
# 허용 spread 안에 들어올 때만 중앙값을 제어 입력으로 확정한다.
INFER_STABLE_FRAMES = 5
DIST_STABLE_SPREAD_M = 0.05       # 5개 거리값의 max-min <= 5 cm
YAW_STABLE_SPREAD_DEG = 2.0       # 5개 yaw값의 max-min <= 2 deg
LATERAL_STABLE_SPREAD_M = 0.02    # 5개 lateral값의 max-min <= 2 cm

# 거리 이동 후 스냅샷을 이용한 좌우 정렬 회전각.
# 파렛트 전면에 수직인 이동 방향으로 90° 회전한 뒤 lateral만큼 전진하고,
# 반대 방향으로 90° 회전하여 다시 파렛트 전면을 바라본다.
LATERAL_TURN_DEG = 90.0
ALIGN_ROTATE_TIMEOUT_SEC = 30.0

# ===== 전역 인터록(정지) =====
# 모든 제어 명령 상태 전환 사이에 STOP을 이 시간만큼 유지합니다.
STOP_SEC = 1.2

# (하위호환) 기존 코드에서 STOP_PAUSE_SEC을 참조하면 STOP_SEC을 사용하도록
STOP_PAUSE_SEC = STOP_SEC

# ===== 최초 탐지 후 파렛트 정면 정렬 =====
# 최초 탐지 후 live PnP 중심 방위각으로 회전한다.
INITIAL_FACE_STOP_SEC = STOP_SEC
INITIAL_FACE_SETTLE_SEC = STOP_SEC
INITIAL_FACE_ROTATE_TIMEOUT_SEC = 30.0

# ===== 전진시간 피팅(가속→정속) 파라미터 =====
# 로그(t_monotonic & dist_z) 기반으로 적합된 파라미터.
# d_acc = 0.5 * FWD_A * FWD_T1^2,  vmax = FWD_A * FWD_T1
# t(d) =
#   d <= d_acc:  FWD_T0 + sqrt(2d/FWD_A)
#   d >  d_acc:  FWD_T0 + FWD_T1 + (d - d_acc)/vmax
USE_PIECEWISE_FWD_FIT = True

FWD_T0   = 0.5069     # s (명령→실이동 시작 지연)
FWD_T1   = 2.0357     # s (가속 구간 지속시간)
FWD_A    = 0.139644   # m/s^2 (초기 가속도)
FWD_SCALE = 1.0       # d_eff = FWD_SCALE*|offset| + FWD_BIAS
FWD_BIAS  = 0.0

# 안전 클램프(전진 명령 타이머)
FWD_MIN_SEC = 1.0
FWD_MAX_SEC = 15.0

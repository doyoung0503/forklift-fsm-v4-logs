# 로그 기반 회전 적합 모델 (deflection 30)

이 구현은 좌·우 회전을 하나의 크기 모델로 합치고, 방향은 최종 명령을 만들 때만
부호로 적용한다. 여기서 `30`은 측정된 토크(N·m)가 아니라 CAN 중립값 127에서의
**조이스틱 편향값**이다. 실제 송신 payload의 두 번째 바이트가 좌회전 157 또는
우회전 97인지 검사하여 `abs(value - 127) == 30`인 로그만 사용한다.

기존 `fit_rotation_model.py`와 실차 FSM의 `rotation_model.py`는 STOP 지연과 등감속을
가정한 별도 모델이다. 이 문서의 새 적합기는 요청한 **정지 직전 속도 → STOP 약
2초 후 관성 회전량의 선형회귀**를 사용한다. 두 관성 모델을 동시에 더하면 관성을
이중 계산하므로, 새 모델을 실차에 적용할 때는 기존 감속 기반 관성식을 교체해야 한다.

## 입력 로그와 시각 기준

한 녹화마다 다음 파일을 사용한다.

- `*_control_seq.jsonl`: `phase=can_tx`, `event=movement_write`,
  `source=command_burst`인 첫 프레임의 실제 Kvaser `write()` 시작/반환 시각
- `*_inference_timing.csv`: 매 이미지의 `frame_i`,
  `camera_input_host_mono_ms`, `camera_sensor_timestamp_ms`, 추론 각도와 품질값

CAN 명령은 host monotonic 시각을 쓴다. 프레임은 `camera_sensor_timestamp_ms`가
단조롭고 clock rate가 정상일 때 전체 녹화의 host 시각에 robust affine 정렬하여
촬영 프레임 간격으로 속도·가속도·2초 horizon을 계산한다. 센서 시각이 없거나 reset,
domain 변경이 있으면 `camera_input_host_mono_ms`로 안전하게 fallback한다. 두 원본
시각과 실제 사용한 timebase는 결과에 모두 남는다. 이 정렬은 전달 지연의 흔들림은
줄이지만 절대 센서→host 지연은 외부 동기화 이벤트 없이 분리할 수 없으므로 정지시간에는
그 고정 지연이 포함될 수 있다. 더 정확한 기계 지연에는 별도 clock calibration 또는
CAN bus capture가 필요하다.

`write_start_host_mono_ms`와 `write_return_host_mono_ms`, 그리고
`command_sync_done`은 모두 `rotation_segments.csv`에 보존된다. 기본 적합 기준점은
write 반환 시각이며 `--can-time-point start|return|midpoint`로 바꿀 수 있다.

새 추론 모델의 결과는 원본 timing CSV를 다시 만들 필요가 없다. 녹화 basename이
같은 CSV에 `frame_i`, `yaw_deg`, 유효성 열, **모델 weight hash**를 기록한 뒤
`--results-dir`, `--results-suffix`로 지정하면 원본 시각과 join한다.
`--model-id-column`으로 지정한 값은 모든 유효 행에 반드시 있어야 한다. 모델 식별자가
전혀 없거나 서로 다른 모델 버전이면 배포하지 않는다.

적합 입력 계약은 `yaw_deg` 열과 `heading` 도메인으로 고정되어 있다. `yaw_deg`는
팔레트 면의 PnP 자세각이어야 하며, 단순히 표적 중심의 방위각을 열 이름만 바꿔서는
안 된다. `center_bearing_deg`에는 회전중심보다 0.68 m 앞에 있는 카메라의 호 운동이
포함되어 거리 의존적이므로 이 적합기는 이를 거부한다. 0.68 m는
`calib/fsm_v4/config.py`의 실측 외부 파라미터이며, 경로계획과 bearing 제어 변환에서
별도로 사용한다.

## 네 구간의 수식

명령 유지시간을 `h`, 정지시간을 `d`, 실제 구동시간을 `q=max(0,h-d)`, 각가속도를
`a`, 가속구간 길이를 `ta`, 최대속도를 `vmax=a*ta`라고 한다.

```text
W(q) = min(a*q, vmax)

D(q) = 0.5*a*q^2                    (q <= ta)
     = vmax*(q - 0.5*ta)            (q > ta)

C(W) = max(0, b0 + b1*W)            (W > 0)
       0                             (W = 0)

F(h) = D(max(0,h-d)) + C(W(max(0,h-d)))
```

- `d`: 높은 임계값으로 움직임 지속을 먼저 확인한 뒤 역방향으로 탐색한 마지막 정지
  이미지와 최초 작은 변화 이미지 사이 구간 중점의 중앙값. 최초 변화 프레임의 실제
  시각과 그 상한 지연도 별도 기록
- `a`, `ta`: 움직임 시작부터 STOP 직전 프레임까지의 각도 시계열을 robust
  segmented curve로 공동 적합
- `vmax`: `a*ta`; `ta` 이후 충분한 프레임이 있는 긴 명령에서만 식별 가능
- `C`: STOP 직전 0.6초의 robust local polynomial로 구한 속도와 STOP+2초 주변
  각도 변화의 선형회귀. 기본은 물리 경계 `C(0)=0`인 `b1*W`이며,
  `--fit-inertia-intercept`를 주면 비음수 `b0+b1*W`를 적합한다.

목표각 역산은 `F(h)=abs(target)`의 단조 역함수를 계산한다. 양수 목표는 RIGHT,
음수 목표는 LEFT로만 바뀌고 유지시간은 같다. 관측된 명령시간/각도 범위를 벗어나면
기본 API는 `feasible=False`를 반환하며 명시적인 extrapolation 없이는 운전값으로
사용하지 않는다.

## 새 모델 자동 적용 파이프라인

FSM v4를 `main_rec_v4.py`로 시작하면 카메라나 CAN을 초기화하기 전에 다음 순서를
자동 실행한다.

1. `extracted/`의 유일한 `.pt` 모델을 선택하고 SHA-256으로 고정한다.
2. 각 `*_raw.mp4`를 새 모델로 다시 추론해 `yaw_deg`/`heading` CSV를 만든다.
3. 영상-`frame_i` 정렬, 유효 pose, 모델 hash, 각도 도메인을 검증한다.
4. 실제 CAN write-return 시각과 촬영 시각을 결합해 정지시간·가속·등속·관성을 적합한다.
5. recording 단위 LOO RMSE, 식별 가능성, 좌우 부호, 토크 편향 30, 모델 hash를
   오프라인 배포기와 FSM 런타임 검증기가 각각 다시 검사한다.
6. 입력 CSV가 적합 도중 바뀌지 않았는지 SHA-256으로 재검사한 뒤
   `calib/fsm_v4/rotation_model.generated.json`을 원자적으로 교체한다.
7. 부모 FSM 프로세스가 artifact와 `.pt` hash를 다시 확인하고, 일치할 때만 카메라/CAN
   실행을 시작한다.

같은 모델 hash의 유효 artifact가 이미 있으면 2~6단계는 즉시 건너뛴다. 새 모델이거나
artifact가 없으면 전체 재추론부터 수행한다. 어느 단계든 실패하면 기존 artifact는
바이트 단위로 보존되고 FSM은 시작되지 않는다. 실행 중 hot-swap은 하지 않으며,
검증된 파일은 배포 이후 시작되는 FSM 프로세스에서만 활성화된다. 각 시도의 입력,
제외 사유, 단계와 오류는 `rotation_fit/out/automatic/<run-id>/`에 남는다.

수동으로 같은 전체 파이프라인만 실행하려면 작업공간 루트에서 다음 명령을 쓴다.

```powershell
py -3 rotation_fit/auto_calibrate_rotation.py
```

다시 적합해야 할 때만 `--force`를 붙인다. `.pt` 파일이 없거나 두 개 이상이면 잘못된
모델을 고르지 않도록 시작 전에 실패한다.

## 실행

현재 timing CSV 안의 안정된 heading 결과를 사용할 때:

```powershell
py -3 rotation_fit/fit_piecewise_rotation_model.py `
  --recordings-dir extracted/depth_cam/rec `
  --angle-column yaw_deg `
  --angle-domain heading
```

새 모델이 예를 들어
`new_inference/<recording>_new_pose.csv`에 `frame_i,yaw_deg,pose_ok,model_hash`
형식으로 결과를 만들었을 때:

```powershell
py -3 rotation_fit/fit_piecewise_rotation_model.py `
  --recordings-dir extracted/depth_cam/rec `
  --results-dir new_inference `
  --results-suffix _new_pose.csv `
  --angle-column yaw_deg `
  --angle-domain heading `
  --valid-column pose_ok `
  --model-id-column model_hash
```

필요 패키지는 다음과 같다.

```powershell
py -3 -m pip install -r rotation_fit/requirements.txt
```

산출물은 기본적으로 `rotation_fit/out/piecewise_linear_inertia/`에 생성된다.

- `rotation_segments.csv`: 명령 시작/종료, 마지막 정지 프레임, 최초 움직임 프레임,
  촬영 센서 시각, host 정렬 시각, 정지시간, STOP 직전 속도, 2초 후 각도,
  관성 회전량, 제외 사유
- `fit_report.json`: 파라미터, 식별 가능성, recording 단위 cluster bootstrap 구간,
  leave-one-recording-out 오차, 좌/우 잔차·관측 부호 점검, 배포 차단 사유
- `rotation_model.json`: 모든 구간이 식별된 경우에만 생성되는 런타임 artifact
- `command_table.csv`: 관측 범위 안의 목표각→명령 유지시간 표

`rotation_model.json`은 다음처럼 바로 사용할 수 있다.

```python
from pathlib import Path
from rotation_fit.piecewise_rotation_model import LogRotationModel

model = LogRotationModel.from_json(Path("rotation_model.json").read_text("utf-8"))
plan = model.command_duration(-8.0)  # LEFT 8 deg
if plan.feasible:
    print(plan.direction, plan.hold_sec)
```

## 자동 제외와 식별 실패

다음 로그는 수치가 있어도 적합에서 제외하거나 별도 표시한다.

- 실제 CAN write 경계가 없거나 실제 편향이 30이 아님
- 명령 전 1초 동안 정지하지 않았거나 기준 각도 잡음/드리프트가 큼
- 한 점 spike가 아니라 0.25초 지속되는 움직임 시작을 찾지 못함
- 마지막 정지 프레임과 최초 움직임 프레임 사이의 큰 누락
- STOP 직전 속도 계산 프레임 부족
- STOP+2초 주변 프레임 부족 또는 그 전에 다른 동작 명령이 시작됨
- 추론 품질 실패, 서로 다른 model id/hash 혼합

최대속도를 주장하려면 가속 종료 뒤 최소 0.25초 이상 등속 구간을 포함한 독립 run이
2개 이상 필요하다. 관성 기울기를 주장하려면 STOP 속도가 서로 다른 독립 run이
최소 3개 필요하다. 조건이 부족하면 `fit_report.json`은 남기되
`rotation_model.json`을 만들지 않는다. 이는 외삽 파라미터가 실차 제어에 들어가는
것을 막기 위한 의도된 동작이다.

추가로 서로 다른 model id가 섞였거나 일부 녹화를 읽지 못했거나, LEFT/RIGHT 관측
부호가 모순되거나, leave-one-recording-out fold 중 하나라도 적합되지 않거나 성공 fold가
4개 미만이거나 RMSE가 기본 3도보다 크면
상태는 `DIAGNOSTIC_ONLY`가 되고 런타임 artifact를 만들지 않는다. 정확도 기준은
`--min-loo-recordings`, `--max-loo-rmse-deg`로 실차 허용오차에 맞게 조정할 수 있다.
`--allow-mixed-model-ids`는 원인 분석용 계산만 허용하며 배포 우회 옵션이 아니다.
재실행을 시작하면 같은 출력 폴더의 이전 산출물 4개를 먼저 제거한다. 따라서 입력 오류나
게이트 실패 뒤에 과거 `rotation_model.json`이 최신 결과처럼 남지 않는다.

권장 수집은 명령 전 1초 이상 정지, STOP 후 2.5초 이상 무명령, 짧은/중간/등속 도달
유지시간을 각각 여러 번 반복하는 것이다. 좌·우 데이터는 모델 적합에는 합치지만,
방향별 bias를 검증하려면 양쪽 모두 균형 있게 수집해야 한다.

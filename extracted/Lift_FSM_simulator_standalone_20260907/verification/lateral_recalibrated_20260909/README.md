# 변경된 로직의 레터럴 재측정 및 FSM 적용

2026-09-09. CAN 타이머 회전 STOP, 근거리 재계획, adaptive 전진 길이를
포함한 현재 로직을 다시 측정하고 초기보정 진입 기준에 적용했다.
초기보정 실행 순서는 기존 IMU 회전 → 횡이동 → 반대 90도 회전을 유지한다.

## 적용 수식

```python
D = -pose.rot_z_pallet_m       # 팔레트 전면에 수직인 차량 회전중심 거리, m
L = abs(pose.rot_x_pallet_m)   # 팔레트 좌표계 횡오차, m
limit = 0.10 + 0.190 * max(0.0, min(D, 5.0) - 2.26)
initial_correction_needed = L > limit
```

초기보정이 활성화되어 있고 아직 수행하지 않은 경우에 적용한다. 기준과 같으면
일반 접근을 선택한다. yaw는 진입 기준에서 제외했고 IMU 회전 목표 계산에는
계속 사용한다. 비정상 포즈, 회전 공간 부족, IMU 및 동작 제한 검사는 유지한다.

검증 범위는 D=3~5m다. 5m 밖에서는 기준을 0.6206m로 고정해 미측정 영역으로
허용량을 늘리지 않는다. 3m 미만에서는 남은 접근 거리에 따라 기준을 줄이고,
staging 거리 이하에서는 0.10m로 고정한다. 이 근거리 확장식은 실측 경계가
아니며, 초기보정에 필요한 최소 공간 검사를 통과해야 실제 동작한다.

## 측정과 여유

좌우 결과가 아래 정밀도에서 일치했다. '안쪽 성공 경계'는 이진탐색과 내부
표본검사로 확인한 값이며 연속 영역 전체에 대한 성공 보장은 아니다.

| D (m) | 안쪽 성공 경계 (m) | 인접 실패 (m) | 적용 기준 (m) |
|---:|---:|---:|---:|
| 3.0 | 0.276172 | 0.281250 | 0.240600 |
| 3.5 | 0.418750 | 0.425000 | 0.335600 |
| 4.0 | 0.650000 | 0.656250 | 0.430600 |
| 4.5 | 0.856250 | 0.862500 | 0.525600 |
| 5.0 | 0.848672 | 0.855625 | 0.620600 |

기본 허용량 b=0.10m와 staging 거리 2.26m를 고정했다.
`k_envelope = min((안쪽 성공 경계 - b)/(D - 2.26)) = 0.2380701014`.
보정 가능 증가분의 80%를 사용하고 소수 셋째 자리로 내림해 k=0.190을 선택했다.
최소제곱 계수 0.296404는 일부 측정 경계를 초과하여 채택하지 않았다.
20% 여유는 공학적 선택이며 실제 구동오차의 통계적 신뢰수준을 뜻하지 않는다.

3m에서는 L=0.28125m가 충돌 판정이지만 0.30m가 성공했고, 5m에서는
0.855625m가 충돌 판정인데 1.0125m가 성공했다. 따라서 큰 성공 구간을
허용 범위로 사용하지 않고 먼저 관측된 안쪽 실패 경계를 재탐색했다.
실패 판정에는 FSM 실패뿐 아니라 시뮬레이터 충돌·잔여 삽입 거리도 포함했다.

## 실험 조건과 검증

- 검색/내부 감사 201회. 거리별 양쪽 부호, 지수적 구간 확장 후 이진탐색,
  25/50/75/90% 내부 표본 확인. 최종 경계 폭 약 5.1~7.0mm.
- 초기보정 OFF, 추가 구동·회전·인식·IMU 오차 0, 전진 선행 STOP 보정 0.
  인식 30Hz, CAN 타이머 STOP 사용. 명목 거리/회전 응답 모델은 유지했다.
- 팔레트 중심을 바라보는 초기 방향, 회전중심 기준 D와 L로 배치.
  카메라 높이 0.50m, 가로 FOV 55도, 포크 한쪽 폭 0.115m.
- FSM은 nine-block 삽입 검사를 사용하고, 독립 시뮬레이터 충돌 검사는 기존
  continuous-opening 형상을 사용한다. 이 두 검사 모두를 통과해야 성공이다.
- 후보식 추가 검증 44/44 성공: 0.25m 간격 9개 거리×좌우 18회,
  절반 레터럴 10회, 초기 방향을 전면과 평행하게 둔 6회,
  기존 전진 선행 STOP 설정을 유지한 10회.
- 적용 후 실제 FSM+가상 IMU+CAN으로 20/20 성공: 5개 거리×좌우×기준의
  98%/105%. 98%에서는 초기보정 없이, 105%에서는 초기보정 후 삽입 완료.
- 관련 단위/회귀 테스트 126개 통과. 실차 시험은 수행하지 않았다.

## 구현과 재현

설정: `depth_cam/calib/fsm_v4/config.py`의 `COARSE_LATERAL_*`.
판단: `coarse.py`의 `coarse_lateral_limit_m`과 `_coarse_required`.
초기보정 trace에 D, L, 계산된 limit와 `distance_adaptive_lateral` 기준을 남긴다.
`COARSE_*` 값은 녹화 metadata에도 저장된다. 프로그램 재시작 또는 시뮬레이터의
새 세션이 필요하다.

시뮬레이터 폴더에서 실행한다. 코드 변경 후에는 새 출력 폴더를 사용해야 한다.

```powershell
python calibrate_lateral.py --disable-predictive --output verification/lateral_recalibrated_20260909
python summarize_lateral.py --output verification/lateral_recalibrated_20260909 --validation-output verification/lateral_recalibrated_20260909_candidate --policy-output verification/lateral_recalibrated_policy_20260909
python verify_adaptive_coarse.py
```

측정은 초기보정 조건을 적용하기 전 source_id
`c43bed3dd543fd920d0a88a1bb6ee88e31e57f703f2fa19484cd8d2687e09f03`로 완료했다.
적용 후 검증 source_id는
`40d6b2795d3211706ebf73e179f68cf6f64f4d1791b4a2dad6a942d7b4f70cc4`다.
`fit.json`은 적용 직전의 피팅 기록이며 적용 결과는 별도 폴더에 보존했다.

[전체 피팅/검증](fit.json) · [검색 경계](boundaries.json) · [검색+후보 검증 CSV](trials.csv) ·
[그래프](lateral_threshold.png) · [적용 후 20회 결과](../adaptive_coarse_applied_20260909/summary.json)

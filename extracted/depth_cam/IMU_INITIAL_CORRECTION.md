# IMU 초기 보정

실행: 이 디렉터리에서 `python main_rec_v4.py`. 기존 모델/장치 시작 검증을 거쳐 자동으로 실행된다. 실제 CAN 동작 설정은 기존 `CAN_ENABLED`를 따른다.

6단계: Acquire Pallet → Initial Correction → Set Standoff → Plan Approach → Execute Approach → Insert Forks.

초기보정 진입은 방향각 대신 거리별 레터럴 기준으로 판단한다. `D=-rot_z_pallet_m`, `L=abs(rot_x_pallet_m)`에 대해 `L > 0.10 + 0.190 * max(0, min(D, 5.0)-2.26)`이면 초기 보정을 한 번 수행한다. D는 카메라 Z가 아닌 팔레트 전면 수직 회전중심 거리(m)다. yaw는 회전 목표 계산에 사용한다. 설정은 `calib/fsm_v4/config.py`의 `COARSE_LATERAL_*`이며, [3~5m 재측정·검증 결과](../Lift_FSM_simulator_standalone_20260907/verification/lateral_recalibrated_20260909/README.md)에 근거한다.

1. 정지 상태에서 1.5초간 gyro Y 영점 편향을 추정한다.
2. 회전 중심의 팔레트 좌표 lateral 부호를 이용해 전면과 평행하고 중심선 쪽으로 향하는 회전각을 계산한다. 오른쪽 lateral이면 `yaw - 90°`, 왼쪽이면 `yaw + 90°`이다.
3. 자이로 누적 상대각이 목표에 도달하면 STOP하고 안정화를 기다린다.
4. `forward_seconds(abs(rot_x_pallet_m))` 동안 FWD 명령을 수행한다. 거리는 시간 모델의 추정값이며 IMU로 이동거리를 측정하지 않는다.
5. 정지 후 첫 회전과 반대 방향으로 상대 90°를 IMU로 회전한다. 이 목표는 두 번째 회전 시작 당시 실제 각도 기준이다. 88° 선행 정지나 추가 회전 보정은 적용하지 않는다.
6. 정지한 상태에서 팔레트 PnP를 다시 확보하고 기존 FSM에 복귀한다. 이후 재인식으로 초기 보정이 반복되지 않는다.

RealSense gyro/accel 스트림을 추가하고 콜백에서 자이로 샘플을 적분한다. 영상은 최신 frameset 큐로 전달한다. 초기 보정의 회전·전진 구간은 추론을 수행하지 않는다. DEBUG_STEP_MODE 사용 시 COARSE_PREPARE에서 SPACE 한 번으로 초기 보정 동작 전체를 수행하고 COARSE_REACQUIRE에서 멈춘다.

기본 제한: lateral 0.02~3m, 정면 yaw 절댓값 90° 미만, 회전 중심의 전면 수직 거리가 STAGING_DISTANCE_M 이상. 마지막 조건은 포크 길이를 고려한 최소 거리 조건이며 전체 차체/주변 장애물 충돌 검사는 아니다. IMU 샘플 0.25초 지연, 명령 갱신 0.30초 중단, 회전 시간 초과, 잘못된 회전 방향, 전진 중 5° 초과 방향 변화, 첫 회전 정착 오차 5° 초과, 재인식 실패 시 정지한다. CAN 송신 스레드도 초기 보정 명령 만료를 확인한다.

장비 검증 필요: gyro Y 부호/장착축, 정지 편향과 진동, 회전 후 관성, 현재 조이스틱 강도에 대한 전진 거리-시간 오차. 단위 테스트는 실제 회전 정확도나 장비 안전성 검증을 대신하지 않는다.

검사: `python -m unittest discover -s tests -p 'test_coarse_imu.py'`

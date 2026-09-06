# CAN 명령 구간 전용 추론 데이터

현재 FSM 최종 선택은 **Cleanlabel + 48점 정지시간/선형 endpoint 함수**입니다. 새 모델/함수의 실행 방법과 보호 조건은 [최종 회전보정 설정](FINAL_ROTATION_RESPONSE.md)을 보세요. 아래 자동 물리 구간 적합 명령은 별도 과거 파이프라인이며 현재 선택한 endpoint 함수를 자동 갱신하지 않습니다.

회전 적합에는 긴 정지 영상 전체가 필요하지 않습니다. 실제 CAN 회전 명령(세기 30)을 기준으로 명령 전 1.5초부터, 명령 유지 구간 전체와 종료 후 2.7초까지 보존합니다. 이는 기존 적합기의 기준 구간(전 1초 / STOP 후 2초 ±0.2초)에 양쪽 0.5초 여유를 더한 값입니다. 겹친 구간은 한 번만 저장합니다. 명령 사이의 긴 정지 구간은 제외하지만 프레임 간격을 줄이는 다운샘플링은 하지 않습니다.

선별은 CAN 로그와 프레임 시각만 사용합니다. 기존 모델의 검출 성공 여부나 yaw 값으로 고르지 않으므로 다음 모델에도 동일한 입력을 사용할 수 있습니다. 명령 뒤 다음 움직임이 끼어들었는지, 기준 구간에서 실제 정지했는지, 데이터가 적합에 충분한지는 기존 적합기가 검증합니다. 명령 종료 경계가 없는 회전은 분리 대상이 아닙니다.

## 저장 형식과 시간

- `rotation_fit/out/command_clips/*_raw.avi`: FFV1 무손실 영상. 원본 디코딩 픽셀과 출력 전체 프레임 해시를 대조합니다.
- `*_inference_timing.csv`: 전체 원본 행과 촬영 시각을 보존합니다. `raw_video_frame_index`는 분리 영상의 프레임 위치이고 `source_raw_video_frame_index`는 원본 영상 위치입니다. 제외 프레임은 새 위치를 비워 둡니다. 원본 `frame_i`는 바꾸지 않습니다.
- `*_control_seq.jsonl`, `*_meta.json`: 원본 CAN 기록과 카메라 실측 정보를 복사합니다.
- `command_clips_manifest.json`: 선택 범위, 원본 해시, 제외 사유, 프레임 수, 무손실 검증 결과입니다. `COMPLETE`일 때 준비가 끝난 데이터입니다.

잘라 붙인 영상의 재생 시간/FPS로 지연·가속·관성을 계산하면 안 됩니다. 기존 추론기는 명시적 프레임 매핑을 읽고, 적합기는 `frame_i`로 원본 시각에 결합합니다. 센서 시각과 호스트 시각의 차이는 기존 적합기의 시간 정렬 절차가 처리합니다. 선택 여유는 일반적인 지터를 위한 값이며, 큰 시계 오류나 누락 프레임을 복구하는 것은 아닙니다.

## 새 모델로 추론

작업공간 루트에서 `--model`만 새 파일 경로로 바꾸면 됩니다. 이 명령은 분리된 영상만 추론하고 결과 로그를 만듭니다.

```powershell
py -3 rotation_fit/batch_infer_rotation_logs.py --model extracted/NEW_MODEL.pt --recordings-dir rotation_fit/out/command_clips --results-dir rotation_fit/out/command_clips_inference
```

검증·적합·검증 후 적용까지 실행할 경우:

```powershell
py -3 rotation_fit/auto_calibrate_rotation.py --model extracted/NEW_MODEL.pt --recordings-dir rotation_fit/out/command_clips
```

자동 적용 명령은 런타임과 동일하게 `extracted` 최상위에 선택할 `.pt`가 하나여야 합니다. 품질 검증을 통과한 경우에만 적용됩니다. 기존 실행의 기본 녹화 경로는 유지되므로, 분리본을 사용하려면 위의 `--recordings-dir`을 지정해야 합니다. 이번 준비 작업은 모델 추론이나 제어값 적용을 실행하지 않습니다.

## 새 녹화 추가 시 다시 준비

```powershell
py -3 rotation_fit/prepare_command_clips.py --dry-run
py -3 rotation_fit/prepare_command_clips.py --output-dir rotation_fit/out/command_clips_next
```

출력 경로가 이미 있으면 덮어쓰지 않습니다. 새 경로로 생성한 다음 추론 명령의 경로도 바꿔 주세요. 분리 과정에서 중단되면 숨김 임시 폴더를 남기고 완성 데이터로 게시하지 않습니다.

추론 횟수 감소율이 실제 실행 시간 감소율과 같지는 않습니다. 영상 디코딩, 무손실 파일 읽기, 모델 초기화, PnP와 함수 적합 시간이 별도로 필요합니다.

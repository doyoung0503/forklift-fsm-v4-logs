# 로컬 C5 추론 준비

## 다운로드한 파일

출처: https://huggingface.co/CanelE452/pallet-pose-yolo26n-c4

고정 리비전: `2401953a1a58cccf378cf0b1f2efa53dc226a4ec`

- `pallet_yolo26n_pose_c4.pt`
- `README.md`
- `inference_config.yaml`
- `C4_PERMUTATIONS.json`
- `SHA256SUMS`

Hugging Face CLI로 위 5개 파일을 내려받고 `hf cache verify`로 검증했습니다.
학습 결과 CSV, 학습 인수, 평가 결과, `.gitattributes`는 추론에 필요하지 않아 제외했습니다.
공개 PT와 `extracted/pallet_yolo26n_pose_c5.pt`는 바이트 단위로 동일합니다.
SHA256: `38c8bc1a2fc128f320af83e6edb709d083dd168e6883649aad176f6ddc1a7a76`.

## 누락된 사용자 정의 코드와 호환본

공개 저장소에는 `pallet_yolo_loss` Python 소스가 없습니다.
따라서 원본 PT를 다시 다운로드하는 것만으로는 `ChallengeC4PoseModel` 로딩 오류가 해결되지 않습니다.
게시자의 README와 추론 설정에 C4는 학습 손실에만 사용되고 추론은 표준 YOLO와 동일하다고 명시되어 있어,
그 계약에 근거해 **추론 전용** 표준 클래스 호환본을 별도로 생성했습니다.

사용할 파일: [pallet_yolo26n_pose_c5_inference.pt](../../training/symmetry/models/pallet_yolo26n_pose_c5_inference.pt)

변환 스크립트: [export_hf_c5_inference.py](../../training/symmetry/export_hf_c5_inference.py)

검증 기록: [호환본 메타데이터](../../training/symmetry/models/pallet_yolo26n_pose_c5_inference.json)

- 원본 YAML의 표준 PoseModel에 모든 state_dict 항목을 strict 로드했습니다.
- 직렬화된 그래프와 재구성한 표준 그래프의 모듈 유형이 일치했습니다.
- 고정 테스트 입력의 두 표준 그래프 출력 차이는 0이었습니다.
- 저장 후 모든 state_dict 텐서의 dtype과 값이 원본과 정확히 일치했습니다.
- 일반 torch.load 및 Ultralytics YOLO 로딩과 실제 영상 추론을 확인했습니다.

이것은 누락된 원래 사용자 정의 Python 구현과 직접 출력을 대조한 검증은 아닙니다.
원래 학습 손실 코드를 재구현하지 않았으므로 이 파일로 일반 `.train()`을 호출하면 C4 학습 손실은 복원되지 않습니다.
원본 C5, v4, 기존 C4 실험 결과와 FSM 기본 모델은 변경하지 않았습니다.

## 추론 예시 (작업 폴더 루트에서 실행)

검증 환경: Python 3.12, torch 2.7.1+cu118, ultralytics 8.4.138.

```python
import cv2
from ultralytics import YOLO

model = YOLO("training/symmetry/models/pallet_yolo26n_pose_c5_inference.pt")
frame = cv2.imread("frame.png")
padded = cv2.copyMakeBorder(frame, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
result = model.predict(padded, imgsz=640, conf=0.4, verbose=False)[0]
if len(result.boxes):
    best = int(result.boxes.conf.argmax())
    points = result.keypoints.data[best].cpu().numpy().copy()
    points[:, :2] -= 100  # 패딩 이전의 원본 영상 좌표
```

패딩은 게시자가 필수로 명시한 조건입니다. PnP를 적용하지 않은 표시에서도 좌표에서 100을 빼야 합니다.

## 요청한 3개 영상 비교

```powershell
python training/symmetry/render_all_rec.py --candidate training/symmetry/models/pallet_yolo26n_pose_c5_inference.pt --candidate-name c5 --padding 100 --conf 0.4 --selection max_conf --out training/symmetry/evaluation/shaky3_v4_vs_c5_20260906 --only forklift_v4_recording_20260903_192254_raw forklift_v4_recording_20260903_192417_raw forklift_v4_recording_20260904_190700_raw
```

기존 출력 폴더는 덮어쓰지 않습니다. 재실행할 때는 새로운 `--out` 폴더를 지정하세요.
왼쪽 v4, 오른쪽 C5이며 양쪽 모두 padding=100, conf=0.4, 최고 신뢰도 박스 선택을 사용합니다.
이전 v4/C4 전체 비교는 padding=0, conf=0.3, 박스 비율 우선 선택이었으므로 조건이 다릅니다.
0~8번 원시 예측을 표시하며 PnP·시간 평활화·전면부 재지정을 적용하지 않습니다.

[비교 영상 목록](../../training/symmetry/evaluation/shaky3_v4_vs_c5_20260906/INDEX.md)

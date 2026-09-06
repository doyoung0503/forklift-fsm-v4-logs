# cleanlabel 모델 다운로드 및 비교

- 저장소: https://huggingface.co/CanelE452/pallet-pose-yolo26n-cleanlabel
- 고정 revision: `c40e610f46331fc385a7ec6b0ff41876142c075c`
- 모델: `pallet_yolo26n_pose_cleanlabel.pt`
- SHA256: `4ee578e02810caae56b786c23a084121f4dfb33941e11d4d3fb216b9bfbbe60e`
- HF CLI로 모델, README, inference_config.yaml, C4_PERMUTATIONS.json, SHA256SUMS를 다운로드하고 `hf cache verify`로 검증했습니다. 학습 결과 파일 등 나머지 원격 파일은 다운로드하지 않았습니다.
- 표준 Ultralytics `PoseModel`, keypoint shape `[9, 3]`으로 직접 로드됩니다. 호환성 변환이나 모델 수정은 하지 않았습니다.

## 비교 조건

v4 / 로컬 파인튜닝 C4 / cleanlabel 순서로 나란히 표시합니다. 여기서 C4는 `training/symmetry/models/pallet_yolo26n_pose_v4_c4_ft_selected.pt`이며, 이전에 C5로 지칭한 Hugging Face C4 파일이 아닙니다.

저장소의 추론 설정에 맞춰 세 모델 모두 원본 BGR 영상에 사방 100px `BORDER_REFLECT_101` 패딩을 넣고 `imgsz=640`, `conf=0.4`로 추론합니다. 최고 신뢰도 검출 하나를 선택하고 패딩을 뺀 좌표를 원본 위에 그립니다. 키포인트 0~8과 연결선을 직접 표시하며 PnP, 전면부 재지정, 시간 평활화는 적용하지 않습니다. 아래쪽 확대 영역은 세 모델에 동일합니다.

원본 영상 전체 1,321프레임과 원본 FPS를 유지합니다. FSM 설정이나 실행 기본 모델은 변경하지 않습니다.

## 재현 명령 (프로젝트 루트, 새 출력 폴더 사용)

```powershell
python training/symmetry/render_all_rec.py --out training/symmetry/evaluation/shaky3_v4_c4_cleanlabel_20260906 --third-model third_party/pallet_pose_yolo26n_cleanlabel/pallet_yolo26n_pose_cleanlabel.pt --third-name cleanlabel --padding 100 --conf 0.4 --selection max_conf --only forklift_v4_recording_20260903_192254_raw forklift_v4_recording_20260903_192417_raw forklift_v4_recording_20260904_190700_raw
python training/symmetry/verify_full_video_decode.py training/symmetry/evaluation/shaky3_v4_c4_cleanlabel_20260906
```

비교 영상과 `INDEX.md`, 모델별 경로·SHA256·검증 정보를 담은 `status.json`은 위 출력 폴더에 저장됩니다. 과거 비교 영상과 기본 2모델 렌더링 기능은 유지됩니다.

# Gaussian Splatting Research

이 문서는 `graphdeco-inria/gaussian-splatting` 기반으로 진행한 연구 내용을 정리한 문서입니다.

## 1) 연구 확장 개요

- ROS2 기반 데이터 수집 파이프라인 추가
- Edge-aware loss 실험 옵션 및 디버그 도구 추가

## 2) 업로드 권장 파일(내 변경분)

### 새로 추가한 파일

- `collect_ros2_3dgs.py`
- `gs_collector_params.yaml`
- `docs/ROS2_DATA_COLLECTION_KO.md`
- `docs/edge_study12_method2_vs_method3.md`

### 기존 코드에서 수정한 파일

- `train.py`
- `utils/loss_utils.py`

## 3) 최소 재현 예시

```bash
python make_blender_split.py -s datasets/scene_01/session_005/export_3dgs --hold 8
python train.py -s datasets/scene_01/export_3dgs_s2 -m output/edge_exp --disable_viewer --iterations 3000
python tools/export_loss_images.py -m output/edge_exp --iteration 3000 --cam_index 0
```

## 4) 라이선스 및 출처

- 원본 프로젝트: https://github.com/graphdeco-inria/gaussian-splatting
- 본 저장소의 기반 코드는 원본 프로젝트 라이선스를 따릅니다.
- 본 문서의 변경 목록은 연구 확장/실험 재현 목적의 추가 코드 중심으로 정리되었습니다.

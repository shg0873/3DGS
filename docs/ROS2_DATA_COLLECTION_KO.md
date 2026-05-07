# ROS2 기반 3DGS 데이터 수집 가이드

`collect_ros2_3dgs.py`는 ROS2 토픽에서 RGB/CameraInfo/TF/Depth를 받아 `transforms.json` 형식으로 저장합니다.

생성 파일(현재 기준):
- `images/*.png`
- `depths_metric/*.(npy|png)`
- `transforms.json`
- `camera_intrinsics.json`
- `report.json`

`report.json`은 현재 `flags`만 기록합니다.

## 1) 실행

```bash
/usr/bin/python3 collect_ros2_3dgs.py --ros-args --params-file gs_collector_params.yaml
```

기본 설정 파일:
- `gs_collector_params.yaml`

## 2) 수집 시작/종료

시작:
```bash
ros2 topic pub --once /gs_collector/command std_msgs/msg/String \
  "{data: '{\"to\":\"gs_collector\",\"data\":\"start\",\"metadata\":{\"task\":\"scene_capture\"}}'}"
```

종료:
```bash
ros2 topic pub --once /gs_collector/command std_msgs/msg/String \
  "{data: '{\"to\":\"gs_collector\",\"data\":\"stop\"}'}"
```

상태 토픽:
- `state_topic` (기본 `/gs_collector/state`)

## 3) 주요 파라미터

- `output_root`: 세션 루트 경로
- `session_name`: 세션 이름(비우면 시간 기반)
- `world_frame`, `camera_frame`: TF 기준 프레임
- `image_topic`: RGB raw 토픽 (`sensor_msgs/Image`)
- `camera_info_topic`: CameraInfo 토픽
- `target_fps`: 저장 FPS
- `sync_tolerance_ms`: TF lookup timeout
- `camera_convention`: `opengl` | `opencv` | `ros`
- `use_latest_tf`: 최신 TF 사용 여부
- `tf_retry_count`, `tf_retry_wait_ms`: TF 재시도
- `motion_gate_enabled`: 정지 상태에서만 저장
- `motion_topic`: `nav_msgs/Odometry`
- `linear_speed_threshold_mps`, `angular_speed_threshold_rps`: 정지 임계값
- `settle_time_ms`: 정지 후 대기 시간
- `require_motion_topic`: 모션 미수신 시 저장 차단
- `depth_enabled`: depth 사용 여부
- `depth_topic`: depth 토픽 (`sensor_msgs/Image`)
- `depth_sync_tolerance_ms`: RGB-depth 동기 허용 오차
- `depth_min_m`, `depth_max_m`: 유효 depth 범위(m)
- `require_depth_topic`: depth 미수신 시 저장 차단
- `depth_output_dir`: metric depth 저장 폴더명 (예: `depths_metric`)
- `depth_buffer_size`: depth 동기화 버퍼 길이
- `depth_raw_format`: metric depth 저장 형식 (`npy` 또는 `png16`)

## 4) Depth 저장 정책 (중요)

현재 수집기는 **원본 metric depth만 저장**합니다.

- `depth_raw_format: npy`
  - `float32` meters로 저장
  - 예: `depths_metric/000123.npy`
- `depth_raw_format: png16`
  - `uint16` millimeters로 저장
  - 예: `depths_metric/000123.png`

참고:
- `transforms.json`의 각 frame에는 `depth_path`가 기록됩니다.
- `depths_gray`, `depths_vis`는 수집기가 직접 만들지 않고 변환 스크립트로 생성합니다.

## 5) metric depth를 그레이스케일/시각화로 변환

추가 스크립트:
- `convert_metric_depth.py`

기능:
- 입력(metric depth: `.npy` 또는 `.png`)을
  - `depths_gray/*.png` (uint16 grayscale)
  - `depths_vis/*.png` (컬러맵 시각화)
  로 변환

실행 예시:
```bash
python convert_metric_depth.py \
  --input-dir datasets/scene_01/session_005/export_3dgs/depths_metric \
  --gray-dir datasets/scene_01/session_005/export_3dgs/depths_gray \
  --vis-dir datasets/scene_01/session_005/export_3dgs/depths_vis \
  --min-m 0.1 \
  --max-m 10.0
```

주의:
- `depths_vis`는 사람이 보기 위한 시각화용입니다(학습용 아님).

## 6) 학습/분할/렌더링

주의:
- `/usr/bin/python3` 대신 가상환경의 `python`을 사용하세요.
- 예: `(3DGS)` 환경 활성화 후 `python train.py ...`

### 6-1) 분할 파일 생성 (`transforms_train.json`, `transforms_test.json`)

```bash
python make_blender_split.py -s datasets/scene_01/session_005/export_3dgs --hold 8
```

### 6-2) 학습 (Depth 사용)

```bash
python train.py -s "datasets/scene_001/session_005/export_3dgs" -d depths_metric -m "output/session_005_depth" --eval --disable_viewer
```

### 6-3) 학습 (Depth 미사용)

```bash
python train.py -s "datasets/scene_01/export_3dgs" -m "output/scene_01_nodepth" --eval --disable_viewer
```

### 6-3-1) 학습 (Depth 미사용 + Edge 손실 추가)

```bash
python train.py -s datasets/scene_01/export_3dgs_s2 -m output/edge_study3_repeat/edge_fast_b_r1 --iterations 5000 --test_interval 250 --save_interval 250 --disable_viewer --edge_cov_start_iter 2000 --edge_cov_warmup_iters 500 --edge_cov_weight_target 0.02
```

설명:
- `--edge_l1_weight 0.03`: GT edge 구간에 가중된 L1 손실
- `--edge_grad_weight 0.05`: 예측/GT gradient 차이 손실
- `--edge_mask_alpha 2.0`: edge 마스크 강도

### 6-4) 렌더링

테스트셋 렌더링:
```bash
python render.py -m output/session_005_depth --skip_train 
```

특정 iteration 렌더링:
```bash
python render.py -m output/session_005_depth --iteration 30000
```

렌더 결과 경로:
- `output/session_005_depth/train/ours_<iter>/renders`
- `output/scene001_edge/test/ours_<iter>/renders`
- GT는 각 경로의 `gt/` 폴더에 저장됩니다.

### 6-5) 정량 평가

```bash
python metrics.py -m output/session_005_depth
```

## 7) COLMAP 사용 시 정리

### 7-1) 최소 입력

COLMAP 기반 파이프라인은 기본적으로 이미지(`input/` 또는 `images/`)만 있으면 시작할 수 있습니다.
카메라 내/외부 파라미터는 COLMAP이 특징점 매칭과 SfM으로 추정합니다.

### 7-2) convert.py 실행 예시

```bash
python convert.py -s datasets/scene_01/export_3dgs
```

필요 시 COLMAP 실행 파일 경로를 명시합니다.

```bash
python convert.py -s datasets/scene_01/export_3dgs --colmap_executable /path/to/colmap
```

### 7-3) 생성/사용되는 주요 파일

- `distorted/database.db`: 특징점/매칭 DB
- `distorted/sparse/0/*`: 왜곡 포함 기준 초기 SfM 결과
- `sparse/0/*`: undistort 후 최종 COLMAP 모델(학습에 사용)
  - `cameras.bin`, `images.bin`, `points3D.bin`
- (선택) `sparse/0_txt/*`: `model_converter`로 변환한 텍스트 파일

### 7-4) distorted/sparse vs sparse 차이

- `distorted/sparse/0`: Mapper가 원본(왜곡 포함) 이미지 기준으로 생성한 모델
- `sparse/0`: `image_undistorter` 이후 정리된 최종 모델  
  -> 3DGS 학습은 보통 `sparse/0`를 사용합니다.

### 7-5) intrinsics 값이 다른 이유

`camera_intrinsics.json`(수집 시 원본 값)과 `sparse/0_txt/cameras.txt`(COLMAP 최적화/undistort 후 값)는
해상도 변경(유효영역 크롭)과 BA 재추정 때문에 달라질 수 있습니다.

### 7-6) train.py에서 실제로 참조하는 데이터

`-s <scene_path>` 실행 시, `<scene_path>/sparse`가 있으면 COLMAP 모드로 로드합니다.
즉 이 경우 `transforms*.json`보다 `sparse/0`의 카메라/포즈/포인트를 우선 사용합니다.




 opencode -s ses_2562e9fb7ffe61S4bQ3jYow0E5  : depth
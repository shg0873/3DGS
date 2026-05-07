# Edge Study 12: 방법 2 vs 방법 3 정리

이 문서는 `edge_study12` 기준으로 논문에 바로 반영할 수 있도록 **방법 2(Edge 손실항 추가)**와 **방법 3(Edge-aware 렌더 기반 RGB 손실)**를 수식과 함께 정리한 내용이다.

## 1. 실험 설정

- 데이터: `datasets/scene_01/export_3dgs_s2`
- 공통 학습 조건: `3000 iterations`, `test/save interval = 250`
- 비교 대상:
  - 방법 1(기준): `noedge_like11`
  - 방법 2: `edge_like11`
  - 방법 3: `m3_d_nowarm`

## 2. 방법 2: Edge 손실항 추가

방법 2는 기본 RGB 재구성 손실에 edge 공분산 손실을 추가하는 방식이다.

### 2.1 RGB 재구성 손실

$$
L_{rgb} = (1-\lambda_{dssim})L_1(I, I_{gt}) + \lambda_{dssim}(1-SSIM(I, I_{gt}))
$$

### 2.2 Edge 공분산 손실

GT에서 Sobel 기반 edge 가중치 $w_i$와 법선 방향을 구한 뒤, 투영 공분산의 법선 방향 분산 $\sigma_{\perp,i}$를 가중 평균한다.

$$
L_{edge,raw} = \frac{\sum_i w_i\sigma_{\perp,i}}{\sum_i w_i + \epsilon}
$$

warmup을 포함한 edge 가중치는

$$
\alpha_t = \min\left(\frac{t-t_s+1}{T_w},1\right),\quad
w_{edge}=\min(w_{target}\alpha_t,0.95)
$$

최종 edge 항은

$$
L_{edge}=w_{edge}L_{edge,raw}
$$

### 2.3 총손실

$$
L_{total}=w_{L1}L_1 + w_{SSIM}(1-SSIM) + L_{edge}
$$

여기서

$$
w_{L1}=(1-w_{edge})(1-\lambda_{dssim}),\quad
w_{SSIM}=(1-w_{edge})\lambda_{dssim}
$$

## 3. 방법 3: Edge-aware 렌더 기반 RGB 손실

방법 3은 별도 $L_{edge}$를 더하지 않고, **RGB 손실 계산에 쓰는 렌더 이미지 자체를 edge-aware하게 변경**한다.

### 3.1 가우시안 크기 제한

가우시안별 edge 강도 $e_i \in [0,1]$에 대해 scale 비율을

$$
r_i = clamp\left(1-e_i(1-r_{min}),\ r_{min},\ 1\right)
$$

로 두고,

$$
\mathbf{s}_i' = r_i\mathbf{s}_i
$$

를 사용해 constrained 렌더 $I_{con}$를 생성한다.

> 본 실험(`m3_d_nowarm`)은 warmup 없이 시작 시점 이후 즉시 full 강도로 적용했다.

### 3.2 Edge band blending

기본 렌더 $I_{base}$와 constrained 렌더 $I_{con}$를 edge 가중치 맵 $W_b$로 결합한다.

$$
I_{loss}=W_b\odot I_{con} + (1-W_b)\odot I_{base}
$$

$$
W_b = clamp(s_b\cdot W_{edge},0,1)
$$

### 3.3 총손실

$$
L_{total}=(1-\lambda_{dssim})L_1(I_{loss},I_{gt}) + \lambda_{dssim}(1-SSIM(I_{loss},I_{gt}))
$$

## 4. edge_study12 결과 요약

| 방법 | 런 이름 | L1 | PSNR | EdgeL1 | EdgePSNR |
|---|---|---:|---:|---:|---:|
| 기준(1) | `noedge_like11` | 0.0112719 | 36.4652 | 0.0196283 | 31.5962 |
| 방법 2 | `edge_like11` | **0.0106955** | **36.8530** | 0.0182200 | 32.1381 |
| 방법 3 | `m3_d_nowarm` | 0.0105121 | 36.5044 | **0.0161382** | **32.6895** |

## 5. 해석

- 전체 화질 지표(PSNR, L1)에서는 방법 2가 방법 3보다 우세했다.
- 경계 중심 지표(EdgeL1, EdgePSNR)에서는 방법 3이 더 우수했다.
- 따라서 목적이 **전체 재구성 품질 최적화**인지, **경계 복원 강화**인지에 따라 방법 선택이 달라질 수 있다.

## 6. 재현 커맨드

### 방법 2 (`edge_like11`)

```bash
python train.py \
  -s datasets/scene_01/export_3dgs_s2 \
  -m output/edge_study12/edge_like11 \
  --disable_viewer \
  --iterations 3000 \
  --test_interval 250 \
  --save_interval 250 \
  --edge_cov_start_iter 1000 \
  --edge_cov_warmup_iters 1000 \
  --edge_cov_percentile 50
```

### 방법 3 (`m3_d_nowarm`)

```bash
python train.py \
  -s datasets/scene_01/export_3dgs_s2 \
  -m output/edge_study12/m3_d_nowarm \
  --disable_viewer \
  --iterations 3000 \
  --test_interval 250 \
  --save_interval 250 \
  --edge_loss_render_enable \
  --edge_loss_render_start_iter 1300 \
  --edge_loss_render_warmup_iters 0 \
  --edge_loss_render_min_scale_ratio 0.55 \
  --edge_loss_render_percentile 80 \
  --edge_loss_render_power 0.7 \
  --edge_loss_render_edgeband_only \
  --edge_loss_render_edgeband_strength 0.8
```

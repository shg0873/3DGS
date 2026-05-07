import os
import sys
import torch
from random import randint
from argparse import ArgumentParser, Namespace

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import (
    l1_loss,
    ssim,
    edge_aware_covariance_loss,
    sample_edge_weights_for_gaussians,
    sobel_edge_weight_map,
)

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def ensure_model_path(args):
    # 출력 경로를 사용자가 명시하지 않은 경우, 충돌을 피하기 위해
    # 간단한 uuid 기반 디렉토리를 자동 생성한다.
    if args.model_path:
        return
    import uuid

    args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])


def train(dataset, opt, pipe, save_iterations, checkpoint_iterations, checkpoint, debug_from):
    # sparse_adam을 선택했는데 확장 모듈이 없으면 즉시 종료한다.
    # (중간에 학습이 깨지는 것보다 초기에 명확히 실패시키는 것이 안전)
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("sparse_adam is not installed. Please install [3dgs_accel].")

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        # 체크포인트가 주어지면 가우시안 상태와 iteration을 복원하여 이어서 학습한다.
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 데이터셋 설정(white/black background)에 맞는 고정 배경값.
    # random_background가 켜지면 iteration마다 override된다.
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    # 방법 2의 edge 손실 raw 값을 안정적으로 스케일링하기 위한 EMA 버퍼.
    edge_raw_scale_ema = None

    # 훈련 카메라를 셔플 없이 복사해 사용하고, 매 스텝 랜덤 샘플링한다.
    cams = scene.getTrainCameras().copy()
    cam_indices = list(range(len(cams)))
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        # 기본 3DGS 학습 루프: 학습률 업데이트 + SH degree 점진 증가.
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not cams:
            cams = scene.getTrainCameras().copy()
            cam_indices = list(range(len(cams)))

        ridx = randint(0, len(cam_indices) - 1)
        viewpoint_cam = cams.pop(ridx)
        _ = cam_indices.pop(ridx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image = render_pkg["render"]
        radii = render_pkg["radii"]
        # densification 단계에서 사용할 렌더 보조 텐서.
        densify_viewspace_points = render_pkg["viewspace_points"]
        densify_visibility_filter = render_pkg["visibility_filter"]

        # 기본적으로는 일반 렌더 결과를 loss 입력으로 사용한다.
        image_for_loss = image

        if opt.edge_loss_render_enable and iteration >= opt.edge_loss_render_start_iter:
            # 방법 3 (Edge-aware render 기반 RGB 손실)
            # 1) GT 이미지에서 edge 강도를 추정하고, 각 가우시안 위치에서 edge 가중치를 샘플링
            # 2) r_i = clamp(1 - e_i(1-r_min), r_min, 1) 비율로 scale 축소
            # 3) 축소된 scale로 constrained render를 한 번 더 생성
            # 4) 옵션에 따라 edge band에서만 base/constrained를 블렌딩해 최종 loss 이미지를 구성
            with torch.no_grad():
                edge_weights = sample_edge_weights_for_gaussians(
                    viewpoint_cam.original_image.cuda(),
                    gaussians.get_xyz,
                    radii,
                    viewpoint_cam.full_proj_transform,
                    edge_percentile=opt.edge_loss_render_percentile,
                    edge_power=opt.edge_loss_render_power,
                ).clamp(0.0, 1.0)
                min_ratio = max(0.0, min(1.0, float(opt.edge_loss_render_min_scale_ratio)))
                # 논문식: edge 강도가 강할수록 축소 비율(ratio)이 작아짐.
                ratio = (1.0 - edge_weights * (1.0 - min_ratio)).clamp(min=min_ratio, max=1.0)

            constrained_scales = gaussians.get_scaling * ratio.unsqueeze(1)
            loss_render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                scales_override=constrained_scales,
            )
            constrained_image = loss_render_pkg["render"]
            if opt.edge_loss_render_edgeband_only:
                with torch.no_grad():
                    # edge band 전용 블렌딩 마스크 생성.
                    edge_map = sobel_edge_weight_map(
                        viewpoint_cam.original_image.cuda(),
                        percentile=opt.edge_loss_render_percentile,
                        power=opt.edge_loss_render_power,
                    )
                    edge_map = (edge_map * float(opt.edge_loss_render_edgeband_strength)).clamp(0.0, 1.0).unsqueeze(0)
                # edge 영역은 constrained, 비-edge는 base를 더 반영한다.
                image_for_loss = edge_map * constrained_image + (1.0 - edge_map) * image
            else:
                # 전 영역에서 constrained render를 loss 입력으로 사용.
                image_for_loss = constrained_image

            # 방법 3를 사용할 때 densification 기준도 constrained 렌더 결과와 정렬한다.
            densify_viewspace_points = loss_render_pkg["viewspace_points"]
            densify_visibility_filter = loss_render_pkg["visibility_filter"]

        if viewpoint_cam.alpha_mask is not None:
            # alpha 마스크가 있는 데이터셋은 유효 픽셀만 loss에 반영한다.
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
            image_for_loss *= alpha_mask

        # 기본 RGB 재구성 손실 (L1 + DSSIM)
        gt_image = viewpoint_cam.original_image.cuda()
        l1_val = l1_loss(image_for_loss, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(image_for_loss.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_val = ssim(image_for_loss, gt_image)

        l_ssim = 1.0 - ssim_val
        w_edge = 0.0
        w_l1 = (1.0 - w_edge) * (1.0 - opt.lambda_dssim)
        w_ssim = (1.0 - w_edge) * opt.lambda_dssim
        loss = w_l1 * l1_val + w_ssim * l_ssim

        # 방법 2 (Edge-aware covariance loss)
        # 방법 3를 사용하지 않거나, 방법 3와 병행 옵션이 켜진 경우에 활성화된다.
        use_edge_cov = (not opt.edge_loss_render_enable) or opt.edge_loss_render_with_cov
        if use_edge_cov and iteration >= opt.edge_cov_start_iter:
            # warmup: edge 손실 가중치를 점진적으로 키워 학습 초기 불안정을 완화.
            warmup = 1.0
            if opt.edge_cov_warmup_iters > 0:
                warmup = min((iteration - opt.edge_cov_start_iter + 1) / float(opt.edge_cov_warmup_iters), 1.0)

            l_edge_raw = edge_aware_covariance_loss(
                gt_image,
                gaussians.get_xyz,
                gaussians.get_scaling,
                gaussians.get_rotation,
                radii,
                viewpoint_cam.full_proj_transform,
                viewpoint_cam.world_view_transform,
                viewpoint_cam.FoVx,
                viewpoint_cam.FoVy,
                edge_percentile=opt.edge_cov_percentile,
                edge_power=opt.edge_cov_power,
                min_weight=opt.edge_cov_min_weight,
            )

            w_edge = min(opt.edge_cov_weight_target * warmup, 0.95)
            w_l1 = (1.0 - w_edge) * (1.0 - opt.lambda_dssim)
            w_ssim = (1.0 - w_edge) * opt.lambda_dssim

            if opt.edge_cov_auto_norm:
                # raw edge 손실의 스케일 변동을 EMA로 정규화하여
                # iteration마다 손실 크기가 급격히 흔들리는 것을 줄인다.
                if edge_raw_scale_ema is None:
                    edge_raw_scale_ema = l_edge_raw.detach()
                else:
                    m = float(opt.edge_cov_norm_momentum)
                    edge_raw_scale_ema = m * edge_raw_scale_ema + (1.0 - m) * l_edge_raw.detach()
                l_edge_norm = l_edge_raw / edge_raw_scale_ema.clamp_min(float(opt.edge_cov_norm_eps))
            else:
                l_edge_norm = l_edge_raw

            # 최종 총손실: RGB 손실 + 가중된 edge covariance 손실.
            loss = w_l1 * l1_val + w_ssim * l_ssim + (w_edge * l_edge_norm)

        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0 or iteration == opt.iterations:
                print(f"[ITER {iteration}] loss={loss.item():.7f}")

            if iteration in save_iterations:
                print(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                # 3DGS 기본 파이프라인: visibility/radii 기반 densify & prune.
                gaussians.max_radii2D[render_pkg["visibility_filter"]] = torch.max(
                    gaussians.max_radii2D[render_pkg["visibility_filter"]],
                    radii[render_pkg["visibility_filter"]],
                )
                gaussians.add_densification_stats(densify_viewspace_points, densify_visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                # 노출(exposure) optimizer와 gaussian optimizer를 순서대로 업데이트.
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                else:
                    gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")


if __name__ == "__main__":
    # 원본 train.py의 인자 체계를 최대한 유지하면서,
    # 본 스크립트는 논문 핵심 학습 루프만 남긴 경량 실행 진입점이다.
    parser = ArgumentParser(description="Minimal training script for paper methods")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.save_iterations is None:
        args.save_iterations = list(range(args.save_interval, args.iterations + 1, args.save_interval))
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    # 출력 경로 생성 및 설정값 스냅샷 저장.
    ensure_model_path(args)
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )
    print("\nTraining complete.")

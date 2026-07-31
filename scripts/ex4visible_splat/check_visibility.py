'''
please  read the train.py, then you can know how to load the model from the checkpoint, and the please run the model on the test set, but you don't need to generate the images, just get the mask from the line 162 of gaussian_render/__init__.py. The 1 in mask means the splat is visible, and 0 means the splat is invisible. Then you can estimate the visibility ratio of the splats in the view. Please get the average visibility ratio of the splats in all test sets for the scenes. And store all the scenes for one view.

you can load the trained model from outputs/ex1reproduce/${dataset}/${scene}/${lmbda}/checkpoints/iter_${loaded_iter}.pth, and the loaded_iter is the max iteration in the checkpoints folder.

Also compare against gaussians.get_mask (gaussian_renderer/__init__.py:37, `binary_grid_masks`): a
learned, per-anchor/per-offset prune mask that is (almost) view-independent -- only the anchor
subset changes across views because of frustum culling, unlike the opacity mask which is genuinely
view-dependent (opacity is predicted from the viewing direction/distance). The two masks live over
the same [N_visible_anchor * n_offsets] set of candidate splats, so they can be compared elementwise;
their intersection is exactly the set of splats that make it into the final render.
'''

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import prefilter_voxel, generate_neural_gaussians

# scene -> dataset, mirrors scripts/ex1reproduce/submit.py
SCENES_BY_DATASET = {
    "mipnerf360": ["garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room", "treehill", "flowers"],
    "db": ["drjohnson", "playroom"],
    "tandt": ["train", "truck"],
}


def find_model_path(output_root, dataset, scene, lmbda):
    # outputs/ex1reproduce is not laid out consistently: most scenes were saved
    # directly under outputs/ex1reproduce/<scene>/<lmbda>, only some also have a
    # outputs/ex1reproduce/<dataset>/<scene>/<lmbda> copy. Try both.
    candidates = [
        Path(output_root) / scene / str(lmbda),
        Path(output_root) / dataset / scene / str(lmbda),
    ]
    for candidate in candidates:
        if (candidate / "point_cloud").is_dir():
            return candidate
    raise FileNotFoundError(f"no trained checkpoint found for dataset={dataset} scene={scene} under {output_root}")


def load_scene_and_gaussians(model_path, source_path):
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    dataset = lp.extract(args)
    dataset.source_path = os.path.abspath(source_path)
    dataset.model_path = str(model_path)
    pipeline = pp.extract(args)

    is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        # matches train.py's --n_features/--log2/--log2_2D defaults used for training
        n_features_per_level=4,
        log2_hashmap_size=13,
        log2_hashmap_size_2D=15,
        decoded_version=True,
        is_synthetic_nerf=is_synthetic_nerf,
    )
    # loaded_iter=-1 -> Scene picks the max iteration found under model_path/point_cloud
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    gaussians.eval()

    return scene, gaussians, pipeline


@torch.no_grad()
def compute_visibility_ratios(scene, gaussians, pipeline):
    # background color is irrelevant here: prefilter_voxel's visibility test and
    # generate_neural_gaussians's opacity mask never read bg_color, it's only
    # used by the rasterizer, which we skip since we don't render images.
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    opacity_ratios = []    # line-162 mask: neural_opacity > 0, view-dependent
    structural_ratios = []  # pc.get_mask: learned per-offset prune mask, view-independent (only the anchor subset varies with the frustum)
    combined_ratios = []    # splats passing both masks -> what actually ends up rendered

    for view in tqdm(scene.getTestCameras(), desc="test views", leave=False):
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        _, _, _, _, _, _, opacity_mask = generate_neural_gaussians(
            view, gaussians, visible_mask=voxel_visible_mask, is_training=False
        )
        # # gaussian_renderer/__init__.py:37 -- same visible_mask, same [N_vis, n_offsets] layout as opacity_mask.
        # # NOTE: gaussians.get_mask's decoded_version branch just returns the raw _mask parameter
        # # unthresholded (it assumes the mask already went through conduct_decoding()'s bitstream
        # # round-trip, which we don't run here -- we load straight from the ply via
        # # load_ply_sparse_gaussian, same as train.py's render_sets()). The raw values are therefore
        # # pre-sigmoid logits (empirically ~[-20, +18]), not 0/1. Binarize the same way the model
        # # itself does at train/non-decoded-eval time: sigmoid(logit) > 0.01.
        # structural_mask = (torch.sigmoid(gaussians._mask[:, :10, :][voxel_visible_mask]).view(-1) > 0.01)

        opacity_ratios.append(opacity_mask.float().mean().item())
        # structural_ratios.append(structural_mask.float().mean().item())
        # combined_ratios.append((opacity_mask & structural_mask).float().mean().item())

    return opacity_ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../../../data/GS",
                         help="path to the GS data root, containing mipnerf360/db/tandt subfolders")
    parser.add_argument("--output_root", type=str, default="outputs/ex1reproduce",
                         help="path containing the trained checkpoints, e.g. <output_root>/<scene>/<lmbda>")
    parser.add_argument("--lmbda", type=float, default=0.004)
    parser.add_argument("--save_path", type=str, default="scripts/ex4visible_splat/visibility_ratio.json")
    args = parser.parse_args()

    results = {}
    for dataset, scenes in SCENES_BY_DATASET.items():
        for scene_name in scenes:
            print(f"\n[{dataset}/{scene_name}] loading checkpoint...")
            model_path = find_model_path(args.output_root, dataset, scene_name, args.lmbda)
            source_path = os.path.join(args.dataset_root, dataset, scene_name)

            scene, gaussians, pipeline = load_scene_and_gaussians(model_path, source_path)
            opacity_ratios = compute_visibility_ratios(scene, gaussians, pipeline)
            n = len(opacity_ratios)
            avg_opacity_ratio = sum(opacity_ratios) / n

            print(f"[{dataset}/{scene_name}] over {n} test views -- "
                  f"opacity_mask(line162)={avg_opacity_ratio:.4f} ")
            results[scene_name] = {
                "dataset": dataset,
                "model_path": str(model_path),
                "loaded_iter": scene.loaded_iter,
                "num_test_views": n,
                "avg_visibility_ratio": avg_opacity_ratio,
                "per_view_visibility_ratio": opacity_ratios,
            }

            del scene, gaussians
            torch.cuda.empty_cache()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved visibility ratios for {len(results)} scenes to {save_path}")


if __name__ == "__main__":
    main()

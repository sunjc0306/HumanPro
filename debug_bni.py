import os
import argparse
from dataclasses import dataclass
import torch
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple, List
from econdataset import SMPLDataset
from reconstruct import ReMesh
@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    # save_single_views: bool
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: float
    validation_grid_nrow: int

    num_views: int
    enable_xformers_memory_efficient_attention: bool
    with_smpl: Optional[bool]

    recon_opt: Dict
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/inference-768-6view_BNI.yaml')
    args, extras = parser.parse_known_args()
    from utils.misc import load_config

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)
    dataset_param = {'image_dir': '/mnt/data1/sjc1/PSHuman/bni', 'seg_dir': None, 'colab': False, 'has_det': True, 'hps_type': 'pixie'}
    econdata = SMPLDataset(dataset_param, device='cuda')

    carving = ReMesh(cfg.recon_opt, econ_dataset=econdata)
    item_data = torch.load('./bni/item_data.pt')

    # scene = batch['filename'][0]
    scene = item_data['scene']
    pose = item_data['pose']
    colors = item_data['colors']
    normals = item_data['normals']
    v_smpl = item_data['v_smpl']
    f_smpl = item_data['f_smpl']
    # carving.optimize_case(scene, pose, colors, normals)
    carving.stich_case(v_smpl,f_smpl,colors,normals,econdata)



# in_tensor["depth_F"], in_tensor["depth_B"] = dataset.render_depth(batch_smpl_verts, batch_smpl_faces)
# in_tensor["BNI_verts"] = []
# in_tensor["BNI_faces"] = []
# in_tensor["body_verts"] = []
# in_tensor["body_faces"] = []
# final_path = f"{args.out_dir}/{cfg.name}/obj/{data['name']}_{idx}_full.obj"
#
# side_mesh = smpl_obj_lst[idx].copy()
# face_mesh = smpl_obj_lst[idx].copy()
# hand_mesh = smpl_obj_lst[idx].copy()
# smplx_mesh = smpl_obj_lst[idx].copy()
#
# # save normals, depths and masks
# BNI_dict = save_normal_tensor(
#     in_tensor,
#     idx,
#     osp.join(args.out_dir, cfg.name, f"BNI/{data['name']}_{idx}"),
#     cfg.bni.thickness,
# )
#
# # BNI process
# BNI_object = BNI(
#     dir_path=osp.join(args.out_dir, cfg.name, "BNI"),
#     name=data["name"],
#     BNI_dict=BNI_dict,
#     cfg=cfg.bni,
#     device=device)
#
# BNI_object.extract_surface(False)
# cv2.imwrite('./debug/depth_F.png', ((BNI_object.F_depth + 1) / 2 * 65536).numpy().astype(np.uint16))
# in_tensor["body_verts"].append(torch.tensor(smpl_obj_lst[idx].vertices).float())
# in_tensor["body_faces"].append(torch.tensor(smpl_obj_lst[idx].faces).long())
# cv2.imwrite('./debug/depth_B.png', ((BNI_object.B_depth + 1) / 2 * 65536).numpy().astype(np.uint16))
# side_mesh = apply_vertex_mask(
#     side_mesh,
#     (SMPLX_object.front_flame_vertex_mask + SMPLX_object.mano_vertex_mask +
#      SMPLX_object.eyeball_vertex_mask).eq(0).float(),
# )
#
# # register side_mesh to BNI surfaces
# side_mesh = Meshes(
#     verts=[torch.tensor(side_mesh.vertices).float()],
#     faces=[torch.tensor(side_mesh.faces).long()],
# ).to(device)
# sm = SubdivideMeshes(side_mesh)
# side_mesh = register(BNI_object.F_B_trimesh, sm(side_mesh), device)
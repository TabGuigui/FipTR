# ego coords multi in multi out
import copy
import tempfile
import torch
import numpy as np
import pyquaternion
import random
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
import mmcv
from mmcv.parallel import DataContainer as DC
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom


@DATASETS.register_module()
class CustomNuScenesMotionEgoMultiframeDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 queue_length=3, 
                 bev_size=(200, 200), 
                 overlap_test=False, 
                 future_frames=0, # for future prediction
                 filter_invalid_sample=True,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

        # occ
        self.n_future = future_frames
        self.filter_invalid_sample = filter_invalid_sample

        # process infos so that they are sorted w.r.t. scenes & time_stamp
        self.data_infos.sort(key=lambda x: (x['scene_token'], x['timestamp']))
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        first_index = index - self.queue_length + 1
        final_index = index
        
        if first_index < 0:
            return None
        # if self.data_infos[first_index]['scene_token'] != self.data_infos[final_index]['scene_token']:
        #     return None
        
        input_dict = self.get_data_info(final_index)
        prev_indexs_list = list(reversed(range(first_index, final_index)))

        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict) # 当前帧的图像 以及 未来的gt
        if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        queue.insert(0, example)

        for i in prev_indexs_list:
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            # else:
            #     print("warning!!!!! the prev dataset is not in the same with ")
            queue.insert(0, copy.deepcopy(example))
        final = self.union2one(queue)

        return final

    def prepare_test_data(self, index):
        """
        test data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: test data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length + 1, index + 1)) # no random 操作
        for index in index_list:
            i = max(0, index)
            input_dict = self.get_data_info(i)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        
        return self.union2one_test(queue)
    
    def union2one_test(self, queue):
        imgs_list = [each['img'][0].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'][0].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = [DC(torch.stack(imgs_list), cpu_only=False, stack=True)]
        queue[-1]['img_metas'] = [DC(metas_map, cpu_only=True)]
        queue = queue[-1]
        return queue

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1] # 未来帧的标注已经在里面了

        if queue["gt_masks"].data.shape[0] != len(queue["gt_bboxes_3d"].data):
            print("warning !!!!!!!!")
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'], # ego2global
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        # for lidar2ego
        input_dict['lidar2ego_rots'] = torch.tensor(pyquaternion.Quaternion(
            info['lidar2ego_rotation']).rotation_matrix)
        input_dict['lidar2ego_trans'] = torch.tensor(
            info['lidar2ego_translation'])

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix

                lidar2cam_r = np.linalg.inv(Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix) # TODO check ego2img 放到bevformer中
                lidar2cam_t = cam_info[
                    'sensor2ego_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # if not self.test_mode:
        annos = self.get_ann_info(index) # gt_bboxes_3d, gt_labels_3d, gt_names, instance_tokens, gt_vis_tokens lidar坐标系
        input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        # occ future related
        prev_indices, future_indices = self.occ_get_temporal_indices(
            index, self.queue_length, self.n_future)

        all_frames = prev_indices + [index] + future_indices
        has_invalid_frame = -1 in all_frames 

        input_dict['has_invalid_frame'] = has_invalid_frame # 有没有失效帧
        input_dict['img_is_valid'] = np.array(all_frames) >= 0 # 哪些帧是失效的

        # might have None if not in the same sequence
        future_frames = [index] + future_indices
        # 'occ_l2e_r_mats', 'occ_l2e_t_vecs', 'occ_e2g_r_mats', 'occ_e2g_t_vecs', 每个都是list，长度为future+1, item是tensor
        occ_transforms = self.occ_get_transforms(future_frames) 
        input_dict.update(occ_transforms) # for gt transform
        input_dict['occ_future_ann_infos'] = self.get_future_detection_infos(future_frames)

        return input_dict

    def occ_get_temporal_indices(self, index, receptive_field, n_future):
        current_scene_token = self.data_infos[index]['scene_token']

        # generate the past
        previous_indices = []

        for t in range(- receptive_field + 1, 0):
            index_t = index + t
            if index_t >= 0 and self.data_infos[index_t]['scene_token'] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []

        for t in range(1, n_future + 1):
            index_t = index + t
            if index_t < len(self.data_infos) and self.data_infos[index_t]['scene_token'] == current_scene_token:
                future_indices.append(index_t)
            else:
                # NOTE: How to deal the invalid indices???
                future_indices.append(-1)

        return previous_indices, future_indices

    def occ_get_transforms(self, indices, data_type=torch.float32):
        """
        get l2e, e2g rotation and translation for each valid frame
        """
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                l2e_r = info['lidar2ego_rotation']
                l2e_t = info['lidar2ego_translation']
                e2g_r = info['ego2global_rotation']
                e2g_t = info['ego2global_translation']

                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)

                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))

        res = {
            'occ_l2e_r_mats': l2e_r_mats,
            'occ_l2e_t_vecs': l2e_t_vecs,
            'occ_e2g_r_mats': e2g_r_mats,
            'occ_e2g_t_vecs': e2g_t_vecs,
        }

        return res

    def get_future_detection_infos(self, future_frames):
        detection_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                detection_ann_infos.append(
                    self.get_ann_info(future_frame), # ego坐标系
                )
            else:
                detection_ann_infos.append(None)
        return detection_ann_infos

    def get_ann_info(self, index):
        info = self.data_infos[index]

        # uniAD中没有mask，全部保留。
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        # need instance and visibility
        gt_bboxes_3d = info["gt_boxes"]
        gt_names_3d = info["gt_names"]
        gt_instance_tokens = info["instance_tokens"]
        gt_vis_tokens = info['visibility_tokens']

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        
        if self.with_velocity:
            gt_velocity = info['gt_velocity']
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        gt_bboxes_3d_ego = copy.deepcopy(gt_bboxes_3d)

        # lidar 2 ego
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_rotation = pyquaternion.Quaternion(
            lidar2ego_rotation).rotation_matrix
        gt_bboxes_3d_ego.rotate(lidar2ego_rotation.T)
        gt_bboxes_3d_ego.translate(lidar2ego_translation)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d_ego,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_tokens=gt_instance_tokens,
            gt_vis_tokens=gt_vis_tokens,
            gt_bboxes_3d_ego=gt_bboxes_3d_ego, # TODO 删除这项
            )
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol with instance segmentation.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

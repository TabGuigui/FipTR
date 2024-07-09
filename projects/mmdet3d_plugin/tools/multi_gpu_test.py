import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from .metrics import IntersectionOverUnion, PanopticMetric

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    # for motion eval
    EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
    num_motion_class = 2

    motion_panoptic_metrics = {}
    motion_iou_metrics = {}
    for key in EVALUATION_RANGES.keys():
        motion_iou_metrics[key] = IntersectionOverUnion(
            num_motion_class).cuda()
        motion_panoptic_metrics[key] = PanopticMetric(
            n_classes=num_motion_class, temporally_consistent=True).cuda()
        
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    bbox_results.extend(bbox_result)
            else:
                bbox_results.extend(result)

            if "has_invalid_frame" in data:
                has_invalid_frame = data['has_invalid_frame'][0]
            else:
                has_invalid_frame = False
            
            if not has_invalid_frame or not has_invalid_frame.item():
                num_occ += 1
                if type(result) == list: # bevpanformer
                    if result[0]['pts_bbox']["segmentation"].dim() == 3: # bevpanformer 3to5
                        motion_segmentation = result[0]['pts_bbox']["segmentation"].unsqueeze(0)
                        motion_instance = result[0]['pts_bbox']["instance"].unsqueeze(0)
                    else:
                        motion_segmentation = result[0]['pts_bbox']["segmentation"].unsqueeze(0).unsqueeze(0) # bevpanformer 3to1
                        motion_instance = result[0]['pts_bbox']["instance"].unsqueeze(0).unsqueeze(0)
                    if "motion_segmentation" in data:
                        motion_targets = {
                        "segmentation": data["motion_segmentation"][0],
                        "instance": data["motion_instance"][0],
                            }
                    else:
                        motion_targets = {
                        "segmentation": data["gt_segmentation"][0],
                        "instance": data["gt_instance"][0],
                            }
                elif "gt_backward_flow" in data:
                    print("fistr lss")
                    motion_segmentation = result['motion_segmentation'] # bs 5 1 200 200 bs*t*1*200*200
                    motion_instance = result["motion_instance"] # bs 5 200 200 bs*t*200*200
                    motion_targets = {
                        "segmentation": data["motion_segmentation"][0],
                        "instance": data["motion_instance"][0],
                            }
                else: # beverse
                    motion_segmentation = result['motion_segmentation'] # bs 5 1 200 200 bs*t*1*200*200
                    motion_instance = result["motion_instance"] # bs 5 200 200 bs*t*200*200
                    motion_targets = {
                            'motion_segmentation': data['motion_segmentation'][0], # bs 5 200 200
                            'motion_instance': data['motion_instance'][0], # bs 5 200 200
                            'instance_centerness': data['instance_centerness'][0], # bs 5 1 200
                            'instance_offset': data['instance_offset'][0], # bs 5 2 200
                            'instance_flow': data['instance_flow'][0], # bs 5 2 200 200
                            'future_egomotion': data['future_egomotions'][0], # bs 7 6
                        }
                    motion_targets, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(motion_targets) 

                for key, grid in EVALUATION_RANGES.items():
                    limits = slice(grid[0], grid[1])
                    motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                            ), motion_targets['segmentation'][..., limits, limits].contiguous().cuda())
                    motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous(),
                                motion_targets["instance"][..., limits, limits].contiguous().cuda())
        if rank == 0:
            for _ in range(data_loader.batch_size * world_size):
                prog_bar.update()

    if rank == 0:
            print(
            '\n[Validation {} / {}, ration: {}]: motion metrics: '.format(num_occ, len(dataset), num_occ / len(dataset))) # occ is computed on one gpu

    for key, grid in EVALUATION_RANGES.items():
        results_str = 'grid = {}: '.format(key)
        iou_scores = motion_iou_metrics[key].compute()
        panoptic_scores = motion_panoptic_metrics[key].compute()
        # logging
        if rank == 0:
            results_str += 'iou = {:.3f}, '.format(
                iou_scores[1].item() * 100)
            for panoptic_key, value in panoptic_scores.items():
                        results_str += '{} = {:.3f}, '.format(
                            panoptic_key, value[1].item() * 100)
            print(results_str)

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None

    mask_results = None
    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
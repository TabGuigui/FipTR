from mmcv import Config
from mmdet3d.datasets import build_dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from mmdet3d.models import build_model
INSTANCE_COLOURS = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])
def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out
def draw(data, i, name):
    semantic_seg = data.cpu().numpy().astype(np.int)
    # semantic_colours = np.array([[255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    semantic_plot = INSTANCE_COLOURS[semantic_seg]
    semantic_plot = make_contour(semantic_plot)
    plt.imshow(semantic_plot)
    plt.savefig("{}.jpg".format(name))

cfg_lss = Config.fromfile("/data/FIS/FISTR_bevformer/projects/configs/fistr_lss_tiny.py")
cfg_bevformer = Config.fromfile('/data/FIS/FISTR_bevformer/projects/configs/fistr_bevformer_tiny.py')

# # import modules from plguin/xx, registry will be updated
if hasattr(cfg_lss, 'plugin'):
    if cfg_lss.plugin:
        import importlib
        if hasattr(cfg_lss, 'plugin_dir'):
            plugin_dir = cfg_lss.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

# dataset_lss = build_dataset(cfg_lss.data.train)
# dataset_bevformer = build_dataset(cfg_bevformer.data.train)

# assert len(dataset_lss) == len(dataset_bevformer)
# from projects.mmdet3d_plugin.datasets.builder import build_dataloader
# data_loader1 = build_dataloader(
#         dataset_lss,
#         samples_per_gpu=1,
#         workers_per_gpu=4,
#         dist=False,
#         shuffle=False,
#         nonshuffler_sampler=cfg_lss.data.nonshuffler_sampler,
#     )
# model = build_model(cfg_lss.model, test_cfg=cfg_lss.get('test_cfg'))


dataset_lss = build_dataset(cfg_lss.data.train)
dataset_bevformer = build_dataset(cfg_bevformer.data.train)

assert len(dataset_lss) == len(dataset_bevformer)

data1 = dataset_lss[10]
data2 = dataset_bevformer[10]

for i in range(5):
    print((data1["motion_instance"][i] == data2["motion_instance"][i]).all())
    for j in range(data1["gt_masks"].data.shape[0]):
        print((data1["gt_masks"].data[j] == data2["gt_masks"].data[j]).all())
draw(data1["gt_masks"].data[5][4], 0, "lss")
draw(data2["gt_masks"].data[5][4], 0, "transformer")

for i in (range(0, len(dataset_bevformer))):
    data1 = dataset_lss[i]
    data2 = dataset_bevformer[i]
    pass



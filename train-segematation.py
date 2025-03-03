import os
import sys
from tqdm import tqdm
# import cv2
import numpy as np
import json
import skimage.draw
import matplotlib
import matplotlib.pyplot as plt
import random

# Root directory of the project
ROOT_DIR = os.path.abspath('Mask_RCNN/')
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn.model import log
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
# Import COCO config
from Mask_RCNN.samples import coco

plt.rcParams['figure.facecolor'] = 'white'


def get_ax(rows=1, cols=1, size=7):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


ROOT_DIR = os.path.abspath('Mask_RCNN/')
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')  # directory to save logs and trained model
# ANNOTATIONS_DIR = 'brain-tumor/data/new/annotations/' # directory with annotations for train/val sets
DATASET_DIR = 'brain-tumor-master/data_cleaned/'  # directory with image data
DEFAULT_LOGS_DIR = 'logs'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class TumorConfig(Config):
    """Configuration for training on the brain tumor dataset.
    """
    # Give the configuration a recognizable name
    NAME = 'tumor_detector'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tumor
    DETECTION_MIN_CONFIDENCE = 0.85
    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001


config = TumorConfig()
config.display()


class BrainScanDataset(utils.Dataset):

    def load_brain_scan(self, dataset_dir, subset):
        """Load a subset of the FarmCow dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("tumor", 1, "tumor")

        # Train or validation dataset?
        assert subset in ["train", "val", 'test']
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(DATASET_DIR, subset, 'annotations_' + subset + '.json')))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "tumor",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a farm_cow dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tumor":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tumor":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


model = modellib.MaskRCNN(
    mode='inference',
    config=config,
    model_dir=DEFAULT_LOGS_DIR
)

model.load_weights(
    COCO_MODEL_PATH,
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
)

# Training dataset.
dataset_train = BrainScanDataset()
dataset_train.load_brain_scan(DATASET_DIR, 'train')
dataset_train.prepare()

# Validation dataset
dataset_val = BrainScanDataset()
dataset_val.load_brain_scan(DATASET_DIR, 'val')
dataset_val.prepare()

dataset_test = BrainScanDataset()
dataset_test.load_brain_scan(DATASET_DIR, 'test')
dataset_test.prepare()

# Since we're using a very small dataset, and starting from
# COCO trained weights, we don't need to train too long. Also,
# no need to train all layers, just the heads should do it.
# print("Training network heads")
# model.train(
#     dataset_train, dataset_val,
#     learning_rate=config.LEARNING_RATE,
#     epochs=1,
#     layers='heads'
# )
#
# # Recreate the model in inference mode
# model = modellib.MaskRCNN(
#     mode="inference",
#     config=config,
#     model_dir=DEFAULT_LOGS_DIR
# )

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


def predict_and_plot_differences(dataset, img_id):
    original_image, image_meta, gt_class_id, gt_box, gt_mask = \
        modellib.load_image_gt(dataset, config,
                               img_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]
    print(r)
    visualize.display_differences(
        original_image,
        gt_box, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        class_names=['tumor'], title="", ax=get_ax(),
        show_mask=True, show_box=True)


def display_image(dataset, ind):
    plt.figure(figsize=(5, 5))
    plt.imshow(dataset.load_image(ind))
    plt.xticks([])
    plt.yticks([])
    plt.title('Original Image')
    plt.show()


import numpy as np
# from sklearn.metrics import average_precision_score


def predict_and_plot_differences(dataset, img_id):
    original_image, image_meta, gt_class_id, gt_box, gt_mask = \
        modellib.load_image_gt(dataset, config,
                               img_id, use_mini_mask=False)

    results = model.detect([original_image], verbose=0)
    r = results[0]

    # def get_ax(rows=1, cols=2, size=7):
    visualize.display_differences(
        original_image,
        gt_box, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'], class_names=['tumor'], title="", show_mask=True,
        show_box=True)


def evaluate_model(dataset, model, cfg):
    APs = list();
    ARs = list();
    F1_scores = list();
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id,
                                                                                  use_mini_mask=False)
        # scaled_image = mold_image(image, cfg)
        # sample = expand_dims(scaled_image, 0)
        results_evaluation = model.detect([image], verbose=0)
        r = results_evaluation[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'],
                                                             r['scores'], r['masks'], iou_threshold=0.6)
        AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.2)
        ARs.append(AR)
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    recall = np.mean(recalls)
    precision = np.mean(precisions)
    F1_scores.append((2 * (np.mean(precisions) * np.mean(recalls))) / (np.mean(precisions) + np.mean(recalls)))

    mAR = np.mean(ARs)
    # recalls = np.mean

    return mAP, mAR, F1_scores, recall, precision


# cfg = TumorConfig()
# mAP, mAR, F1_score, recall, precision = evaluate_model(dataset_test, model, cfg)
#
# f_score_test = (2 * mAP * mAR) / (mAP + mAR)
#
# print('f1-score-test', f_score_test)
# print('Map', mAP)
# print('Recalls', recall)
# print('Precision', precision)


def evaluate_model_pic(dataset, model, cfg):
    APs = list()
    ARs = list()
    F1_scores = list()
    precisions = []
    recalls = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        results_evaluation = model.detect([image], verbose=0)
        r = results_evaluation[0]
        AP, precisions_, recalls_, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.6)
        AR, positive_ids = utils.compute_recall(r["rois"], gt_bbox, iou=0.2)
        ARs.append(AR)
        APs.append(AP)
        precisions.append(np.mean(precisions_))
        recalls.append(np.mean(recalls_))
    mAP = np.mean(APs)
    mAR = np.mean(ARs)

    # Plot evaluation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(precisions, label='Precision')
    plt.plot(recalls, label='Recall')
    plt.xlabel('Image ID')
    plt.ylabel('Score')
    plt.title('Precision and Recall for Test Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

    return mAP, mAR

cfg = TumorConfig()
mAP, mAR = evaluate_model_pic(dataset_test, model, cfg)
print('Mean Average Precision (mAP):', mAP)
print('Mean Average Recall (mAR):', mAR)
# ind = 0
# display_image(dataset_val, ind)
# predict_and_plot_differences(dataset_val, ind)
#
# ind = 10
# display_image(dataset_val, ind)
# predict_and_plot_differences(dataset_val, ind)
#
# ind = 11
# display_image(dataset_val, ind)
# predict_and_plot_differences(dataset_val, ind)
# #
# ind = 12
# display_image(dataset_val, ind)

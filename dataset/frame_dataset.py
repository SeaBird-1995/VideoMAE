from torch.utils.data import Dataset
import os
import io
from collections import defaultdict
import numpy as np
import time

import logging
from PIL import Image, ImageOps
from einops import rearrange

logger = logging.getLogger(__file__)


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


class FrameDataset(Dataset):
    def __init__(self,
                 path_prefix,
                 frame_list,
                 frame_length=16,
                 sampling_rate=2,
                 transform=None
                 ):
        logger.info("Loading frame list from {}".format(frame_list))
        self._path_to_frames, _ = load_image_lists(frame_list, prefix=path_prefix, return_list=True)
        self.frame_length = frame_length
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __getitem__(self, item):
        frame_paths = self._path_to_frames[item]
        duration = len(frame_paths)
        selected_frame_indices = self._sample_train_indices(duration)
        images = self.retry_load_images([frame_paths[frame] for frame in selected_frame_indices])

        process_data, mask = self.transform((images, None))  # T*C,H,W
        process_data = rearrange(process_data, '(t c) h w -> c t h w', t=self.frame_length)
        return process_data, mask

    def __len__(self):
        return len(self._path_to_frames)

    def _sample_train_indices(self, video_length):
        delta = video_length - (self.frame_length * self.sampling_rate) + 1
        if delta > 0:
            start = np.random.randint(0, delta)
        else:
            start = 0
        indices = start + np.arange(0, self.frame_length * self.sampling_rate, self.sampling_rate, dtype=np.int)
        indices = np.clip(indices, 0, video_length - 1)
        return indices

    @staticmethod
    def retry_load_images(image_paths, retry=10):
        """
        This function is to load images with support of retrying for failed load.

        Args:
            image_paths (list): paths of images needed to be loaded.
            retry (int, optional): maximum time of loading retrying. Defaults to 10.
        Returns:
            imgs (list): list of loaded PIL images.
        """
        for i in range(retry):
            imgs = []
            for image_path in image_paths:
                with open(image_path, "rb") as f:
                    buff = io.BytesIO(f.read())
                img = Image.open(buff)
                img = ImageOps.exif_transpose(img)
                img = img.convert('RGB')

                imgs.append(img)

            if all(img is not None for img in imgs):
                return imgs
            else:
                logger.warning("Reading failed. Will retry.")
                time.sleep(1.0)
            if i == retry - 1:
                raise Exception("Failed to load images {}".format(image_paths))

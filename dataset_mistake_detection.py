# -*- coding: utf-8 -*-

""" Implements a dataset object which allows to read representations from LMDB datasets."""
import lmdb
import numpy as np
import pandas as pd
from torch.utils import data
from tqdm import tqdm
import json
from utils import filelist_to_df
from collections import Counter


class SequenceDataset(data.Dataset):
    def __init__(self, path_to_lmdb,
                 path_to_annots,
                 path_to_info,
                 mode,  # 'test' or 'train'
                 label_type='coarse', # ['coarse', 'label'],
                 img_tmpl="{video}/{view}/{view}_{frame:010d}.jpg",
                 fps=30,
                 args=None):
        """
            Inputs:
                path_to_lmdb: path to the folder containing the LMDB dataset
                path_to_csv: path to training/validation csv
                label_type: which label to return (coarse = label, fine = remark)
                img_tmpl: image template to load the features
                fps: framerate
        """

        with open(path_to_info, 'r') as f:
                splits = json.load(f)
                files = splits[f'{mode}_session_set']
                files = [f + '.csv' for f in files]
        self.annotations = filelist_to_df(path_to_annots, files)
        self.annotations = self.annotations.reset_index(drop=True)
        self.num_samples_per_class = self.annotations[label_type].value_counts().sort_index().tolist()
        print(self.num_samples_per_class)
        self.task = args.task
        self.view = args.view

        self.path_to_lmdb = path_to_lmdb
        self.fps = fps
        self.label_type = label_type
        self.img_tmpl = img_tmpl

        self.recent_sec1 = args.recent_sec1
        self.recent_sec2 = args.recent_sec2
        self.recent_sec3 = args.recent_sec3
        self.recent_sec4 = args.recent_sec4
        self.recent_dim = args.recent_dim

        self.spanning_sec = args.spanning_sec
        self.span_dim1 = args.span_dim1
        self.span_dim2 = args.span_dim2
        self.span_dim3 = args.span_dim3

        self.feat_dim = args.video_feat_dim

        self.debug_on = args.debug_on

        # initialize some lists
        self.ids = []  # mistake ids
        self.discarded_ids = []  # list of ids discarded (e.g., if there were
        # no enough frames before the beginning of the mistake
        self.discarded_labels = [] # list of labels discarded (e.g., if there 
        # were no enough frames before the beginning of the mistake
        self.recent_frames = []  # recent past
        self.spanning_frames = []  # spanning past
        self.labels = []  # labels of each mistake

        # populate them
        self.__populate_lists()

        # if a list to datasets has been provided, load all of them
        if isinstance(self.path_to_lmdb, list):
            self.env = [lmdb.open(l_m, readonly=True, lock=False) for l_m in self.path_to_lmdb]
        else:
            # otherwise, just load the single LMDB dataset
            self.env = lmdb.open(self.path_to_lmdb, readonly=True, lock=False)

    def __populate_lists(self):
        count_debug = 0
        """ Samples a sequence for each mistake and populates the lists. """
        for _, a in tqdm(self.annotations.iterrows(), 'Populating Dataset', total=len(self.annotations)):
            count_debug += 1
            if self.debug_on:
                if count_debug > 20:
                    break

            # sample frames before the beginning of the mistake
            recent_f, spanning_f = self.__get_snippet_features(a.start, a.end, a.video)

            # check if there were enough frames before the beginning of the mistake
            # if the smaller frame is at least 1, the sequence is valid
            if spanning_f is not None and recent_f is not None:
                self.spanning_frames.append(spanning_f)
                self.recent_frames.append(recent_f)
                self.ids.append(a.name)

                # handle whether a list of labels is required (e.g., [label, remark]), rather than a single label
                if isinstance(self.label_type, list):
                    self.labels.append(a[self.label_type].values.astype(int))
                else:  # single label version
                    self.labels.append(a[self.label_type])
            else:
                # if the sequence is invalid, do nothing, but add the id to the discarded_ids list
                self.discarded_ids.append(a.name)
                if isinstance(self.label_type, list):
                    self.discarded_labels.append(a[self.label_type].values.astype(int))
                else: #single label version
                    self.discarded_labels.append(a[self.label_type])

        print(f'Number of mistakes: {len(self.ids)}')
        print(f'Number of discarded mistakes: {len(self.discarded_ids)}')



    def __get_snippet_features(self, point_start, point_end, video):

        # Spanning snippets
        start_spanning = max(point_start - (self.spanning_sec * self.fps), 0)
        if self.task == 'online':
            end_recent1 = end_recent2 = end_recent3 = end_recent4 = end_spanning = point_end # CHANGE made here to make online
        else:
            end_spanning = point_end + (self.spanning_sec * self.fps)
            # Recent snippets
            end_recent1 = int(point_end + (self.recent_sec1 * self.fps))
            end_recent2 = int(point_end + (self.recent_sec2 * self.fps))
            end_recent3 = int(point_end + (self.recent_sec3 * self.fps))
            end_recent4 = int(point_end + (self.recent_sec4 * self.fps))

        select_spanning_frames1 = np.linspace(start_spanning, end_spanning, self.span_dim1 + 1, dtype=int)
        select_spanning_frames2 = np.linspace(start_spanning, end_spanning, self.span_dim2 + 1, dtype=int)
        select_spanning_frames3 = np.linspace(start_spanning, end_spanning, self.span_dim3 + 1, dtype=int)

        spanning_past = [self.__get_frames_from_indices(video, select_spanning_frames1),
                         self.__get_frames_from_indices(video, select_spanning_frames2),
                         self.__get_frames_from_indices(video, select_spanning_frames3)]

        # Recent snippets
        start_recent1 = int(max(point_start - (self.recent_sec1 * self.fps), 0))
        start_recent2 = int(max(point_start - (self.recent_sec2 * self.fps), 0))
        start_recent3 = int(max(point_start - (self.recent_sec3 * self.fps), 0))
        start_recent4 = int(max(point_start - (self.recent_sec4 * self.fps), 0))

        select_recent_frames1 = np.linspace(start_recent1, end_recent1, self.recent_dim + 1, dtype=int)
        select_recent_frames2 = np.linspace(start_recent2, end_recent2, self.recent_dim + 1, dtype=int)
        select_recent_frames3 = np.linspace(start_recent3, end_recent3, self.recent_dim + 1, dtype=int)
        select_recent_frames4 = np.linspace(start_recent4, end_recent4, self.recent_dim + 1, dtype=int)

        recent_past = [self.__get_frames_from_indices(video, select_recent_frames1),
                       self.__get_frames_from_indices(video, select_recent_frames2),
                       self.__get_frames_from_indices(video, select_recent_frames3),
                       self.__get_frames_from_indices(video, select_recent_frames4)]

        return recent_past, spanning_past

    def __get_frames_from_indices(self, video, indices):
        list_data = []
        for kkl in range(len(indices) - 1):
            cur_start = np.floor(indices[kkl]).astype('int')
            cur_end = np.floor(indices[kkl + 1]).astype('int')
            list_frames = list(range(cur_start, cur_end + 1))
            list_data.append(self.__get_frames(list_frames, video))
        return list_data

    def __get_frames(self, frames, video):
        """ format file names using the image template """
        frames = np.array(list(map(lambda frame: self.img_tmpl.format(video=video, view=self.view, frame=frame), frames)))
        return frames

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """ sample a given sequence """

        # get spanning and recent frames
        spanning_frames = self.spanning_frames[index]
        recent_frames = self.recent_frames[index]

        # return a dictionary containing the id of the current sequence
        # this is useful to produce the jsons for the challenge
        out = {'id': self.ids[index]}

        # read representations for spanning and recent frames
        out['recent_features'], out['spanning_features'] = read_data(recent_frames, spanning_frames, self.env,
                                                                     self.feat_dim)

        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label

        return out


def read_representations(recent_frames, spanning_frames, env, feat_dim):
    """ Reads a set of representations, given their frame names and an LMDB environment."""

    recent_features1 = []
    recent_features2 = []
    recent_features3 = []
    recent_features4 = []
    spanning_features1 = []
    spanning_features2 = []
    spanning_features3 = []
    for e in env:
        spanning_features1.append(get_max_pooled_features(e, spanning_frames[0], feat_dim))
        spanning_features2.append(get_max_pooled_features(e, spanning_frames[1], feat_dim))
        spanning_features3.append(get_max_pooled_features(e, spanning_frames[2], feat_dim))

        recent_features1.append(get_max_pooled_features(e, recent_frames[0], feat_dim))
        recent_features2.append(get_max_pooled_features(e, recent_frames[1], feat_dim))
        recent_features3.append(get_max_pooled_features(e, recent_frames[2], feat_dim))
        recent_features4.append(get_max_pooled_features(e, recent_frames[3], feat_dim))

    spanning_features1 = np.concatenate(spanning_features1, axis=-1)
    spanning_features2 = np.concatenate(spanning_features2, axis=-1)
    spanning_features3 = np.concatenate(spanning_features3, axis=-1)

    recent_features1 = np.concatenate(recent_features1, axis=-1)
    recent_features2 = np.concatenate(recent_features2, axis=-1)
    recent_features3 = np.concatenate(recent_features3, axis=-1)
    recent_features4 = np.concatenate(recent_features4, axis=-1)

    spanning_snippet_features = [spanning_features1, spanning_features2, spanning_features3]
    recent_snippet_features = [recent_features1, recent_features2, recent_features3, recent_features4]

    return recent_snippet_features, spanning_snippet_features


def get_max_pooled_features(env, frame_names, feat_dim):
    list_features = []
    missing_features = []

    for kkl in range(len(frame_names)):
        with env.begin() as e:
            pool_list = []
            for name in frame_names[kkl]:
                dd = e.get(name.strip().encode('utf-8'))
                if dd is None:
                    continue
                data_curr = np.frombuffer(dd, 'float32')  # convert to numpy array
                feat_dim = data_curr.shape[0]
                pool_list.append(data_curr)

            if len(pool_list) == 0:  # Missing frames indices
                missing_features.append(kkl)
                list_features.append(np.zeros(feat_dim, dtype='float32'))
            else:
                max_pool = np.max(np.array(pool_list), 0)
                list_features.append(max_pool.squeeze())

    for index in missing_features[::-1]: 
        list_features[index] = list_features[0]
        
    list_features = np.stack(list_features)
    return list_features

def read_data(recent_frames, spanning_frames, env, feat_dim):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)"""

    # if env is a list
    if isinstance(env, list):
        # read the representations from all environments
        return read_representations(recent_frames, spanning_frames, env, feat_dim)
    else:
        # otherwise, just read the representations
        env = [env]
        return read_representations(recent_frames, spanning_frames, env, feat_dim)

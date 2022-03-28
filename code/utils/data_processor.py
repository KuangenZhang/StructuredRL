#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:47:19 2020

@author: kuangen
"""
import argparse
import os
import json
import xml.etree.cElementTree as ET
import logging

import numpy as np
import glob

from scipy.spatial.transform import Rotation as R
from scipy import signal
from utils import utils

def parse_motions(path):
    '''
    output frame data: [root_pos, root_rot, joint_positions]
    '''
    xml_tree = ET.parse(path)
    xml_root = xml_tree.getroot()
    xml_motions = xml_root.findall('Motion')
    motions = []

    if len(xml_motions) > 1:
        logging.warn('more than one <Motion> tag in file "%s", only parsing the first one', path)
    motions.append(_parse_motion(xml_motions[0], path))
    return motions


def _parse_motion(xml_motion, path):
    xml_joint_order = xml_motion.find('JointOrder')
    if xml_joint_order is None:
        raise RuntimeError('<JointOrder> not found')

    joint_names = []
    joint_indexes = []
    for idx, xml_joint in enumerate(xml_joint_order.findall('Joint')):
        name = xml_joint.get('name')
        if name is None:
            raise RuntimeError('<Joint> has no name')
        joint_indexes.append(idx)
        joint_names.append(name)

    atlas_joint_names = \
    ['BPz_joint', 'BPx_joint', 'BPy_joint', 'BTz_joint', 'BTx_joint', 'BTy_joint',
     'LSz_joint', 'LSy_joint', 'LEz_joint', 'LEx_joint', 'LWy_joint', 'LWy_joint', 'LWy_joint',
    'BUNx_joint', 'RSz_joint', 'RSy_joint', 'REz_joint', 'REx_joint',
    'RWy_joint', 'RWy_joint', 'RWy_joint', 'LHz_joint', 'LHy_joint',
    'LHx_joint', 'LKx_joint', 'LAx_joint', 'LAy_joint', 'RHz_joint',
    'RHy_joint', 'RHx_joint', 'RKx_joint', 'RAx_joint', 'RAy_joint']
    atlas_joint_indices = []
    for atlas_joint_name in atlas_joint_names:
        atlas_joint_indices.append(joint_names.index(atlas_joint_name))

    frames = []
    xml_frames = xml_motion.find('MotionFrames')
    if xml_frames is None:
        raise RuntimeError('<MotionFrames> not found')
    for xml_frame in xml_frames.findall('MotionFrame'):
        frames.append(_parse_frame(xml_frame, joint_indexes, atlas_joint_indices))

    return joint_names, frames


def _parse_frame(xml_frame, joint_indexes, atlas_joint_indices):

    '''
    mimic_data_quat = [frame_time (1), root xyz (3), root orientation (4), chest (4), neck (4), rightHip (4),
    rightKnee (1), rightAnkle (4), rightShoulder (4), rightElbow (1), leftHip (4), leftKnee (1),
    leftAnkle (4), leftShoulder (4), leftElbow (1)]

    mimic_data_euler= [frame_time (1), root xyz (3), root orientation (3, 4:7), chest (3, 7:10), neck (3, 10:13),
    rightHip (3, 13:16), rightKnee (1), rightAnkle (3, 17:20), rightShoulder (3, 20:23), rightElbow (1),
    leftHip (3, 24:27), leftKnee (1), leftAnkle (3, 28:31), leftShoulder (3, 31:34), leftElbow (1)]
    '''

    n_joints = len(joint_indexes)
    xml_joint_pos = xml_frame.find('JointPosition')
    if xml_joint_pos is None:
        raise RuntimeError('<JointPosition> not found')
    joint_pos = _parse_list(xml_joint_pos, n_joints, joint_indexes)

    xml_root_pos = xml_frame.find('RootPosition')
    if xml_root_pos is None:
        raise RuntimeError('<RootPosition> not found')
    root_pos = _parse_list(xml_root_pos, 3)

    xml_root_rot = xml_frame.find('RootRotation')
    if xml_root_rot is None:
        raise RuntimeError('<RootRotation> not found')
    root_rot = _parse_list(xml_root_rot, 3)

    # # convert motion data to mimic data
    # joint_pos_mimic = joint_pos[6:12] + joint_pos[33:37] + joint_pos[28:31] + \
    # joint_pos[37:40] + [joint_pos[7]] + joint_pos[17:21] + joint_pos[12:15] + \
    # joint_pos[21:24] + [joint_pos[15]]
    # mimic_data_rotvec = [0.01] + root_pos + root_rot + joint_pos_mimic

    # convert motion data to atlas data
    # joint_pos_atlas = joint_pos[5, 3, 4, 23, 22, 16, 15, 25, 25, 25, 0, 39, 38, 32, 31,
    #                             41, 41, 41, 19, 18, 17, 20, 12, 13, 35, 34, 33, 36, 28, 29]
    joint_pos = np.asarray(joint_pos)

    joint_pos_atlas =[0] * 3 + \
                     list(joint_pos[atlas_joint_indices[6:8]]) + [0] * 6 + \
                     list(joint_pos[atlas_joint_indices[14:16]]) + [0] * 5 + \
                     list(joint_pos[atlas_joint_indices[21:]])

    atlas_data_rotvec = [0.01] + list(0.001 * np.asarray(root_pos)) + root_rot + joint_pos_atlas
    return atlas_data_rotvec


def _parse_list(xml_elem, length, indexes=None):
    if indexes is None:
        indexes = range(length)
    elems = [float(x) for idx, x in enumerate(xml_elem.text.rstrip().split(' ')) if idx in indexes]
    if len(elems) != length:
        raise RuntimeError('invalid number of elements')
    return elems


def mimic_quat_to_rotvec(mimic_data_quat):
    '''
            mimic_data_quat = [frame_time (1), root xyz (3), root orientation (4, 4:8), chest (4, 8:12),
            neck (4, 12:16), rightHip (4, 16:20), rightKnee (1, 20), rightAnkle (4, 21:25), rightShoulder (4, 25:29),
            rightElbow (1, 29), leftHip (4, 30:34), leftKnee (1, 34), leftAnkle (4, 35:39), leftShoulder (4, 39:43),
            leftElbow (1, 43)]

            mimic_data_rotvec = [frame_time (1), root xyz (3), root orientation (3, 4:7), chest (3, 7:10), neck (3, 10:13),
            rightHip (3, 13:16), rightKnee (1, 16), rightAnkle (3, 17:20), rightShoulder (3, 20:23), rightElbow (1, 23),
            leftHip (3, 24:27), leftKnee (1, 27), leftAnkle (3, 28:31), leftShoulder (3, 31:34), leftElbow (1, 34)]
    '''
    quat_indices = [4, 8, 12, 16, 21, 25, 30, 35, 39]
    rotvec_indices = [4, 7, 10, 13, 17, 20, 24, 28, 31]
    mimic_data_quat = np.asarray(mimic_data_quat)
    mimic_data_rotvec = np.zeros(shape=(mimic_data_quat.shape[0], 35))
    mimic_data_rotvec[:, 0] = mimic_data_quat[:, 0]
    mimic_data_rotvec[:, 1:4] = mimic_data_quat[:, 1:4]
    mimic_data_rotvec[:, [16, 23, 27, 34]] = mimic_data_quat[:, [20, 29, 34, 43]]
    for i in range(len(quat_indices)):
        # Quaternion representation to axis–angle
        rot = R.from_quat(mimic_data_quat[:, quat_indices[i]: (quat_indices[i] + 4)])
        mimic_data_rotvec[:, rotvec_indices[i]:(rotvec_indices[i] + 3)] = rot.as_rotvec()
    return mimic_data_rotvec

def mimic_rotvec_to_quat(mimic_data_rotvec):
    '''
        mimic: x, y, z = mmm: x, z, y
        mimic_data_quat = [frame_time (1), root xyz (3), root orientation (4, 4:8), chest (4, 8:12),
        neck (4, 12:16), rightHip (4, 16:20), rightKnee (1), rightAnkle (4, 21:25), rightShoulder (4, 25:29),
        rightElbow (1), leftHip (4, 30:34), leftKnee (1), leftAnkle (4, 35:39), leftShoulder (4, 39:43),
        leftElbow (1)]

        mimic_data_rotvec = [frame_time (1), root xyz (3), root orientation (3, 4:7), chest (3, 7:10), neck (3, 10:13),
        rightHip (3, 13:16), rightKnee (1), rightAnkle (3, 17:20), rightShoulder (3, 20:23), rightElbow (1),
        leftHip (3, 24:27), leftKnee (1), leftAnkle (3, 28:31), leftShoulder (3, 31:34), leftElbow (1)]
    '''
    quat_indices = [4, 8, 12, 16, 21, 25, 30, 35, 39]
    rotvec_indices = [4, 7, 10, 13, 17, 20, 24, 28, 31]
    mimic_data_rotvec = np.asarray(mimic_data_rotvec)
    mimic_data_quat = np.zeros(shape=(mimic_data_rotvec.shape[0], 44))
    mimic_data_quat[:, 0] = mimic_data_rotvec[:, 0]
    # mimic_data_quat[:, 1:4] = 0.001 * (mimic_data_rotvec[:, 1:4] - mimic_data_rotvec[[0], 1:4])
    mimic_data_quat[:, [1, 3]] = 0.001 * (mimic_data_rotvec[:, [1, 2]] - mimic_data_rotvec[[0], [1, 2]])
    mimic_data_quat[:, [2]] = 0.001 * mimic_data_rotvec[:, [3]]
    # mimic_data_quat[:, 2] = 1
    mimic_data_rotvec[:, 1:4] = mimic_data_quat[:, 1:4]
    mimic_data_quat[:, [20, 29, 34, 43]] = mimic_data_rotvec[:, [16, 23, 27, 34]]
    # mimic_data_rotvec_new[:, [1, 3]] = -0.001 * (mimic_data_rotvec[:, [1, 2]] -
    #                                              mimic_data_rotvec[[0], [1, 2]])
    # mimic_data_rotvec_new[:, [2]] = 0.001 * mimic_data_rotvec[:, [3]]
    # mimic_data_rotvec_new[:, [4]] = mimic_data_rotvec[:, [4]] - np.pi
    # mimic_data_rotvec_new[:, [6]] = mimic_data_rotvec[:, [6]] - mimic_data_rotvec[[0], [6]]
    for i in range(len(rotvec_indices)):
       # Axis–angle representation to quaternion
       rot_mat = np.copy(mimic_data_rotvec[:, rotvec_indices[i]: (rotvec_indices[i] + 3)])
       rot_mat[:, 0] = mimic_data_rotvec[:, rotvec_indices[i]] - np.pi
       rot_mat[:, 2] = mimic_data_rotvec[:, rotvec_indices[i] + 1] - mimic_data_rotvec[[0], rotvec_indices[i] + 1]
       rot_mat[:, 1] = mimic_data_rotvec[:, rotvec_indices[i] + 2] - mimic_data_rotvec[[0], rotvec_indices[i] + 2]

       mimic_data_rotvec[:, rotvec_indices[i]: (rotvec_indices[i] + 3)] = rot_mat
       rot = R.from_rotvec(mimic_data_rotvec[:, rotvec_indices[i]:(rotvec_indices[i]+3)])
       mimic_data_quat[:, quat_indices[i]: (quat_indices[i] + 4)] = rot.as_quat()
    return mimic_data_quat, mimic_data_rotvec

def load_mmm_as_atlas(file_name):
    joint_names, atlas_data_rotvec = parse_motions(file_name)[0]
    # Initial data are not stable.
    # Smooth the motion data.
    atlas_data_rotvec = np.asarray(atlas_data_rotvec)
    atlas_data_rotvec[:, 1:] = signal.medfilt2d(atlas_data_rotvec[:, 1:], kernel_size=(5, 1))
    original_indices = [10, 11, 18, 19, 25, 26, 31, 32]
    atlas_original_rotvec = np.copy(atlas_data_rotvec[:, original_indices])
    atlas_data_rotvec[:, 7:] -= atlas_data_rotvec[[-1], 7:]
    atlas_data_rotvec[:, original_indices] = atlas_original_rotvec

    root_pose = atlas_data_rotvec[:, [2, 1, 3, 5, 4, 6]]
    root_pose[:, 0:5] -= root_pose[[-1], 0:5]  # offset final pose except the yaw
    root_pose[:, [1, 4]] = -root_pose[:, [1, 4]]
    root_pose[:, 2] += 0.97
    atlas_data_rotvec[:, 1:7] = root_pose
    vec_len = atlas_data_rotvec.shape[0]
    return atlas_data_rotvec[int(0.1*vec_len):int(0.9*vec_len)]


def convert_atlas_to_2d_walk(atlas_data_rotvec):
    '''
    :param atlas_data_rotvec: [0: time_step, 1:7: root_pose: x, y, z, q_x, q_y, q_z]
    :return: walker_2d_rot_vec: y, z, q_x, q_r_hip, q_r_knee, q_r_ankle, q_l_hip, q_l_knee, q_l_ankle
    '''
    walker_2d_rot_vec = np.c_[atlas_data_rotvec[:, 2:5], atlas_data_rotvec[:, 33:36], atlas_data_rotvec[:, 27:30]]
    walker_2d_rot_vec[:, 2] = -walker_2d_rot_vec[:, 2]
    walker_2d_rot_vec[:, 1] += -walker_2d_rot_vec[[0], 1] + 0.1
    return walker_2d_rot_vec

def save_2d_walk_net_data(folder = '../data/im_motion/walk'):
    file_list = glob.glob(folder + '/*.npy')
    state_all = None
    action_all = None
    skip_step = 2
    time_step = 0.01 * skip_step
    for file_name in file_list:
        walker_2d_rot_vec = np.load(file_name)
        walker_2d_rot_vec = walker_2d_rot_vec[::skip_step]
        #[z-z0, cos(error_yaw), sin(error_yaw), v_x, v_y, v_z, roll, pitch, joint angles, joint velocity, foot pressures]
        state = np.zeros((walker_2d_rot_vec.shape[0] - 2, 22))
        state[:, 0] = walker_2d_rot_vec[1:-1, 0] - walker_2d_rot_vec[1, 0] # z-z0
        state[:, 1] = 1 # cos(error_yaw)
        state[:, [5, 3]] = 0.3 * (walker_2d_rot_vec[1:-1, 0:2] - walker_2d_rot_vec[:-2, 0:2]) / time_step # 0.3 * vx, 0.3 * vz
        state[:, 7] = -walker_2d_rot_vec[1:-1, 2] # pitch
        state[:, 8:-2:2] = utils.normalize_angle(walker_2d_rot_vec[1:-1, -6:]) # joint angles
        state[:, 9:-2:2] = 0.1 * ((walker_2d_rot_vec[1:-1, -6:] - walker_2d_rot_vec[:-2, -6:]) / time_step) # joint speed
        if state_all is None:
            state_all = state
        else:
            state_all = np.r_[state_all, state]

        action = utils.normalize_angle(walker_2d_rot_vec[2:, -6:])  # 2:, represent the next time step

        if action_all is None:
            action_all = action
        else:
            action_all = np.r_[action_all, action]
    data = {'state': state_all, 'action': action_all}
    np.save(folder + '_dataset.npy', data, allow_pickle=True)


def save_walker_2d_data(folder = '../data/im_motion/walk'):
    file_list = glob.glob(folder + '/*.xml')
    for file_name in file_list:
        atlas_data_rotvec = load_mmm_as_atlas(file_name)
        walker_2d_rot_vec = convert_atlas_to_2d_walk(atlas_data_rotvec)
        np.save(file_name + 'walker_2d.npy', walker_2d_rot_vec)
    save_2d_walk_net_data(folder)

def save_atlas_data(folder = '../data/im_motion/walk'):
    file_list = glob.glob(folder + '/*.xml')
    for file_name in file_list:
        atlas_data_rotvec = load_mmm_as_atlas(file_name)
        np.save(file_name + 'motion.npy', atlas_data_rotvec)

def load_mmm_as_mimic(file_name):
    joint_names, mimic_data_rotvec = parse_motions(file_name)[0]
    mimic_data_quat, _ = mimic_rotvec_to_quat(np.copy(mimic_data_rotvec))
    return mimic_data_quat, mimic_data_rotvec


def main(args):
    input_path = args.input
    
    print('Scanning files ...')
    files = glob.glob(input_path+ '/*.xml')

    # Parse all files.
    print('Processing data in "{}" ...'.format(input_path))
    all_ids = []
    all_motions = []
    reference_joint_names = None
    mimic_joint_id = []
    for idx, mmm_path in enumerate(files):
        print('  {}/{}'.format(idx + 1, mmm_path)),
        # Load motion.
#        mmm_path = file_name
        assert os.path.exists(mmm_path)
        joint_names, mimic_data_rotvec = parse_motions(mmm_path)[0]
        if reference_joint_names is None:
            reference_joint_names = joint_names[:]
        elif reference_joint_names != joint_names:
            print('skipping, invalid joint_names {}'.format(joint_names))
            continue

        all_ids.append(mmm_path)
        all_motions.append(np.array(mimic_rotvec_to_quat(mimic_data_rotvec), dtype='float32'))
        print('done')
    print('done, successfully processed {} motions and their annotations'.format(len(all_motions)))
    print('')

    # At this point, you can do anything you want with the motion and annotation data.
    return all_motions, all_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', default = '../data/beam_balance/files_motions_704',type=str,)
    all_motions, all_ids = main(parser.parse_args())
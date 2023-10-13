"""Utilities."""

import json
import numpy as np
import os


def smooth_and_subsample(data_vec,
                         kernel_half_width=0.05,
                         sample_rate=0.005,
                         print_logs=False,
                         firing_rate_sample_rate=0.001):
    """data_vec is [num_timesteps]."""

    if print_logs:
        print(f'len(data_vec): {len(data_vec)}')

    # Create kernel
    if print_logs:
        print('     creating kernel')
    kernel_half_bins = int(
        np.ceil(kernel_half_width / firing_rate_sample_rate))
    half_kernel = np.linspace(0.,1., kernel_half_bins)[1:]
    kernel = np.concatenate([half_kernel[:-1], half_kernel[::-1]])
    kernel /= np.sum(kernel)
    # Convert from spikes per bin units to Hz
    kernel /= firing_rate_sample_rate

    # Convolve with kernel
    if print_logs:
        print('    convolving')
    data_vec = np.convolve(data_vec, kernel, mode='full')

    # Normalize to remove edge effects
    if print_logs:
        print('     normalizing')
    edge_half_width = kernel_half_bins - 2
    data_vec[edge_half_width: 2 * (edge_half_width)] += (
        data_vec[:edge_half_width][::-1]
    )
    data_vec[-2 * edge_half_width: -edge_half_width] += (
        data_vec[-edge_half_width:][::-1]
    )
    data_vec = data_vec[edge_half_width: -edge_half_width]

    # Subsample
    if print_logs:
        print('     subsampling')
    sample_rate_bins = int(np.round(sample_rate / firing_rate_sample_rate))
    if print_logs:
        print(f'        sample_rate_bins = {sample_rate_bins}')
    data_vec = data_vec[::sample_rate_bins]
    if print_logs:
        print(f'        data_vec.shape = {data_vec.shape}')
    
    return data_vec, sample_rate_bins


def load_behavior_data(open_source_session_dir):
    # Load behavior data
    print('Loading behavior data')
    behavior_data_dir = os.path.join(open_source_session_dir, 'behavior')
    behavior_data = {}
    print('    broken_fixation')
    broke_fixation_path = os.path.join(
        behavior_data_dir, 'trials.broke_fixation.json')
    behavior_data['broke_fixation'] = json.load(open(broke_fixation_path, 'r'))
    print('    response_location')
    response_location_path = os.path.join(
        behavior_data_dir, 'trials.response.location.json')
    behavior_data['response_location'] = json.load(
        open(response_location_path, 'r'))
    return behavior_data
    

def load_task_data(open_source_session_dir):
    # Load task data
    print('Loading task data')
    task_data_dir = os.path.join(open_source_session_dir, 'task')
    task_data = {}
    print('    object_blanks')
    object_blanks_path = os.path.join(
        task_data_dir, 'trials.object_blanks.json')
    task_data['object_blanks'] = json.load(open(object_blanks_path, 'r'))
    print('    relative_phase_times')
    relative_phase_times_path = os.path.join(
        task_data_dir, 'trials.relative_phase_times.json')
    task_data['relative_phase_times'] = json.load(
        open(relative_phase_times_path, 'r'))
    print('    stimuli_init')
    stimuli_init_path = os.path.join(
        task_data_dir, 'trials.stimuli_init.json')
    task_data['stimuli_init'] = json.load(open(stimuli_init_path, 'r'))
    print('    reward_duration')
    reward_duration_path = os.path.join(
        task_data_dir, 'trials.reward.duration.json')
    task_data['reward_duration'] = json.load(open(reward_duration_path, 'r'))
    return task_data

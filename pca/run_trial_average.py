"""Run trial averaging on a user-specified session.

Usage:
$ python3 run_trial_average.py 2022-06-01
"""

import json
import numpy as np
import os
import utils
import sys

_ONLY_GOOD_UNITS = True  # Whether to only keep good units (single neurons)
_DATASET_DIR = os.path.join(os.getcwd(), '../data')
_WRITE_DIR = os.path.join(os.getcwd(), 'data/stimulus_delay_good_units')
_PHASES = ['stimulus', 'delay']  # Phases to trial average for
_MS_PER_PHASE = [1000, 1000]  # Milliseconds per phase
_MIN_TRIALS_PER_CONDITION = 10  # Minimum number of trials per condition
_SMOOTH_HALF_WIDTH = 0.05  # Half width of smoothing kernel in seconds
_SMOOTH_SAMPLE_RATE = 0.01  # Sampling frequency in seconds

_TRIANGLE_THETAS = [
    -5 * np.pi / 6,  # bottom-left
    -1 * np.pi / 6,  # top-left
    np.pi / 2,  # right
]


def _theta_to_location(theta, atol=0.001):
    theta = theta % (2 * np.pi)
    for i, ref in enumerate(_TRIANGLE_THETAS):
        close = (
            np.isclose(theta, ref, atol=atol) or
            np.isclose(theta + 2 * np.pi, ref, atol=atol) or
            np.isclose(theta - 2 * np.pi, ref, atol=atol)
        )
        if close:
            return i
    return None


def _process_neuron(stimuli_init, response_location, neural_data, write_path):

    # Compute responses
    trials_dict = {
        stim: []
        for stim in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    }
    
    for trial_index in range(len(stimuli_init)):
        stim_init = stimuli_init[trial_index]
        
        # Get object locations
        obj_thetas = np.array([
            np.arctan2(x['x'] - 0.5, x['y'] - 0.5) % (2 * np.pi)
            for x in stim_init
        ])
        locations = [
            _theta_to_location(theta, atol=0.001) for theta in obj_thetas
        ]
        
        # Get target location
        for i, s in enumerate(stim_init):
            if s['target']:
                target_index = i
        target_loc = locations[target_index]
        
        # Get response location and reject if no response
        response = response_location[trial_index]
        if response is None:
            continue
        response_loc = _theta_to_location(
            np.arctan2(response[0] - 0.5, response[1] - 0.5), atol=0.5 * np.pi)
        
        # Reject if not on triangle
        if np.any([x is None for x in locations]):
            continue
        else:
            stim_locations = tuple(sorted(locations))
            
        # Reject if neural is None
        neural_per_phase = [
            neural_data[phase][trial_index] for phase in _PHASES
        ]
        if any(x is None for x in neural_per_phase):
            continue
        
        # Reject if not successful trial
        if response_loc != target_loc:
            continue
        
        # Concatenate neural data across phases
        neural = np.zeros(sum(_MS_PER_PHASE))
        time_index = 0
        for ms, neur in zip(_MS_PER_PHASE, neural_per_phase):
            rel_end = min(ms, len(neur))
            neural[time_index: time_index + rel_end] = neur[:rel_end]
            time_index += ms
            
        # Add trial to trials_dict
        trials_dict[stim_locations].append(neural)
    
    # Average and smooth neural
    average_dict = {}
    for k, v in trials_dict.items():
        if len(v) < _MIN_TRIALS_PER_CONDITION:
            return
        average_v = np.sum(v, axis=0) / len(v)
        smooth_neural, _ = utils.smooth_and_subsample(
            average_v, kernel_half_width=_SMOOTH_HALF_WIDTH,
            sample_rate=_SMOOTH_SAMPLE_RATE,
        )
        average_dict[k] = smooth_neural
    
    # Write data
    np.save(write_path, average_dict)


def main(session):
    print(f'session: {session}')
    dataset_dir = os.path.join(_DATASET_DIR, session)
    write_dir = os.path.join(_WRITE_DIR, session)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
        
    # Load stimuli and response location
    response_location_path = os.path.join(
        dataset_dir, 'trials.response.location.json')
    response_location = json.load(open(response_location_path, 'r'))
    stimuli_init_path = os.path.join(dataset_dir, 'trials.stimuli_init.json')
    stimuli_init = json.load(open(stimuli_init_path, 'r'))

    # Iterate through probes
    probe_names = [x for x in os.listdir(dataset_dir) if x[:5] == 'probe']
    for probe_name in probe_names:
        print(f'probe_name: {probe_name}')

        # Load cluster labels
        probe_dir = os.path.join(dataset_dir, probe_name)
        print('Loading behavior data')
        clusters_labels_path = os.path.join(probe_dir, 'clusters.labels.tsv')
        clusters_labels = np.genfromtxt(
            clusters_labels_path,
            delimiter='\t',
            skip_header=True,
            dtype=str,
        )
        clusters_labels = {
            int(cluster_id): label for cluster_id, label in clusters_labels
        }
        unit_ids = list(clusters_labels.keys())
        
        # Keep only good units if necessary
        if _ONLY_GOOD_UNITS:
            unit_ids = [x for x in unit_ids if clusters_labels[x] == 'good']

        # Load neural data
        print('Loading neural data')
        neural_data = {}
        for phase in _PHASES:
            probe_phase_dir = os.path.join(probe_dir, phase)
            neural_data_phase = {}
            for cluster_id in unit_ids:
                cluster_data_path = os.path.join(
                    probe_phase_dir, str(cluster_id) + '.json')
                neural_data_phase[cluster_id] = json.load(
                    open(cluster_data_path, 'r'))
            neural_data[phase] = neural_data_phase
        
        # Iterate through neurons, processing each one
        for unit_id in unit_ids:
            print(f'    Processing unit {unit_id}')
            neural = {phase: neural_data[phase][unit_id] for phase in _PHASES}
            name = f'{probe_name}_{unit_id}'
            write_path = os.path.join(write_dir, name)
            _process_neuron(stimuli_init, response_location, neural, write_path=write_path)


if __name__ == "__main__":
    session = sys.argv[1]
    main(session)

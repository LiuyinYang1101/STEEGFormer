# Vendored from the Nieto et al. "Thinking out loud" inner-speech tutorial repository
# (dataset: OpenNeuro ds003626; https://github.com/N-Nieto/Inner_Speech_Dataset and the
# accompanying eeg-tutorials). Provides loaders for the dataset's pre-epoched *derivatives*
# (*_eeg-epo.fif / *_events.dat) plus the event/label helpers used by preprocess.py.
# Unmodified except for this header; see the upstream repo for licensing/attribution.
# ---------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# conda-env: mne

import warnings
import pickle
from itertools import product

import mne
import numpy as np
import pandas as pd

mne.set_log_level(verbose='warning') # to avoid info at terminal
warnings.filterwarnings(action="ignore", category=DeprecationWarning ) 
warnings.filterwarnings(action="ignore", category=FutureWarning )
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Load Nieto dataset in various formats

def load_trials(root_dir, subjects=None, sessions=None):
    
    # dataframe
    cols = ['sub', 'ses', 'eeg', 'exg', 'sample', 'label', 'mode']
    dct = {name: [] for name in cols}   
    
    # dataset structure
    if subjects is None: subjects = np.arange(1, 11)
    if sessions is None: sessions = np.arange(1, 4)
    subject_names = [f'sub-{i:02d}' for i in subjects]
    session_names = [f'ses-{i:02d}' for i in sessions]
    
    def update_dct(dct, sub, ses, eeg, exg, events):
    
        dct['sub'].extend([sub] * len(events))
        dct['ses'].extend([ses] * len(events))  
        dct['eeg'].extend(eeg)
        dct['exg'].extend(exg) 
        dct['sample'].extend(events[:, 0]) 
        dct['label'].extend(events[:, 1]) 
        dct['mode'].extend(events[:, 2])   

        return dct     
    
    # load data into dataframe
    for sub, subn in zip(subjects, subject_names):
        for ses, sesn in zip(sessions, session_names):            
            file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'
                
            # eeg
            file_pth = file_dir + subn + '_' + sesn + '_eeg-epo.fif'
            eeg = mne.read_epochs(file_pth, verbose='WARNING')._data
            
            # exg
            file_pth = file_dir + subn + '_' + sesn + '_exg-epo.fif'
            exg = mne.read_epochs(file_pth, verbose='WARNING')._data
            
            # events
            file_pth = file_dir + subn + '_' + sesn + '_events.dat'
            events = np.load(file_pth, allow_pickle=True)                 
            
            dct = update_dct(dct, sub, ses, eeg, exg, events)
                    
    return pd.DataFrame.from_dict(dct)

def load_session_baselines(root_dir, subjects=None, sessions=None):
    
    # dataframe
    cols = ['sub', 'ses', 'baseline']
    dct = {name: [] for name in cols}   
    
    # dataset structure
    if subjects is None: subjects = np.arange(1, 11)
    if sessions is None: sessions = np.arange(1, 4)
    subject_names = [f'sub-{i:02d}' for i in subjects]
    session_names = [f'ses-{i:02d}' for i in sessions]
    
    def update_dct(dct, sub, ses, baseline):
    
        dct['sub'].append(sub)
        dct['ses'].append(ses)  
        dct['baseline'].append(baseline)

        return dct     
    
    # load data into dataframe
    for sub, subn in zip(subjects, subject_names):
        for ses, sesn in zip(sessions, session_names):            
            file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'
                
            # baseline
            file_pth = file_dir + subn + '_' + sesn + '_baseline-epo.fif'
            baseline = mne.read_epochs(file_pth, verbose='WARNING')._data         
            
            dct = update_dct(dct, sub, ses, baseline)
    
    # split eeg and exg
    df = pd.DataFrame.from_dict(dct)
    split_eeg_exg = lambda x: np.split(x, [128, -1], axis=0)
    df['baseline'] = df['baseline'].apply(np.squeeze)
    df['eeg'], df['exg'], _ = zip(*df['baseline'].apply(split_eeg_exg))
      
    return df

def load_derivatives(root_dir, subjects=None, sessions=None):
    
    # dataframe
    dct = {'sub': [], 'ses': [], 'dtype': [], 'data': []}    
    
    # dataset structure
    if subjects is None: subjects = np.arange(1, 11)
    if sessions is None: sessions = np.arange(1, 4)
    subject_names = [f'sub-{i:02d}' for i in subjects]
    session_names = [f'ses-{i:02d}' for i in sessions]
    dtype_names = ['eeg', 'exg', 'baseline', 'events', 'report'] 
    
    def update_dct(dct, sub, ses, dtype, data):
        dct['sub'].append(sub)
        dct['ses'].append(ses)
        dct['dtype'].append(dtype)
        dct['data'].append(data)  
        return dct     
    
    # load data into dataframe
    for sub, subn in zip(subjects, subject_names):
        for ses, sesn in zip(sessions, session_names):            
            for dtype in dtype_names:
                file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'
                
                # derrivatives
                if dtype in ['eeg', 'exg', 'baseline']:
                    file_pth = file_dir + subn + '_' + sesn + '_' + dtype + '-epo.fif'
                    data = mne.read_epochs(file_pth, verbose='WARNING')._data
                    dct = update_dct(dct, sub, ses, dtype, data)
                
                # events
                if dtype in ['events']:
                    file_pth = file_dir + subn + '_' + sesn + '_events.dat'
                    data = np.load(file_pth, allow_pickle=True)
                    dct = update_dct(dct, sub, ses, dtype, data)
                    
                # reports
                if dtype in ['report']:
                    file_pth = file_dir + subn + '_' + sesn + '_report.pkl'
                    with open(file_pth, 'rb') as input:
                        data = pickle.load(input)
                    dct = update_dct(dct, sub, ses, dtype, data)
                    
    return pd.DataFrame.from_dict(dct)

def load_raw(root_dir, subjects=None, sessions=None):
    
    # dataframe
    dct = {'sub': [], 'ses': [], 'dtype': [], 'data': []}    
    
    # dataset structure
    if subjects is None: subjects = np.arange(1, 11)
    if sessions is None: sessions = np.arange(1, 4)
    subject_names = [f'sub-{i:02d}' for i in subjects]
    session_names = [f'ses-{i:02d}' for i in sessions]
    
    def update_dct(dct, sub, ses, dtype, data):
        dct['sub'].append(sub)
        dct['ses'].append(ses)
        dct['data'].append(data)  
        return dct     
    
    # load data into dataframe
    for sub, subn in zip(subjects, subject_names):
        for ses, sesn in zip(sessions, session_names):    
            file_dir = root_dir + '/' + subn + '/' + sesn + '/eeg/'

            # raw data
            file_pth = file_dir + subn + '_' + sesn + '_task-innerspeech_eeg.bdf'
            data = mne.io.read_raw_bdf(input_fname=file_pth, preload=True,verbose='WARNING')
            dct = update_dct(dct, sub, ses, 'eeg', data)
            
    return pd.DataFrame.from_dict(dct)

def load_subject_info(root_dir, subjects=None, sessions=None):
   
    # dataset structure
    if subjects is None: subjects = np.arange(1, 11)
    if sessions is None: sessions = np.arange(1, 4)
    subject_names = [f'sub-{i:02d}' for i in subjects]
    session_names = [f'ses-{i:02d}' for i in sessions]
        
    # load data into dataframe
    dcts = []
    for sub, subn in zip(subjects, subject_names):   
        for ses, sesn in zip(sessions, session_names):   
            sesn = 'ses-01'        
            file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'              
            file_pth = file_dir + subn + '_' + sesn + '_report.pkl'
            with open(file_pth, 'rb') as input:
                data = pickle.load(input)
            data['sub'] = sub
            data['ses'] = ses
            dcts.append(data)       
                            
    return pd.DataFrame(dcts)

def load_mne(root_dir, subject, session):
    
    subn = f'sub-{subject:02d}'
    sesn = f'ses-{session:02d}'
      
    file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'
        
    # eeg, exg, baseline, events
    file_pth = file_dir + subn + '_' + sesn + '_eeg-epo.fif'
    eeg = mne.read_epochs(file_pth, verbose='WARNING')
    
    # exg
    file_pth = file_dir + subn + '_' + sesn + '_exg-epo.fif'
    exg = mne.read_epochs(file_pth, verbose='WARNING')
    
    # baseline
    file_pth = file_dir + subn + '_' + sesn + '_baseline-epo.fif'
    baseline = mne.read_epochs(file_pth, verbose='WARNING')
        
    # events
    file_pth = file_dir + subn + '_' + sesn + '_events.dat'
    events = np.load(file_pth, allow_pickle=True)                 
        
    return eeg, exg, baseline, events

def load_events(root_dir, subject=None, session=None):
    
    subn = f'sub-{subject:02d}'
    sesn = f'ses-{session:02d}' 
    file_dir = root_dir + '/derivatives/' + subn + '/' + sesn + '/'
    file_pth = file_dir + subn + '_' + sesn + '_events.dat'
    events = np.load(file_pth, allow_pickle=True)                 
        
    return events

def load_bdf(root_dir, subject=None, session=None):
    
    subn = f'sub-{subject:02d}'
    sesn = f'ses-{session:02d}' 
    file_dir = root_dir + '/' + subn + '/' + sesn + '/eeg/'
    file_pth = file_dir + subn + '_' + sesn + '_task-innerspeech_eeg.bdf'
    data = mne.io.read_raw_bdf(input_fname=file_pth, preload=True,verbose='WARNING')
      
    return data

# get features

def params2names(param_dict):
    return param_dict

def concat_pkl_dataframes(names, cols, name2file_fnc=lambda x: str(x)):
    dfs = []
    for name in names:
        df = pd.read_pickle(name2file_fnc(name))
        df = df.filter(cols)
        dfs.append(df)
    return pd.concat(dfs, keys=names)

# dictionaries
LABEL_DICT = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
MODE_DICT = {'spoken': 0, 'imagined': 1, 'visual': 2}
EXG_DICT = {
        'earlobe': [0, 1], 
        'temple': [2, 3],
        'eye': [4, 5],
        'mouth': [6,7]
        }
DETAIL_EXG_DICT = {
        'earlobe left': 0, 'earlobe right': 1,
        'temple left': 2, 'temple right': 3,
        'eye right above': 4, 'eye right below': 5,
        'orbicularis oris right superior': 6, 'orbicularis oris right superior': 7
        }

def label_dct(): return LABEL_DICT
def mode_dct(): return MODE_DICT
def exg_dct(): return EXG_DICT
def detail_exg_dct(): return DETAIL_EXG_DICT

# channels

def get_montage():
    # montage has attributes ch_names and dis
    return mne.channels.make_standard_montage("biosemi128")

def get_channelnames():
    
    return ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31', 'B32', 
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']

# reformat events (Nieto et al.)

def update_events(events, tup2int_dct):
    tup_labels = list(zip(events[:,1], events[:,2]))
    int_labels = list(map(tup2int_dct.get, tup_labels))
    events_ = [events[:,0], np.zeros_like(events[:,0]), int_labels]
    return np.stack(events_,  axis=-1)

def add_baseline_event(events, raw, sub, ses, baseline_label):
    # raw events
    if (sub == 10 and ses==1):
        raw_events = mne.find_events(raw, initial_event = True, consecutive = True, min_duration = 0.002)  
    else:
        raw_events = mne.find_events(raw, initial_event=True, consecutive=True)
    # original baseline label is 13
    bevent = raw_events[np.where(raw_events[:,2] == 13)]
    bevent[:,2] = baseline_label
    return np.vstack([bevent, events])


# General utils
def merge_dicts(*args):
    """
    Map dictionairies with labels to integers (~one-hot)
    """
    keys = [list(arg.keys()) for arg in args]
    vals = [list(arg.values()) for arg in args]
    key_dct = {k: v for v, k in enumerate(product(*keys))}
    val_dct = {k: v for v, k in enumerate(product(*vals))}
    return key_dct, val_dct


def channel_distances(chpos_dct, chpos_dct_ref=None):
    if chpos_dct_ref is None: chpos_dct_ref = chpos_dct
    nch, nch_b = len(chpos_dct), len(chpos_dct_ref)
    dist = np.zeros((nch, nch_b))
    pos_dct = {i: v for i, (k, v) in enumerate(chpos_dct.items())}
    pos_dct_ref = {i: v for i, (k, v) in enumerate(chpos_dct_ref.items())}
    for i,j in product(pos_dct.keys(), pos_dct_ref.keys()):
        dist[i,j] = np.linalg.norm(pos_dct[i] - pos_dct_ref[j])
    return dist

def subsample_channels(chpos_dct, n_channels, n_neighbors=2):
    # subsample based on closest neighbors
    # iterate remove channels with lowest average distance to neighbors
    dist = channel_distances(chpos_dct)
    np.fill_diagonal(dist, np.inf)
    new_dct = chpos_dct.copy()
    
    def remove_channel(dist):
        # lowest average distance
        mean_dist = np.zeros(len(dist))
        for i, vals in enumerate(dist):
            mean_dist[i] = np.mean(np.sort(vals)[:n_neighbors])
        # find channel
        idx = np.where(mean_dist == mean_dist.min())[0][0] 
        ch = list(new_dct.keys())[idx]
        # delete channel
        for axis in range(dist.ndim): dist = np.delete(dist, idx, axis=axis)
        del new_dct[ch]
        return dist
    
    while len(new_dct) > n_channels:
        dist = remove_channel(dist)
    
    return new_dct, dist

def closest_channels(chpos_dct, chpos_dct_ref):
    
    # keep channels based on distance to reference montage
    channels, channels_ref = list(chpos_dct.keys()),list(chpos_dct_ref.keys())
    dist = channel_distances(chpos_dct, chpos_dct_ref)
    # map each channel to closest reference channel
    ch_ref_min_dist = np.argmin(dist, axis=-1) 
    ch2ref_dct = {channels[idx]: channels_ref[idx_ref] for idx, idx_ref in enumerate(ch_ref_min_dist)}
    # new chpos dict with best matches to reference montage
    ch_min_dist = np.argmin(dist, axis=0)
    new_dct = {channels[idx]: chpos_dct[channels[idx]] for idx in ch_min_dist}
    # ref2ch_dct = {channels[idx]: chpos_dct[channels[idx]] for idx in ch_min_dist}
    return new_dct, ch2ref_dct #, ref2ch_dct  

# subsample by maximizing total distance between electrodes
# subsample by maximizing minimal distance between 2 electrodes

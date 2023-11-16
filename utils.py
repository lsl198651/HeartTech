import numpy as np
import os
import torch
from helper_code import find_patient_files, load_patient_data, get_num_locations, get_locations, get_patient_id,get_murmur, get_outcome, load_wav_file
from helper_code import compare_strings, get_age, get_sex, get_height, get_weight, get_pregnancy_status


murmur_mapping_str2int = {'Present': 0, 'Unknown': 1, 'Absent': 2}
murmur_mapping_int2str = {0: 'Present', 1: 'Unknown', 2: 'Absent'}

outcome_mapping_str2int = {'Abnormal': 0, 'Normal': 1}
outcome_mapping_int2str = {0: 'Abnormal', 1: 'Normal'}

age_wise_avg_height_dict = {
    'Neonate': 49.33,
    'Infant': 63.29,
    'Child': 114.91,
    'Adolescent': 153.73,
    'Young Adult': 175.00,
    'nan': 110.80
}

age_wise_avg_weight_dict = {
    'Neonate': 3.42,
    'Infant': 7.40,
    'Child': 23.94,
    'Adolescent': 50.08,
    'Young Adult': 60.00,
    'nan': 23.63
}

def convert_label_to_int(str_label):
    return label_mapping_str2int[str_label]

def convert_label_to_str(int_label):
    return label_mapping_int2str[int_label]

def get_murmur_locations(data):
    locations = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            locations = l.split(': ')[1].split('+')
            break
    if locations is None:
        raise ValueError('No murmur location available!')
    return locations

def get_patient_recording_files(data, num_locations):
    recording_files = []
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations:
            recording_files.append(entries[2])
        else:
            break
    return recording_files
    
def load_recordings_with_labels(data_folder, 
                                included_labels=['Present', 'Absent'],
                                list=None):
    patient_files_arr, recording_files, murmurs, outcomes = [], [], [], []
    patient_files = find_patient_files(data_folder)
    for pf in patient_files:#pf是每一个txt文件路径        
        patient_data = load_patient_data(pf)
        patient_id=get_patient_id(patient_data)
        if patient_id in list:
            patient_murmur = get_murmur(patient_data)
            if patient_murmur not in included_labels:
                continue
            patient_murmur = included_labels.index(patient_murmur)
            patient_outcome = outcome_mapping_str2int[get_outcome(patient_data)]
            locations = get_locations(patient_data)
            murmur_locations = get_murmur_locations(patient_data)
            p_recordings = get_patient_recording_files(patient_data, len(locations))
            # Label recording as present if murmur can be seen in the corresponding location, otherwise label as absent
            for i in range(len(locations)):#present但是不是有杂音的听诊区，丢弃
                if included_labels[patient_murmur] == 'Present' and locations[i] not in murmur_locations:
                    continue
                else:
                    patient_files_arr.append(pf)
                    recording_files.append(os.path.join(data_folder, p_recordings[i]))
                    murmurs.append(patient_murmur)
                    outcomes.append(patient_outcome)
        else:
            continue
            
    patient_files_arr = np.array(patient_files_arr, dtype=np.str_)
    recording_files, murmurs, outcomes = np.array(recording_files, dtype=np.str_), np.array(murmurs, dtype=np.int_), np.array(outcomes, dtype=np.int_)
    
    return patient_files_arr, recording_files, murmurs, outcomes
    

def load_patient_features(data):    
    age_group = get_age(data)
    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6.
    elif compare_strings(age_group, 'Child'):
        age = 6. * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15. * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20. * 12
    else:
        # age = 0.
        age = 6. * 12
    
    sex_features = np.zeros(2, dtype=np.float32)
    sex = get_sex(data)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    
    if get_pregnancy_status(data):
        pregnancy = 1.
    else:
        pregnancy = 0.
    
    height = np.float32(get_height(data))
    weight = np.float32(get_weight(data))
    if np.isnan(height): 
        height = np.float32(age_wise_avg_height_dict[age_group])
    if np.isnan(weight): 
        weight = np.float32(age_wise_avg_weight_dict[age_group])
    
    # Number of features: 1 + 2 + 1 + 1 + 2 = 6
    features = np.hstack(([age], sex_features, [height], [weight],[pregnancy])).astype(np.float32)
    return features

# Refer to: https://github.com/rwightman/pytorch-image-models/blob/b7cb8d0337b3e7b50516849805ddb9be5fc11644/timm/utils/metrics.py#L7
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_accuracy(output, target):
    batch_size = target.size(0)
    _, pred = torch.max(output, dim = 1)
    n_correct = (pred == target).sum()
    acc = n_correct / batch_size
    return acc
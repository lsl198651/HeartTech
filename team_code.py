#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Import libraries and functions. You can change or remove them.
#
################################################################################
import csv
import os
from helper_code import *
import numpy as np
import torch
import torch.nn.functional as F

from config import *
from data import Preprocessor, PCGDataset
from torch.utils.data import DataLoader
from HMSSNet import Hierachical_MS_Net
from utils import AverageMeter, calc_accuracy, load_patient_features
from loss import LabelSmoothingCrossEntropy

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments.
#
################################################################################

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    verbose = verbose >= 1
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('train.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        train_list = [row[0] for row in reader]
    
    # Build Datasets and Loaders
    if verbose: 
        print('Loading datasets...')
    train_preprocessor = Preprocessor(**PREPROCESSING_CFG, 
                                      mode = 'train')
    
    train_dataset = PCGDataset(data_folder, 
                               preprocessor = train_preprocessor, 
                               classes = DATASET_CFG['murmur_classes'],
                               target = 'murmur',
                               train_list=train_list)
    train_loader = DataLoader(train_dataset, 
                              shuffle=True,
                              drop_last=True, 
                              **DATALOADER_CFG)
    
    if verbose:
        print('Building up Torch CNN and optimizer...')
    murmur_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_murmur_classes'], **MODEL_CFG).to(device)
    optimizer = torch.optim.AdamW(murmur_classifier.parameters(), **OPTIMIZER_CFG)
    criterion = LabelSmoothingCrossEntropy(TRAINING_CFG['label_smoothing'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-7, verbose=verbose)

    # Stage 1: Train the classifier for Murmur classification
    if verbose:
        print('Training model for murmur classification...')
    for epoch in range(TRAINING_CFG['epochs']):
        if verbose:
            print(f'Epoch {epoch} starts...')
        train_epoch(train_loader, murmur_classifier, optimizer, criterion, scheduler, device, TRAINING_CFG['print_freq'])
        if verbose:
            print('\n')
        save_challenge_model(model_folder, murmur_classifier, file_name='murmur_classifier')
        
    # # Stage 2: Train the classifier for Outcome classification
    # train_dataset.target = 'outcome'
    # outcome_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_outcome_classes'], include_patient_data=True, **MODEL_CFG).to(device)
    # optimizer = torch.optim.AdamW(outcome_classifier.parameters(), **OPTIMIZER_CFG)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, min_lr=1e-7, verbose=verbose)
    # if verbose:
    #     print('Training model for outcome classification...')
    # for epoch in range(TRAINING_CFG['epochs']):
    #     if verbose:
    #         print(f'Epoch {epoch} starts...')
    #     train_epoch(train_loader, outcome_classifier, optimizer, criterion, scheduler, device, TRAINING_CFG['print_freq'], verbose)
    #     if verbose:
    #         print('\n')
    #     save_challenge_model(model_folder, outcome_classifier, file_name='outcome_classifier')
        
    if verbose:
        print('Done.')
            
def train_epoch(dataloader, model, optimizer, criterion, scheduler=None, device='cuda', print_freq=10, verbose=False):
    model.train()
    acc_meter, loss_meter = AverageMeter(), AverageMeter()

    for i, (multi_scale_specs, patient_features, targets) in enumerate(dataloader):
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        patient_features = patient_features.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        preds = model(multi_scale_specs, patient_features)
        
        batch_loss = criterion(preds, targets.long())
        batch_acc = calc_accuracy(preds, targets)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        loss_meter.update(batch_loss.item(), batch_size)
        acc_meter.update(batch_acc.item(), batch_size)
            
        if verbose and i != 0 and i % print_freq == 0:
            print(f'Training Iteration: {i}\n '\
                  f'Loss: {loss_meter.avg:.6f} \n'\
                  f'Accuracy: {acc_meter.avg:.4%}')
            
    print(f'Training Loss: {loss_meter.avg:.6f} \n'\
          f'Accuracy: {acc_meter.avg:.4%}')
            
    if scheduler:
        scheduler.step(loss_meter.avg)


def calc_pred_locations(preds, window_size=3, interval=0.5, freq=2000):
    interval = int(interval * freq)
    window_size = int(window_size * freq)
    recording_length = window_size + (preds.shape[0] - 1) * interval
    unknown_probs = []
    
    location_preds = np.zeros((recording_length, preds.shape[1]))
    for i in range(len(preds)):
        location_preds[i*interval: i*interval + window_size, :] += preds[i]
    location_preds = np.argmax(location_preds, -1)
    return location_preds


@torch.no_grad()
def recording_murmur_diagnose(multi_scale_specs, murmur_classifier, murmur_classes, interval):
    murmur_logits = murmur_classifier(multi_scale_specs)
    murmur_probs = F.softmax(murmur_logits, -1).cpu().numpy()
    location_preds = calc_pred_locations(murmur_probs, 
                                        window_size=PREPROCESSING_CFG['length'],
                                        interval=interval,
                                        freq=PREPROCESSING_CFG['frequency'])
    class_duration = np.bincount(location_preds, minlength=len(murmur_classes)) / PREPROCESSING_CFG['frequency']
    
    if class_duration[1] / sum(class_duration) > 0.8:
        pred = 1
    else:
        if class_duration[0] >= 3:
            pred = 0
        else:
            pred = 2
    return pred
    

@torch.no_grad()
def run_challenge_model(model, data, recordings, verbose):
    (device, preprocessor, murmur_classifier, murmur_classes) = model #outcome_classifier,, outcome_classes
    interval = 1.0
    recording_murmur_counts = np.zeros(len(murmur_classes), dtype=np.int_)
    
    patient_features = torch.from_numpy(load_patient_features(data)).unsqueeze(0).to(device)
    recording_murmur_preds = np.zeros(len(recordings), dtype=np.int_)
    recording_outcome_preds = np.zeros(len(recordings), dtype=np.int_)
    for i in range(len(recordings)):
        multi_scale_specs, qualities = preprocessor(recordings[i], 4000, interval=interval)
        multi_scale_specs = [s.to(device) for s in multi_scale_specs]
        recording_murmur_preds[i] = recording_murmur_diagnose(multi_scale_specs, murmur_classifier, murmur_classes, interval)
        
        # outcome_logits = outcome_classifier(multi_scale_specs, patient_features.repeat(multi_scale_specs[0].shape[0], 1))
        # segment_outcome_preds = torch.max(outcome_logits, dim=1)[1].cpu().numpy()
        # segment_outcome_counts = np.bincount(segment_outcome_preds, minlength=len(outcome_classes))
        # recording_outcome_preds[i] = 0 if (segment_outcome_counts[0] / sum(segment_outcome_counts)) > 0.33 else 1
        
    recording_murmur_counts = np.bincount(recording_murmur_preds, minlength=len(murmur_classes))
    # recording_outcome_counts = np.bincount(recording_outcome_preds, minlength=2)
    
    murmur_labels = np.zeros(len(murmur_classes), dtype=np.int_)
    #如果没有1，则为absent
    if recording_murmur_counts[1] == 0:
        murmur_labels=0
    else:#否则present
    # elif recording_murmur_counts[1] > 0 and recording_murmur_counts[2] < 2:
        murmur_labels= 1
        # murmur_labels[2] = 1
        # murmur_probabilities = recording_murmur_counts / recording_murmur_counts.sum()
    
    # outcome_labels = np.zeros(2, dtype=np.int_)
    # idx = 0 if recording_outcome_counts[0] > 0 else 1
    # outcome_labels[idx] = 1
    # outcome_probabilities = np.zeros(2)
    # outcome_probabilities[idx] = 1.
    
    classes = murmur_classes #+ outcome_classes[:2]
    labels = murmur_labels#np.concatenate((murmur_labels, outcome_labels))
    # probabilities = murmur_probabilities#np.concatenate((murmur_probabilities, outcome_probabilities))

    return labels#, probabilities,classes, 

    
# ################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, model, file_name='murmur_classifier'):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(model_folder, f'{file_name}.pth'))
    

# Load your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    preprocessor = Preprocessor(mode='test', **PREPROCESSING_CFG)
    
    murmur_checkpoint = torch.load(os.path.join(model_folder, 'murmur_classifier.pth'), map_location=device)
    murmur_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_murmur_classes'], **MODEL_CFG).to(device)
    murmur_classifier.load_state_dict(murmur_checkpoint['model_state_dict'])
    murmur_classifier.eval()
    
    # outcome_checkpoint = torch.load(os.path.join(model_folder, 'outcome_classifier.pth'), map_location=device)
    # outcome_classifier = Hierachical_MS_Net(num_classes=DATASET_CFG['num_outcome_classes'], include_patient_data=True, **MODEL_CFG).to(device)
    # outcome_classifier.load_state_dict(outcome_checkpoint['model_state_dict'])
    # outcome_classifier.eval()
    
    murmur_classes = DATASET_CFG['murmur_classes']
    # outcome_classes = DATASET_CFG['outcome_classes']
    
    return (device, preprocessor, murmur_classifier, murmur_classes)#outcome_classifier, , outcome_classes

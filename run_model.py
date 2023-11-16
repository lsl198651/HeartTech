#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for running models for the 2022 Challenge. You can run it as follows:
#
#   python run_model.py model data outputs
#
# where 'model' is a folder containing the your trained model, 'data' is a folder containing the Challenge data, and 'outputs' is a
# folder for saving your model's outputs.

import numpy as np, os, sys
from helper_code import *
from team_code import load_challenge_model, run_challenge_model
from config import *
import csv
import torch
from torcheval.metrics.functional import binary_auprc, binary_auroc, binary_f1_score, binary_confusion_matrix, binary_accuracy, binary_precision, binary_recall

# Run model.
def run_model(model_folder, data_folder, output_folder, allow_failures, verbose):
    # Load models.
    if verbose >= 1:
        print('Loading Challenge model...')

    model_folder = model_folder
    data_folder = data_folder
    output_folder = output_folder
    model = load_challenge_model(model_folder, verbose) ### Teams: Implement this function!!!

    # Find the patient data files.
    patient_files = find_patient_files(data_folder)#找txt文件
    num_patient_files = len(patient_files)#txt数量

    if num_patient_files==0:
        raise Exception('No data was provided.')

    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(output_folder, exist_ok=True)

    # Run the team's model on the Challenge data.
    if verbose >= 1:
        print('Running model on Challenge data...')
    labels_all=[]
    with open('val.csv', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        val_list = [row[0] for row in reader]
    val_patient=[]
    for i in range(num_patient_files):
        patient_data = load_patient_data(patient_files[i])
        id=get_patient_id(patient_data)
        if id in val_list:
            val_patient.append(patient_files[i])
    target_all=[]
    output_all=[]
    # Iterate over the patient files.
    for i in range(len(val_patient)):
        if verbose >= 2:
            print('    {}/{}...'.format(i+1, len(val_patient)))

        patient_data = load_patient_data(val_patient[i])
        recordings = load_recordings(data_folder, patient_data)
        target=get_murmur(patient_data)
        murmur_target=1 if target=="Present" else 0
        target_all.append(murmur_target)
        # Allow or disallow the model to fail on parts of the data; helpful for debugging.
        try:
            labels = run_challenge_model(model, patient_data, recordings, verbose) ### Teams: Implement this function!!!
            output_all.append(labels)
        except:
            if allow_failures:
                if verbose >= 2:
                    print('... failed.')
                classes, labels, probabilities = list(), list(), list()
            else:
                raise
    # 计算指标：
    for i in range(len(target_all)):
        print(target_all[i],output_all[i])

    target_patient,output_patient=torch.tensor(target_all),torch.tensor(output_all)
    
    acc=binary_accuracy(target_patient,output_patient)
    roc=binary_auroc(target_patient,output_patient)
    prc=binary_auprc(target_patient,output_patient)
    f1=binary_f1_score(target_patient,output_patient)
    cm=binary_confusion_matrix(target_patient,output_patient)
    print(f'acc:{acc:.3%}\n roc:{roc:.3f}\n prc:{prc:.3f}\n f1:{f1:.3f}')
    print(cm)


        # # Save Challenge outputs.
        # head, tail = os.path.split(patient_files[i])
        # root, extension = os.path.splitext(tail)
        # output_file = os.path.join(output_folder, root + '.csv')
        # patient_id = get_patient_id(patient_data)
        # save_challenge_outputs(output_file, patient_id, classes, labels, probabilities)

    if verbose >= 1:
        print('Done.')

if __name__ == '__main__':
    # # Parse the arguments.
    # if not (len(sys.argv) == 4 or len(sys.argv) == 5):
    #     raise Exception('Include the model, data, and output folders as arguments, e.g., python run_model.py model data outputs.')

    # # Define the model, data, and output folders.
    # model_folder = sys.argv[1]
    # data_folder = sys.argv[2]
    # output_folder = sys.argv[3]

    # # Allow or disallow the model to fail on parts of the data; helpful for debugging.
    allow_failures = False

    # # Change the level of verbosity; helpful for debugging.
    # if len(sys.argv)==5 and is_integer(sys.argv[4]):
    #     verbose = int(sys.argv[4])
    # else:
    #     verbose = 1
    data_folder=r'D:\Shilong\murmur\Dataset\PCGdataset\training_data'
    model_folder=r'D:\Shilong\murmur\00_Code\LM\HeartTech3\model'
    output_folder=r'D:\Shilong\murmur\00_Code\LM\HeartTech3\output'

    run_model(model_folder, data_folder, output_folder, allow_failures, verbose=2)

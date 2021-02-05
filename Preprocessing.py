import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


def preprocessing(two_dimensional=False):
    """
    pre-processing

    Precondition: fMRI data stored in dataset
    Postcondition: processed data is returned as np arrays

    @returns all data for each participant + one hot encoded labels
    """
    # objective runs to access
    run_nums = [1, 2, 4, 8, 9]
    # the amount of runs we want to loop through
    iter_runs = 5

    # instance variables
    start_people = 1
    end_people = 4
    num_people = end_people - start_people + 1
    THRESH = 100
    iter = 0  # number of run iterations

    # preallocate numpy arrays
    processed_fmri = np.empty([num_people * iter_runs, 209, 42, 80, 80], dtype=np.float32)
    if two_dimensional:
        processed_binary = np.empty([num_people * iter_runs, 209 * 42], dtype=np.uint16)
    else:
        processed_binary = np.empty([num_people * iter_runs, 209], dtype=np.uint16)

    for p in range(start_people, end_people + 1):
        for r in random.sample(run_nums, iter_runs):
            bold_dir = r'sub-0{0}/func/sub-0{0}_task-heatpainwithregulationandratings_run-0{1}_bold.nii.gz'.format(p, r)
            event_dir = r'sub-0{0}/func/sub-0{0}_task-heatpainwithregulationandratings_run-0{1}_events.tsv'.format(p, r)
            if p >= 10:
              bold_dir = r'sub-{0}/func/sub-{0}_task-heatpainwithregulationandratings_run-0{1}_bold.nii.gz'.format(p, r)
              event_dir = r'sub-{0}/func/sub-{0}_task-heatpainwithregulationandratings_run-0{1}_events.tsv'.format(p, r)

            t_bold_path = os.path.join(r'dataset', bold_dir)
            t_event_path = os.path.join(r'dataset', event_dir)

            # images
            img = nib.load(t_bold_path)
            # print(img.header)
            img_data = np.asarray(img.dataobj)
            processed_fmri[iter] = img_data.T
            # print("img_data" in locals())
            # print("img" in locals())
            print('partcipant {0}, iter: {1}'.format(p, iter))
            print('run #:', r)

            # events
            df = pd.read_csv(t_event_path, sep='\t', header=0)
            df = df['ratings'].values

            count_not_nan = 0  # counter variable for non-nan values
            run_events = np.empty(11, dtype=np.uint16)
            for i in df:
                if not np.isnan(i):
                    run_events[count_not_nan] = i
                    # if i >= THRESH:
                    #     run_events[count_not_nan] = 0
                    # else:
                    #     run_events[count_not_nan] = 1
                    count_not_nan += 1
            if two_dimensional:
                run_events = np.repeat(run_events, 798)
            else:
                run_events = np.repeat(run_events, 19)

            processed_binary[iter] = run_events

            # delocalize variables
            img.uncache()
            del img_data, img
            del run_events, df
            del t_bold_path, t_event_path
            del bold_dir, event_dir

            iter += 1  # increment loop variable

    if two_dimensional:
        processed_fmri = processed_fmri.reshape(processed_fmri.shape[0] * processed_fmri.shape[1] * 42, 1, 80, 80)
        processed_binary = processed_binary.flatten()

    print('Processed fmri shape:', processed_fmri.shape)
    print('Processed binary shape:', processed_binary.shape)

    print('processed_fmri', processed_fmri.nbytes / 1_000_000_000, 'gb')
    print('processed_binary', processed_binary.nbytes / 1_000_000_000, 'gb')

    # memory usage is doubled during split
    print('before split')

    X_train, X_test, y_train, y_test = train_test_split(processed_fmri, processed_binary)
    print('X_train:', X_train.nbytes / 1_000_000_000, 'gb')
    print('y_train:', y_train.nbytes / 1_000_000_000, 'gb')
    print('X_test:', X_test.nbytes / 1_000_000_000, 'gb')
    print('y_test:', y_test.nbytes / 1_000_000_000, 'gb')

    print('after split')

    # uncomment code segment to show slices

    # from visual import Visual
    # for one in range(41):
    #     img_data_reshaped = processed_fmri[0, :, :, :]
    #
    #     slice_0 = img_data_reshaped[one, :, :]
    #     slice_1 = img_data_reshaped[:, one, :]
    #     slice_2 = img_data_reshaped[:, :, one]
    #
    #     Visual().show_slices([slice_0, slice_1, slice_2])
    #     plt.show()
    #
    # plt.plot(processed_binary)

    del processed_fmri, processed_binary

    print('X_train shape', X_train.shape)
    print('y_train shape', y_train.shape)
    print('X_test shape', X_test.shape)
    print('y_test shape', y_test.shape)

    return X_train, X_test, y_train, y_test
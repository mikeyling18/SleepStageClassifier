import os
import pyedflib
import joblib
import time

import model as m
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from utils import train, evaluate

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn import metrics
from plots import plot_confusion_matrix

np.set_printoptions(threshold=np.nan)


def load_sleep_dataset(path):
    """Extracts all patients' nightly EEG and annotation data

    Parameters
    ----------
    path : string
        Directory to data/ folder, directory contents should be:
        data/
            interim/
                target_train.json
            raw/
                ...
                sleep-cassette/
                ...

    Returns
    -------
    tuple
        [0] List of patient EEG data
        [1] List of patient annotation data
    """
    sleep_cassette_files = os.listdir(f'{path}/raw/sleep-cassette/')

    eeg_list_all = []
    annotation_list_all = []

    for target in [x for x in range(20)]:
        eeg_list = []
        annotation_list = []
        for night in range(1, 3):

            # Find individual patient
            subject_id_str = str(target)
            if len(subject_id_str) == 1:
                subject_id_str = '0' + subject_id_str
            night_str = str(night)

            # Whole night sleep recordings containing EEG
            try:
                eeg = pyedflib.EdfReader(
                    f'{path}/raw/sleep-cassette/SC4' + subject_id_str + night_str + 'E0-PSG.edf')
            except OSError:
                # One patient only has a single night of data, not two nights
                continue
            eeg_signal = eeg.readSignal(0)
            eeg_list.append(np.array(eeg_signal))

            # Annotations of sleep recordings corresponding to EEG
            annotation_file_path = list(filter(
                lambda x: x.startswith(
                    'SC4' + subject_id_str + night_str) and x.endswith('-Hypnogram.edf'),
                sleep_cassette_files))[0]
            annotation = pyedflib.EdfReader(f'{path}/raw/sleep-cassette/' + annotation_file_path)
            annotation_list.append((
                np.array(annotation.readAnnotations()[0]),
                np.array(annotation.readAnnotations()[1]),
                np.array(annotation.readAnnotations()[2])))

        # Combine EEG and annotation for a single patient
        eeg_list_all.append(eeg_list)
        annotation_list_all.append(annotation_list)

    return (eeg_list_all, annotation_list_all)


def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """Calculate a sliding window over a signal.

    Parameters
    ----------
    data : numpy array
        The array to be slided over.

    size : int
        The sliding window size

    stepsize : int
        The sliding window stepsize. Defaults to 1.

    axis : int
        The axis to slide over. Defaults to the last axis.

    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.

    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occur.

    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided


def process_data_for_1d(eeg_annotation_lists, not_test_set):
    """Prepares data for input into PyTorch for modelling by
    creating training sets of five epochs each (with the center
    epoch being the epoch of interest for prediction). Also
    ensures that label classes are (relatively) balanced on
    training and validation sets. Held out testing sets do not
    have their classes adjusted.

    Parameters
    ----------
    eeg_annotation_lists : tuple of lists
        Contains the input eeg data for modelling along with
        the training labels. Tuple should be formatted as:
        (flattened_eeg_list, flattened_annotation_list)

    not_test_set : boolean
        Switch to denote the final test set. This ensures
        that class balancing is only done on training/validation

    Returns
    -------
    TensorDataset
        Dataset containing inputs and labels to be passed
        into a PyTorch model
    """
    eeg_list = eeg_annotation_lists[0]
    annotation_list = eeg_annotation_lists[1]
    data = np.array([])
    labels = np.array([])

    # Set random state
    prng = np.random.RandomState(42)

    for subject in range(len(eeg_list)):

        eeg = eeg_list[subject]
        annotation = annotation_list[subject]
        inputs = sliding_window(data=eeg, size=15000, stepsize=3000)

        median_index_of_each_input = np.median(
            sliding_window(data=np.arange(eeg.size), size=15000, stepsize=3000),
            axis=1).astype(int)

        values = annotation[2][np.digitize(median_index_of_each_input / 100, annotation[0]) - 1]

        indices_to_delete = np.where(
            (values == 'Sleep stage ?') |
            (values == 'Movement time') |
            (values == '')
        )

        values = np.delete(values, indices_to_delete)
        inputs = np.delete(inputs, indices_to_delete, axis=0)

        if not_test_set:
            # Reduce the amount of Wake labels during training and validation
            # to avoid a large class imbalance while learning. All labels
            # except W use the full observed label count, while the number
            # of W labels is set to the highest label count from (1, 2, 3, R)
            values[values == 'Sleep stage 4'] = 'Sleep stage 3'
            unique, counts = np.unique(values, return_counts=True)
            max_no_w = np.max(counts[:-1])

            stage_1 = prng.choice(np.where(values == 'Sleep stage 1')[0], counts[0], replace=False)
            stage_2 = prng.choice(np.where(values == 'Sleep stage 2')[0], counts[1], replace=False)
            stage_3 = prng.choice(np.where(values == 'Sleep stage 3')[0], counts[2], replace=False)
            stage_r = prng.choice(np.where(values == 'Sleep stage R')[0], counts[3], replace=False)
            stage_w = prng.choice(np.where(values == 'Sleep stage W')[0], max_no_w, replace=False)
            indexes = np.sort(
                np.concatenate((stage_1, stage_2, stage_3, stage_r, stage_w), axis=0))

            values = values[indexes]
            inputs = inputs[indexes]

        values[values == 'Sleep stage 1'] = 0
        values[values == 'Sleep stage 2'] = 1
        values[values == 'Sleep stage 3'] = 2
        values[values == 'Sleep stage 4'] = 2
        values[values == 'Sleep stage R'] = 3
        values[values == 'Sleep stage W'] = 4

        if data.shape[0] == 0:
            data = inputs
        else:
            data = np.concatenate((data, inputs), axis=0)

        if labels.shape[0] == 0:
            labels = values
        else:
            labels = np.concatenate((labels, values), axis=0)

    if not_test_set is not True:
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))

    X = torch.from_numpy(data.astype('float32')).unsqueeze(1)
    Y = torch.from_numpy(labels.astype('long'))

    dataset = TensorDataset(X, Y)

    return dataset


def load_sleep_dataset_targets(targets, eeg_list_all, annotation_list_all):
    """Combines EEG values and annotation labels from multiple
    patients into a single group for use in training, testing,
    validation.

    Parameters
    ----------
    targets : array-like
        Contains integer patient IDs used to define a group

    eeg_list_all : array-like
        Extracted EEG data of all patients from the original
        raw dataset

    annotation_list_all : array-like
        Extracted EEG annotation data of all patients from
        the original raw dataset

    Returns
    -------
    tuple of lists
        [0] Combined EEG values for all targets
        [1] Combined EEG annotation labels for all targets
    """

    eeg_list = [eeg_list_all[index] for index in targets]
    flattened_eeg_list = [val for sublist in eeg_list for val in sublist]

    annotation_list = [annotation_list_all[index] for index in targets]
    flattened_annotation_list = [val for sublist in annotation_list for val in sublist]

    return (flattened_eeg_list, flattened_annotation_list)


def main():
    num_epochs = 3
    batch_size = 50

    # Define working directory locations
    data_dir = '../../data/'
    graph_dir = '../../graphs'

    output_dir = '../output/model'
    os.makedirs(output_dir, exist_ok=True)

    # Set environment seeds
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.manual_seed(42)
    prng = np.random.RandomState(42)

    # Prepare values for model metric reporting
    all_test_loss = []
    all_test_accuracy = []
    all_test_results = []
    all_best_acc = 0.0
    best_file = 'SleepCNNBest.pth'

    # Extract raw EEG and annotation data from full raw dataset
    eeg_list_all, annotation_list_all = load_sleep_dataset(data_dir)

    start = time.time()
    for fold in range(20):
        # Divide subject list for 20-fold cross validation
        subjects = [x for x in range(20)]
        test = [fold]
        subjects.remove(fold)
        training = prng.choice(subjects, 15, replace=False)
        validate = list(set(subjects) - set(training))

        # Format train, validation, and test datasets for PyTorch
        train_dataset = process_data_for_1d(
            load_sleep_dataset_targets(training, eeg_list_all, annotation_list_all),
            not_test_set=True)
        validation_dataset = process_data_for_1d(
            load_sleep_dataset_targets(validate, eeg_list_all, annotation_list_all),
            not_test_set=True)
        test_dataset = process_data_for_1d(
            load_sleep_dataset_targets(test, eeg_list_all, annotation_list_all),
            not_test_set=False)

        # Load dataset splits into PyTorch
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True)

        # Load CNN model and parameters
        model = m.SleepCNN_1D()
        if torch.cuda.is_available():
            print('Unleashing CUDA, zoom zoom!')
            model = model.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        criterion.to(device)
        save_file = 'SleepCNN.pth'

        best_val_acc = 0.0
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []

        for epoch in range(num_epochs):
            # Iteratively train and validate network
            train_loss, train_accuracy = train(
                model, device, train_loader, criterion, optimizer, epoch, fold)

            valid_loss, valid_accuracy, valid_results = evaluate(
                model, device, valid_loader, criterion)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)

            # Save model with best accuracy for final evaluation
            if valid_accuracy > best_val_acc:
                best_val_acc = valid_accuracy
                torch.save(model, os.path.join(output_dir, save_file))
            if valid_accuracy > all_best_acc:
                all_best_acc = valid_accuracy
                torch.save(model, os.path.join(output_dir, best_file))

        # Run final evaluation with best seen performing model
        best_model = torch.load(os.path.join(output_dir, save_file))
        test_loss, test_accuracy, test_results = evaluate(
            best_model, device, test_loader, criterion)

        # Update master list of results for use in cross-validation
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        [all_test_results.append(x) for x in test_results]

    end = time.time()
    print(f'Total fold time took {(end - start) / 60.} mins.')

    # Summarize results across all folds from cross-validation
    joblib.dump(all_test_results, graph_dir + '/all_test_results.gz')
    y_true = [x[0] for x in all_test_results]
    y_pred = [x[1] for x in all_test_results]

    accuracy = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    global_precision = metrics.precision_score(y_true, y_pred, average='micro')

    # Save model summary metrics to file
    with open(graph_dir + '/result_metrics.txt', 'w') as f:
        f.write(f'Class Precision: {precision}')
        f.write(f'\nGlobal precision: {global_precision}')
        f.write(f'\nClass Recall: {recall}')
        f.write(f'\nClass F1-score: {f1_score}')
        f.write(f'\nAccuracy: {accuracy}')

    # Plot final confusion matrices with info from all cv folds
    class_names = ['Non-REM 1', 'Non-REM 2', 'Non-REM 3', 'REM', 'Wake']
    plot_confusion_matrix(all_test_results, class_names, outdir=graph_dir, normalize=False)
    plot_confusion_matrix(all_test_results, class_names, outdir=graph_dir, normalize=True)


if __name__ == '__main__':
    main()

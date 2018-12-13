import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(results, class_names, outdir, normalize=False):
    """Creates and saves a confusion matrix of predicted vs
    observed labels.

    Parameters
    ----------
    results : 2-d array
        Nested list of model results [[y_true, y_pred], ...]

    class_names : array-like
        Label names to use when plotting. The position of a label
        in `class_names` must correspond to the integer value
        from `results`.

    outdir : string
        Directory location to save the resulting image to

    normalize : boolean
        Switch to stretch colorbar across the original, raw
        class counts (False), or to normalize colors and
        class counts before plotting (True)

    Returns : None
    """
    true = [x[0] for x in results]
    predicted = [x[1] for x in results]
    cm = confusion_matrix(true, predicted)
    cmap = plt.cm.Blues
    np.set_printoptions(precision=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fname = 'confusion_matrix_normalized'
        title = 'Confusion Matrix (Normalized counts)'
        print('Normalized confusion matrix')
    else:
        fname = 'confusion_matrix_raw_counts'
        title = 'Confusion Matrix (Raw counts)'
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.savefig(f'{outdir}/{fname}.png', bbox_inches='tight')
    plt.close()

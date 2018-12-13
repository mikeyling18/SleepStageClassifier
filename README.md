# cse6250-sleep-study
Sleep stage classification using EEG data with a convolutional neural network

## Project Organization
```
    ├── data
    │   └── raw          : Original data from physionet.org
    │
    ├── graphs           : Summary plots and results saved during model training
    │
    ├── README.md
    │
    ├── report
    │   └── final        : Final project report and presentation slides
    │
    ├── requirements.yml : Python environment description for use with Conda
    │
    ├── src
    │   ├── code         : All ETL, model building, and analysis code
    │   ├── notebook     : Exploratory analysis and data visualization
    │   └── output       : Exploratory analysis and data visualization
    │       └── model    : Saved object of the trained CNN model
```

## Raw Data Download
For reproducibility, the raw data used for this project can be downloaded from physionet.org via
```
$ rsync -Cavz physionet.org::sleep-edfx data/raw
```
and a link for this project submission has also been made [available on Google Drive](https://drive.google.com/open?id=1O5FRj0We-E2PiIWGKIzRcwI7o0iG3xK-).

## Regenerating Trained CNN
To replicate the results of this project, first be sure to properly mirror the project's environment using Conda.
```
$ conda create --name sleep-study --file requirements.yml
$ source activate sleep-study
```
Once the raw data has been downloaded from Physionet as described above, train and evaluate the model.
```
(sleep-study)$ cd src/code/
(sleep-study)$ python3 train_sleep.py
```
The resulting plots and summary metrics will be found in `graphs/` and the trained model objects can be found in `src/output/model/`. Load the trained PyTorch model object back into memory for later use with
```
torch.load('src/output/model/SleepCNNBest.pth')
```
## Final Reports/Presentations
Links to the final video presentation and slide deck can be found in the final report, saved in `report/final/`.

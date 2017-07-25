import os

logs = './logs/'
models = './models/'
submissions = './submissions/'
data = './data/'
validations = './validations/'
predictions = './predictions/'
thresholds = './thresholds/'
ensemble_weights = './ensemble_weights/'
xgb_configurations = './xgb_configurations/'

train_jpg = '../Planet/train-jpg/'
train_tif = '../Planet/train-tif-v2/'
test_jpg = '../Planet/test-jpg/'
train_csv = '../Planet/train_v2.csv'

dirs = [
    logs,
    models,
    submissions,
    data,
    validations,
    predictions,
    thresholds,
    ensemble_weights,
    xgb_configurations
]

data = [train_jpg, train_tif, test_jpg]

files = [train_csv]

for supplementary_dir in dirs:
    if os.path.isdir(supplementary_dir):
        continue

    if not os.path.isfile(supplementary_dir[:-1]):
        os.makedirs(supplementary_dir)
        print('Created directory', supplementary_dir)

for data_dir in data:
    if os.path.isdir(data_dir):
        continue

    else:
        print('Directoy {} does not exists. Please either put the training/test data in the appropriate directories or '
              'change the path.'.format(data_dir))


for file in files:
    if os.path.isfile(file):
        continue

    else:
        print('File {} does not exists. Please either put the file in the appropriate directories or '
              'change the path.'.format(file))

# KagglePlanetPytorch
This repository contains the basic code of our 9th place submission.

For questions refere to https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36887 or create an issue!

## Requirements
Basic requirements are
  *Scitkit-Learn
  *Scikit-Image
  *Torchsample
  *Pytorch
  *XGBoost

This list may not be exhaustive!

## Training a network
Just run the nn_finetune-files.

## Create predictions for a network
Choose the network in predict.py and run it. Predictions are then saved to /predictions.

## Calculate thresholds
This step is only necessary because of the current implementation. Run save_thresholds.py for your model. The saved thresholds we be used in the next step to compare XGBoost to averaging.

## Make a submission from a single 5-fold model
Specify the network in model_tta_hyperopt.py and run it. This will run hyper parameter optmization for XGBoost. The approach chosen in this file is probably not good at all, since this was the first time I used XGBoost and only had a week to the competition deadline. Please tell me if you can do better. Also if you can make the same basic approach work for model ensembling, tell me! :)
Submission are saved to /submissions

## Make a weighted submissions from different submission files
Just specify your submissions and weights in submit_ensemble.py

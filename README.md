# Convolutional neural networks for scops owl sound classification

1. Download the dataset (mono wav files) from the following link: https://zenodo.org/record/7387375#.Y4jus3ZBy5c
2. Unzip and put the dataset in the 'data' directory
3. Set the desired configuration for the training phase via `config.py`
4. Start training the model: `python train.py`
5. Use `inference.ipynb` to perform the inference stage and set the necessary configuration in `config.py`

If you find the above dataset and/or this code useful for you research I will appreciate if you could cite the following paper: https://doi.org/10.1016/j.procs.2020.12.010 as well as John Martinsson's github (https://github.com/johnmartinsson/bird-species-classification), which is the repository that I largely adopted for my work.

Please let me know if you have any issue or question regarding this work.

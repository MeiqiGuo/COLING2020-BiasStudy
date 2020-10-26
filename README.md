# Inflating Topic Relevance with Ideology: A Case Study of Political Ideology Bias in Social Topic Detection Models

This is a PyTorch implementation of the experiments described in our submitted paper for COLING 2020.

## Environment Setup

>pip install -r requirements.txt


## Training off-the-shell NLP models and prediction inference

1) Pre-process data;

2) Change configuration files in the folder "myallennlp/training_config" by adding the paths to training and dev datasets;

>"train_data_path": TRAIN_FILE,

>"validation_data_path": DEV_FILE,

3) Run training command:

>mkdir save

>allennlp train myallennlp/training_config/topical_extractor_\<model\>.jsonnet --serialization-dir save/\<model\> --include-package myallennlp
  
4) Inference command:

>python myallennlp/prediction.py <path_to_saved_model> --pred_file <path_to_test_file>
  
## Training and testing our proposed adversarial approach

For ELMo+ADV and GloVe+ADV:
>python mymodel/main.py --base_model \<model\> --train_data_path <path_to_train_file> --val_data_path <path_to_dev_file> --test_data_path <path_to_test_file> --save_root <path_to_save_root> --train --test --lr 4e-4 --batch_size 64 --n_epoch 75

For BERT+ADV:
>python mymodel/main.py --base_model \<model\> --train_data_path <path_to_train_file> --val_data_path <path_to_dev_file> --test_data_path <path_to_test_file> --save_root <path_to_save_root> --train --test --lr 3e-5 --batch_size 32 --n_epoch 5

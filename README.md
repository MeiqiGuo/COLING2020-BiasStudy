# Inflating Topic Relevance with Ideology: A Case Study of Political Ideology Bias in Social Topic Detection Models

This is a PyTorch implementation of the experiments described in our paper accepted in COLING 2020.

## Environment Setup

>pip install -r requirements.txt


## Training off-the-shell NLP models and prediction inference

1) Pre-process data;

We used Tweets collected by [Yang, et al.](https://github.com/picsolab/TRIBAL-release) as our raw data corpus. We pre-processed the tweets by 1) removing all emoji; 2) removing all website links; 3) removing @\[USER\]. You could do any pre-processing here for your own purpose. Then we created training and dev sets by using seed keywords. Please refer to our paper for more information. The final format of the train and dev data should be a csv file with one tweet instance per line. Each line has three elements which are respectively the political group label (Blue/Red), the tweet text and the label for the topic relevance (0/1).

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

## Citation

If you make use of this code, please kindly cite our paper:

Meiqi Guo, Rebecca Hwa, Yu-Ru Lin and Wen-Ting Chung. 2020. Inflating Topic Relevance with Ideology: A Case Study of Political Ideology Bias in Social Topic Detection Models. In The Proceedings of The 28th International Conference on Computational Linguistics (COLING-2020).

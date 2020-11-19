# SpeechCommands
TensorFlow Speech Commands Recognition

## TensorFlow Speech Commands Dataset

https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data



### training the model

## Spectrograms(STFT) to two-dimensional CNN
python train_2D_DCNN_STFT.py -sample_rate 16000 -batch_size 64 -output_representation spec -data_dirs Data/train/train/audio

## Spectrograms(STFT) to two-dimensional CNN
python train_2D_DCNN_MFCC.py -sample_rate 16000 -batch_size 64 -output_representation mfcc -data_dirs Data/train/train/audio

## Raw signal to one-dimensional Dilated CNN
python train_1D_DCNN_Raw.py -sample_rate 16000 -batch_size 64 -output_representation raw -data_dirs Data/train/train/audio

## Spectrograms(STFT) to one-dimensional Dilated CNN
python train_1D_DCNN_MFCC.py -sample_rate 16000 -batch_size 64 -output_representation mfcc -data_dirs Data/train/train/audio

## MFCC to one-dimensional Dilated CNN
python train_1D_DCNN_STFT.py -sample_rate 16000 -batch_size 64 -output_representation spec -data_dirs Data/train/train/audio





## Best Result
After the training the model for 30 epochs, the following confusion matrix was generated for the best performing model which is MFCC to one-dimensional Dilated CNN


[029]: val_categorical_accuracy: 0.94
Predicted  _silence_  _unknown_  down   go  left   no  off   on  right  stop   up  yes
Actual                                                                                
_silence_        327          0     0    0     0    0    0    0      0     0    0    0
_unknown_          3       1458    14    5     4    9    5   11     10     2    9    2
down               0          5   242    2     0    9    0    1      1     0    0    0
go                 3         17     7  221     0    4    1    0      2     1    1    0
left               0          7     0    0   233    3    0    0      1     0    1    2
no                 0         13     5    3     0  246    0    0      0     0    2    0
off                0          3     0    2     0    0  232    2      0     0   15    0
on                 0         13     0    0     0    0    4  238      0     0    1    0
right              0         12     0    0     1    0    0    0    238     0    2    0
stop               1         10     3    1     0    1    4    1      0   217    5    0
up                 0         12     0    1     0    0    5    2      0     1  238    0
yes                0          7     2    1     6    3    0    0      0     0    0  240

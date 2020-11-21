# Pytorch Implementation of [EleAtt-RNN: Adding Attentiveness to Neurons in Recurrent Neural Networks]


## Visualization of the Learned Views

![image](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition/blob/master/image/visulization.png)

Figure 3: Frames of (a) the similar posture captured from different viewpoints for the same subject, and (b) the same action “drinking” captured from different viewpoints for different subjects. 2nd row: original skeletons. 3rd row: Skeleton representations from the observation viewpoints of our VA-RNN model. 4th row: Skeleton representations from the observation viewpoints of our VA-CNN model.


## Data Preparation

We use the NTU60 RGB+D dataset as an example for description. We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset

- Process the data
```bash
 cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


## Training

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 1

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 1

# For RNN-based model with view adaptation module
python va-rnn.py --model VA --aug 1 --train 1

# For RNN-based model without view adaptation module
python va-rnn.py --model baseline --aug 1 --train 1
```



## Testing

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 0

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 0

# For RNN-based model with view adaptation module
python va-rnn.py --model VA --aug 1 --train 0

# For RNN-based model without view adaptation module
python va-rnn.py --model baseline --aug 1 --train 0
```

## Reference


```

https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition

@article{zhang2019view,
  title={View adaptive neural networks for high performance skeleton-based human action recognition},
  author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2019},
}

@inproceedings{zhang2017view,
  title={View adaptive recurrent neural networks for high performance human action recognition from skeleton data},
  author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2117--2126},
  year={2017}
}

```


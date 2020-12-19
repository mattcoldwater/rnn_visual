# Pytorch Implementation of [EleAtt-RNN: Adding Attentiveness to Neurons in Recurrent Neural Networks]

## Pre-trained Model

[Pre-trained Model](https://drive.google.com/drive/folders/1-859NwPYWt2UQoNpCdxwo5gn2QLsj5YV?usp=sharing)

## Data Preparation

We use the NTU120 RGB+D dataset as an example for description. We need to first dowload the [NTU120-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset

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


## Training and Testing

```bash
# 3 att
python gru.py --aug 0 --experiment att3_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 3
python gru.py --aug 0 --experiment att3_gru0 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 3

# 0 att
python gru.py --aug 0 --experiment att0_gru3 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100
python gru.py --aug 0 --experiment att0_gru3 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1

# 1 att
python gru.py --aug 0 --experiment att1_gru2 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 1
python gru.py --aug 0 --experiment att1_gru2 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 1

# 2 att
python gru.py --aug 0 --experiment att2_gru1 --print_freq 500 --batch_size 256 --lr 0.005 --train 1 --max_epoches 100 --att 2
python gru.py --aug 0 --experiment att2_gru1 --print_freq 500 --batch_size 256 --lr 0.005 --train 0 --max_epoches 1 --att 2
```

## Visualizing skeleton

```bash
#With generated attention response
#create skeleton instance
sk = Draw3DSkeleton(file=data_from_net, save_path=path_test_result, is_file_txt=False)
#set generated relative attention response
sk.set_relative_response(attention_response=attention_response, arrange_required=True)
#visualize the skeleton with attention in a 3D plot
sk.visual_skeleton_animate(use_relative_response=True, scattersize_max=300, sleep_time=0.1, is_image_save=True, skeleton_color='r', joint_color='blue')

#With constant(aritificial) attention
#create skeleton instance
sk_plain = Draw3DSkeleton(file=attention_response, save_path=path_rest_result_plain, is_file_txt=False) 
#set constant attention response
sk_plain.set_relative_response(attention_response=None, arrange_required=False)
#visualize the skeleton with attention in a 3D plot
sk_plain.visual_skeleton_animate(use_relative_response=True, scattersize_max=50, sleep_time=0.5, is_image_save=True, skeleton_color='r', joint_color='blue')
```

## Reference


```
1. P. Zhang, J. Xue, C. Lan, W. Zeng, Z. Gao, and N. Zheng, “EleAtt-RNN: Adding Attentiveness to Neurons in Recurrent Neural Networks,” IEEE Transactions on Image Processing, vol. 29, pp. 1061–1073, 2020. 
2. J. Liu, A. Shahroudy, M. Perez, G. Wang, L.-Y. Duan, and A. C. Kot, “NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 10, pp. 2684–2701, 2020. 
3. A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang, “NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis,” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 
4. P. Zhang, C. Lan, J. Xing, W. Zeng, J. Xue, and N. Zheng, “microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition,” GitHub. [Online]. Available: https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition. [Accessed: 18-Dec-2020]. 
5. A. Paszke et al., "Pytorch: An imperative style high-performance deep learning library", Proc. Adv. Neural Inf. Process. Syst., pp. 8026-8037, 2019.
6. Enriccorona, “enriccorona/human-motion-prediction-pytorch,” GitHub. [Online]. Available: https://github.com/enriccorona/human-motion-prediction-pytorch. [Accessed: 18-Dec-2020]. 
7. XiaoCode-er, “XiaoCode-er/3D-Skeleton-Display,” GitHub. [Online]. Available: https://github.com/XiaoCode-er/3D-Skeleton-Display. [Accessed: 18-Dec-2020]. 
```

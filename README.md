# Pytorch Implementation of [EleAtt-RNN: Adding Attentiveness to Neurons in Recurrent Neural Networks]


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


## Reference


```
1. Pengfei Zhang et al. Adding Attentiveness to Neurons in Recurrent Neural Networks. In IEEE, 2020.
2. Jun Liu, et al. NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding. In IEEE, 2019.
3. A. Shahroudy, J. Liu, T.-T. Ng, and G. Wang. Ntu rgb+d: A large scale dataset for 3d human activity analysis. In CVPR, 2016.
4. Microsoft. (n.d.). Microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition. Retrieved December 12, 2020, from https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition
5. PyTorch documentation¶. (n.d.). Retrieved December 12, 2020, from https://pytorch.org/docs/stable/index.html
```
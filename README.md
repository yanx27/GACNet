# Graph Attention Convolution for Point Cloud Segmentation

This is pytorch implmentation of GACNet, but **not official version**.
![](pic.png)

## Download Data
Run `download_data.sh` and save dataset in `./indoor3d_sem_seg_hdf5_data/`

## Train Model
Run `python train_semseg.py --batchSize 16`

## Announcement
It is only a personal realization, and the experimental results do not represent the model in paper. There are still many hyper parameters that need to be adjusted when the author publishes the source code

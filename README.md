# Learning to Localize in New Environments from Synthetic Training Data

Dominik Winkelbauer, Maximilian Denninger, Rudolph Triebel
ICRA 2021
[PDF](https://arxiv.org/abs/2011.04539)

## BibTeX

```
@inproceedings{WinkelbauerICRA21,
    author={Winkelbauer, Dominik and Denninger, Maximilian and Triebel, Rudolph},
    title={Learning to Localize in New Environments from Synthetic Training Data},
    booktitle={ICRA},
    year={2021}
}
```

## Environment

The code was tested under the following conditions:

```
python 3.6.9

tensorflow~=2.2.0
numpy~=1.18.5
h5py~=2.10.0
opencv-python~=4.4.0.42
tqdm~=4.32.1
imageio~=2.3.0
scipy~=1.4.1
matplotlib~=3.1.3
```

You might setup a conda environment via the following commands:
```
conda create --name ExReNet python=3.6.9
conda activate ExReNet
conda install -c anaconda tensorflow-gpu==2.2.0 numpy==1.18.5 h5py==2.10.0 --yes
conda install -c conda-forge tqdm==4.32 imageio==2.3.0 scipy==1.4.1 keras-applications --yes
conda install matplotlib --yes
python -m pip install image-classifiers==1.0.0b1
python -m pip install opencv-python
```

## Training data

This section describes how to prepare the data from different datasets to be able to use them for training.

### ScanNet

<details><summary>Expand</summary>
<p>

#### Download

Download ScanNet v2 dataset, only `.sens` and `.txt` files are required.

#### Extract images from .sens files

This will extract the single frames from the `.sens` files and store them as images (we hereby only use every 10th frame).

```
python3 external/scannet/batch.py <scannet>/scans
python3 external/scannet/batch.py <scannet>/scans_test
```

#### Resize and project images

Now the images are resized to the resolution `128x128` and a border is added to simulate a focal length `530`.
Depth images are resized in the same way and are stored as `.raw` binary files.

```
python3 data/ScanNet_resize.py <scannet>/scans 128 530
python3 data/ScanNet_resize.py <scannet>/scans_test 128 530
```

#### Calculate intersection measures

In this step we calculate for all combinations of frames from a scene how much they intersect.

```
python3 data/ScanNet_build_ious.py <scannet>/scans
python3 data/ScanNet_build_ious.py <scannet>/scans_test
```

#### Build pairs

Based on these intersections, now pairs are formed and listed in the specified text files.

```
python3 data/ScanNet_build_pairs.py <scannet>/scans <scannet>/pairs_d0.6_a30.txt
python3 data/ScanNet_build_pairs.py <scannet>/scans_test <scannet>/pairs_test_d0.6_a30.txt
```

#### Shuffle pairs

In the last step the pairs are now shuffled

```
python3 data/shuffle_pairs.py <scannet>/pairs_d0.6_a30.txt
python3 data/shuffle_pairs.py <scannet>/pairs_test_d0.6_a30.txt
```

#### Set paths in config file

To make use of the data during training, set the following lines in `config/default.json`:
```
    "train_pair_file": "<scannet>/pairs_d0.6_a30_shuffle.txt",
    "val_pair_file": "<scannet>/pairs_test_d0.6_a30_shuffle.txt",
    "train_data_path": "<scannet>/scans_128",
    "val_data_path": "<scannet>/scans_test_128"
```

</p>
</details>

### Synthetic data

<details><summary>Expand</summary>
<p>
In this section the procedure is described how to generate training data via BlenderProc and SUNCG.

#### Generate data

Download BlenderProc from https://github.com/DLR-RM/BlenderProc

Copy custom modules
```
cp data/PairwiseSuncgCameraSampler.py <BlenderProc>/src/camera/
cp data/Mixed3d.py <BlenderProc>/src/provider/sampler/
```

In BlenderProc root:

```
python run.py <ExReNet>/data/blenderproc_config_dense_pairs.yaml <house.json> <output_dir>/<house_id>
```

Here `<house.json>` should point to one `house.json` file in the SUNCG dataset and `<output_dir>/<house_id>` sets the output path.
Do this step multiple time for different houses until enough training data has been collected.
Then split the output directories into `<synth_dir>/suncg` and `<synth_dir>/suncg_test`

#### Calculate intersection measures

In this step we calculate for all combinations of frames from a scene how much they intersect.

```
python3 data/Suncg_build_ious.py <synth_dir>/suncg
python3 data/Suncg_build_ious.py <synth_dir>/suncg_test
```

#### Collect pairs

Based on these intersections, now pairs are formed and listed in the specified text files.

```
python3 data/Suncg_build_pairs.py <synth_dir>/suncg <synth_dir>/train_pairs.txt
python3 data/Suncg_build_pairs.py <synth_dir>/suncg_test <synth_dir>/test_pairs.txt
```

#### Shuffle pairs

In the last step the pairs are now shuffled

```
python3 data/shuffle_pairs.py <synth_dir>/train_pairs.txt
python3 data/shuffle_pairs.py <synth_dir>/test_pairs.txt
```

#### Set paths in config file

To make use of the data during training, set the following lines in `config/default.json`:
```
    "train_pair_file": "<synth_dir>/train_pairs.txt",
    "val_pair_file": "<synth_dir>/test_pairs.txt",
    "train_data_path": "<synth_dir>/suncg",
    "val_data_path": "<synth_dir>/suncg_test"
```

</rel_path>
</details>

### Custom data

<details><summary>Expand</summary>
<p>

The training data should have the following data structure:
```
data/
data/images
data/images/1790.jpg
data/images/1790.raw
data/images/...
data/pairs.txt
data/pairs_val.txt
```

- `data/images/1790.jpg` one image, jpg, should be already in 128x128
- `data/images/1790.raw` a binary file (readable by `tf.io.read_file`) containing the depth image in float32 with shape [128, 128].
- `data/pairs.txt` lists all images pairs that should be used for training, one row describes one pair / training sample
- `data/pairs_val.txt` lists all images pairs that should be used for validation, one row describes one pair / val sample


Structure of a row in the pairs text files:

```
<rel_path img1> <rel_path img2> <row-wise 4x4 t-mat pose img1> <row-wise 4x4 t-mat pose img2> <fx> <fy> <cx> <cy>
```

The pairs should already be shuffled!

Example:

```
1790.jpg 1740.jpg -0.998281 0.002259 -0.058561 5.117176 0.048069 0.603176 -0.796158 5.589637 0.033524 -0.797605 -0.602248 1.204097 0.0 0.0 0.0 1.0 -0.860071 0.235089 -0.452781 4.913857 0.510174 0.395488 -0.763749 5.435265 -0.00048 -0.887875 -0.460084 1.257236 0.0 0.0 0.0 1.0 577.870605 577.870605 319.5 239.5
```

The poses are represented as a transformation matrix mapping points from the camera frame to the world frame.
The camera frame is defined as Y up, -Z forward, X right.

In the `.json` training config set the paths to your custom training data:
```
    "train_pair_file": "data/pairs.txt",
    "val_pair_file": "data/pairs_val.txt",
    "train_data_path": "data/images",
    "val_data_path": "data/images"
```

The images for train and val can also be stored in different directories.

</rel_path>
</details>

## Run training

The following line will now run the training based on the default parameters in `config/default.json` for 5000 iterations.
The tensorboard files and model parameters will be stored at `runs/default`.

```
python3 train.py config/default.json runs/default 5000
```

### Train with dropout

To get an uncertainty estimate, use to following command to train a model with dropout.

```
python3 train.py config/uncertainty.json runs/uncertainty 5000
```

## Evaluation data

After training the model can now be evaluated.

### Prepare 7-Scenes

<details><summary>Expand</summary>
<p>

#### Download 7-scenes

Download and unpack the 7-Scenes data set from https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/.

#### Convert to hdf5

The following script now prepares RGB and poses of each frame and stores them into .hdf5 files.
`<7scenes>` should point to your 7-Scenes copy.

```
python3 data/7scenes_to_hdf5.py <7scenes>
```

</p>
</details>

## Run Evaluation

### Inference 

Evaluate the network stored in `runs/default` on 7-Scenes. 

```
python3 batch_inference.py config/default.json runs/default <7scenes> 
```

#### Inference with scale

Evaluate the network stored in `runs/default` on 7-Scenes using scale information.

```
python3 batch_inference.py config/default.json runs/default <7scenes> --scale
```

#### Inference with uncertainty

Evaluate the network stored in `runs/uncertainty` on 7-Scenes using scale and uncertainty information.

```
python3 batch_inference.py config/uncertainty.json runs/uncertainty <7scenes> --scale --uncertainty
```

## Pretrained models

You can download some pretrained models via the [GitHub release page](https://github.com/DLR-RM/ExReNet/releases).
After downloading, unpack the zip into the `runs/` directory and then run the inference command as specified in the following subsections.
Make sure you have prepared your 7-Scenes dataset like described in [Prepare 7-Scenes](https://github.com/DLR-RM/ExReNet#prepare-7-scenes).

### scannet_default.zip

This model was trained on ScanNet with the `default.json` config.
For evaluation on 7-Scene run:

```
python3 batch_inference.py config/default.json runs/scannet_default/ <7scenes> --legacy --scale
```

The `--legacy` flag is necessary here, as the network was trained with a slightly older code that applied an additional [90° rotation](https://github.com/DLR-RM/ExReNet/blob/main/src/data/Data.py#L107) to the relative poses.

Should output:

```
chess/test: 0.060m 2.15°
fire/test: 0.092m 3.20°
heads/test: 0.041m 3.30°
office/test: 0.071m 2.17°
pumpkin/test: 0.109m 2.65°
redkitchen/test: 0.085m 2.57°
stairs/test: 0.329m 7.34°
Mean: 0.112m 3.34°
```

### suncg_default.zip

This model was trained on synthetic data generated with BlenderProc and SUNCG scenes. 
For training, the `default.json` config was used.
For evaluation on 7-Scenes run:

```
python3 batch_inference.py config/default.json runs/suncg_default/ <7scenes> --scale
```

Should output:

```
chess/test: 0.048m 1.63°
fire/test: 0.074m 2.54°
heads/test: 0.033m 2.71°
office/test: 0.059m 1.75°
pumpkin/test: 0.074m 2.04°
redkitchen/test: 0.068m 2.10°
stairs/test: 0.194m 4.87°
Mean: 0.079m 2.52°
```

The results here are even a bit better than the ones listed in the paper.
This is caused by the fact that more accurate matching labels for all image pairs were used for training.

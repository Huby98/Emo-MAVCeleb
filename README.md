# EMO-MAVceleb

<img width="1292" height="516" alt="Picture-ID0005-New" src="https://github.com/user-attachments/assets/c1a661ae-2382-444f-80d9-088a226b32a3" />


## Cite

If you use this work make sure to cite the baseline for this 

[Albanie et al., 2018](https://github.com/albanie/mcnCrossModalEmotions)

```bibtex
@inproceedings{albanie2018emotion,
  title={Emotion recognition in speech using cross-modal transfer in the wild},
  author={Albanie, Samuel and Nagrani, Arsha and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={292--301},
  year={2018}
}
```

## Setup

To make use of the teacher network originally used, I used the [MatConvNet inside Matlab](https://github.com/vlfeat/matconvnet)

This requires specific installation guidelines to be useable in 2025.

I downloaded Visual Studio 2017 in combination with Matlab R2019b (it might work with other combinations as well) 
To use Matlab correct I started it like this

<img width="335" height="102" alt="image" src="https://github.com/user-attachments/assets/716f5ffa-2f4d-4860-8030-7524c6002dca" />

Then entering ""C:\Program Files\MATLAB\R2019b\bin\matlab.exe" (or wherever the Matlab version is installed)

I always used this after running the setup from the [MatConvNet website](https://www.vlfeat.org/matconvnet/)

```powershell
p = 'D:\albanie\matconvnet-1.0-beta25'; <-- here the link to my matconvnet installaton
cd(p); addpath(fullfile(p,'matlab')); vl_setupnn;
addpath(genpath(fullfile(p,'contrib','autonn')));
addpath(genpath(fullfile(p,'contrib','mcnExtraLayers')));
addpath(genpath(fullfile(p,'contrib','mcnDatasets')));
addpath(genpath(fullfile(p,'contrib','mcnCrossModalEmotions')));
```

## Teacher (Face Side)

After using the setup, to use the teacher script were created inside Matlab (labelthemall)
For this to work first the splitting of the dataset had to be done (Train/Val/Test) 










# TomoEdge

This repo including an implementation of TomoGAN generator(inference only) using cuDNN, it also has a cuda implmentation of gridrec reconstuction by (Viktor Nikitin)

One may use it for education purpose of cuDNN. 

We used the code for raid computed tomography reconstruction at edge using Jetson TX2. slides page 9 of https://lzhengchun.github.io/file/CELS-coffee-talk-20200415.pdf
Basically, we used gridrec to get a reasonably ok reconstruction, then use a trained TomoGAN to remove noise and artifacts. 
So, the overall motivation was to have tomography reconstruction near beamline to guide experiment. 

Please cite our works, as follows, if you used this repo for your research 

```
@inproceedings{liu2019deep,
    title={Deep Learning Accelerated Light Source Experiments},
    author={Zhengchun Liu and Tekin Bicer and Rajkumar Kettimuthu and Ian Foster},
    year={2019},
    booktitle={2019 IEEE/ACM Third Workshop on Deep Learning on Supercomputers (DLS)},
    pages={20--28},
    doi={10.1109/DLS49591.2019.00008}
}

@article{liu2020tomogan,
  title={TomoGAN: low-dose synchrotron x-ray tomography with generative adversarial networks: discussion},
  author={Liu, Zhengchun and Bicer, Tekin and Kettimuthu, Rajkumar and Gursoy, Doga and De Carlo, Francesco and Foster, Ian},
  journal={Journal of the Optical Society of America A},
  volume={37},
  number={3},
  pages={422--434},
  year={2020},
  doi={10.1364/JOSAA.375595},
  publisher={Optical Society of America}
}

```

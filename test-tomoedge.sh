#! /bin/bash

rm tomoedge.out
nvcc tomoEdge.cpp tomoGAN.cu gridrec.cu -lcufft -lcudnn -o tomoedge.out -std=c++11 #-Xcompiler -fopenmp
./tomoedge.out
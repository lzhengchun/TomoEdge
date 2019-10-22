#! /bin/bash

rm fbp.out
nvcc test_fbp.cpp fbp.cu -lcufft -o fbp.out
./fbp.out
#! /bin/bash

rm gridrec.out
nvcc test_gridrec.cpp gridrec.cu -lcufft -o gridrec.out
./gridrec.out
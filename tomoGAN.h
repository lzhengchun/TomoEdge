#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <iostream>

using namespace std;

typedef uint32_t uint32;
typedef uint8_t  uint8;

//function to print out error message from cuDNN calls
#define cudnnErrchk(exp){ \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

#define cudaErrchk(ans)  cudaAssert((ans), __FILE__, __LINE__) 
inline void cudaAssert(cudaError_t code, string file, int line){
    if (code != cudaSuccess){
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << "; file: " << file << ", line:" << line << std::endl;
        exit(-1);
    }
}

class tomoGAN
{
    uint32 n_img_in, c_img_in, h_img_in, w_img_in;
    float *input_buf, *output_buf;     // input and output buf on device
    float *layer_buf1, *layer_buf2;    // reusable buffer for layer output and input
    float *box1_out_buf, *box2_out_buf, *box3_out_buf; // box output for later concatenation
    float *weights_d[19];

    // network description
    //                                0  1   2   3   4   5    6    7    8    9   10   11  12   13  14  15  16  17  18
    const unsigned int conv_ch[19] = {3, 8,  32, 32, 64, 64,  128, 128, 128, 256, 64, 64, 128, 32, 32, 64, 32, 32, 16};
    const unsigned int  n_conv[19] = {8, 32, 32, 64, 64, 128, 128, 128, 128, 64,  64, 64, 32,  32, 32, 32, 32, 16, 1};
    const unsigned int conv_sz[19] = {1, 3,  3,  3,   3, 3,   3,   3,   2,   3,   3,  2,  3,   3,  2,  3,  3,  1,  1};

public:
    tomoGAN(uint32 img_n, uint32 img_c, uint32 img_h, uint32 img_w, float *weights_h);
    ~tomoGAN();
    void inference(uint8 *img_in, uint8 *img_out);
};

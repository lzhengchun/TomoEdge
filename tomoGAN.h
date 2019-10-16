#include <cuda.h>
#include <cudnn.h>
#include <string>
#include <iostream>
#include <chrono>

#define DEBUG_PRINT true
using namespace std;

typedef uint32_t uint32;
typedef uint8_t  uint8;

//function to print out error message from cuDNN calls
#define cudnnErrchk(exp){ \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " \
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

    cudnnHandle_t cudnn_handle;
    cudnnTensorDescriptor_t bias_desc;
    cudnnActivationDescriptor_t relu_activ_desc;

    //network description             0  1   2   3   4   5    6    7    8    9   10   11  12  13  14  15
    const uint32 conv_ch[16] = {1, 8,  32, 32, 64, 64,  128, 128, 256, 64, 128, 32, 64, 32, 32, 16};
    const uint32  n_conv[16] = {8, 32, 32, 64, 64, 128, 128, 128, 64,  64, 32,  32, 32, 32, 16, 1};
    const uint32 conv_sz[16] = {1, 3,  3,  3,   3, 3,   3,   3,   3,   3,  3,   3,   3, 3,  1,  1};

public:
    tomoGAN(uint32 img_n, uint32 img_c, uint32 img_h, uint32 img_w, float *weights_h);
    ~tomoGAN();
    void predict(float *img_in, float *img_out);
    
private:
    void prinf_tensor_sz(cudnnTensorDescriptor_t tensor);

    void printf_layer_io(cudnnTensorDescriptor_t tensor_in, cudnnTensorDescriptor_t tensor_out);

    void maxpooling(float *mem_in, float *mem_out,
                    cudnnTensorDescriptor_t &tensor_in_desc, 
                    cudnnTensorDescriptor_t &tensor_out_desc);

    void conv2d(uint32 in_ch, uint8 knl_sz, uint32 n_knl, 
                float *mem_in, float *mem_out, float *weights,
                cudnnTensorDescriptor_t &tensor_in_desc, 
                cudnnTensorDescriptor_t &tensor_out_desc, 
                bool relu);

    void upsampling(float *mem_in, float *mem_out,
                    cudnnTensorDescriptor_t &tensor_in_desc, 
                    cudnnTensorDescriptor_t &tensor_out_desc);
};

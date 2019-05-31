#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


__global__ void GEMMLowpKernel(const float* input, const int N, float* out,
                               float scale, float zero_point) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      out[i] = input[i];
    //   if (enforce_true_zero)
      out[i] = (out[i]*scale) - zero_point;
    //   else
    //     out[i] = (out[i] + shift) / scale;
      out[i] = roundf(out[i]);
      out[i] = fminf(out[i], clamp_max);
      out[i] = fmaxf(out[i], clamp_min);
    //   if (enforce_true_zero)
      out[i] = (out[i] + zero_point) / scale;
    //   else
    //     out[i] = out[i] * scale - shift;
  }
}

#define block_count 32
#define thread_per_block 1024
// Wrapper for ATen
at::Tensor float2gemmlowp(at::Tensor input, float range, float offset, int num_bits, float clamp_min, float clamp_max, bool integral_zero_point) {

    int N = input.numel();
    auto output = at::zeros_like(input);
    long long n = (0x1l << num_bits) - 1;
    float scale = n/range;
    float zero_point = offset*scale;
    float zero_point = integral_zero_point ? roundf(zero_point) : zero_point;
    GEMMLowpKernel<<<block_count, thread_per_block>>>(input.data<float>(), N, output.data<float>(), scale, zero_point);

    return out;
}
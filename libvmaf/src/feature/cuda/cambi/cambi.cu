/**
 *
 *  Copyright 2016-2023 Netflix, Inc.
 *  Copyright 2025 NVIDIA Corporation.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "common.h"
#include "cuda_helper.cuh"

#include <iostream>


template<typename T, bool is_scaling>
__device__ T get_input_pixel_10b(const T* input_image, int x, int y, int stride, float ratio_x, float ratio_y, int bpc) {
    unsigned ori_x = x;
    unsigned ori_y = y;
    if constexpr (is_scaling) {
        const float start_x = ratio_x / 2.f - 0.5f;
        const float start_y = ratio_y / 2.f - 0.5f;
        ori_x = (int)(start_x + (x * ratio_x + 0.5f));
        ori_y = (int)(start_y + (y * ratio_y + 0.5f));
    }
    int idx = ori_y * stride / sizeof(T) + ori_x;

    const T max_val = (1 << bpc) - 1;
    T pixel_value = input_image[idx];
    T value_10b;
    assert(pixel_value <= max_val);  // Ensure pixel value is within valid range
    if (bpc >= 10 && std::is_same<T, uint16_t>::value) {
        int shift_factor = bpc - 10;
        int rounding_offset = shift_factor == 0 ? 0 : 1 << (shift_factor - 1);
        value_10b = (pixel_value + rounding_offset) >> shift_factor;
    } else if (bpc == 9 && std::is_same<T, uint16_t>::value) {
        value_10b = pixel_value << 1;
    } else if (bpc <= 8 && std::is_same<T, uint8_t>::value) {
        int shift_factor = 10 - bpc;
        value_10b = pixel_value >> shift_factor;
    }
    return value_10b;
}


template<typename T, int num_values_per_thread, bool is_scaling>
__device__ void preprocess(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * num_values_per_thread;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    __restrict__ T* input_image = (T*)image.data[0];
    __restrict__ T* output_image = (T*)preprocessed.data[0];

    float ratio_x = (float)image.w[0] / width;
    float ratio_y = (float)image.h[0] / height;

#pragma unroll
    for (int i = 0; i < num_values_per_thread; i++) {
        if (x + i < width && y < height) {
            T pixel_value = get_input_pixel_10b<T, is_scaling>(input_image, x + i, y, image.stride[0], ratio_x, ratio_y, image.bpc);
            if (enc_bitdepth < 10) {
                T pixel_value_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value, 1);
                T pixel_value_low = 0;
                T pixel_value_low_rhs = 0;
                if (y < height - 1) {
                    T pixel_value_low = get_input_pixel_10b<T, is_scaling>(input_image, x + i, y + 1, image.stride[0], ratio_x, ratio_y, image.bpc);
                    T pixel_value_low_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value_low, 1);
                }
                // average 4 pixels to reduce bitdepth from 10 to enc_bitdepth
                output_image[y * preprocessed.stride[0]/sizeof(T) + (x + i)] = (pixel_value + pixel_value_rhs + pixel_value_low + pixel_value_low_rhs) >> 2;
            } else {
                output_image[y * preprocessed.stride[0]/sizeof(T) + (x + i)] = pixel_value;
            }
        } else {
            // with this else branch we ensure that we read 0's after image boundary
            T pixel_value = 0;
            T pixel_value_rhs = __shfl_down_sync(0xFFFFFFFF,  pixel_value, 1);
        }

    }
}

#define TEMPALTE_PREPROCESS(type, num_values_per_thread, scaling)                                                                             \
__global__ void preprocess_##type##_##scaling(const VmafPicture image, VmafPicture preprocessed, int width, int height, int enc_bitdepth) {   \
    preprocess<type, num_values_per_thread, scaling>(image, preprocessed, width, height, enc_bitdepth);                                       \
}

template<bool apply_decimation>
__device__ void decimate_and_filter_kernel(
    const uint16_t* __restrict__ input,
    uint16_t* __restrict__ output,
    int in_width, int in_height,
    int out_width, int out_height,
    int stride)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= out_height || j >= out_width) return;

    // Decimate: sample every 2nd pixel (if apply_decimation is true)
    uint16_t value;
    if (apply_decimation) {
        value = input[(i << 1) * stride + (j << 1)];
    } else {
        value = input[i * stride + j];
    }

    // Load 3x3 neighborhood into shared memory for mode filter
    __shared__ uint16_t smem[18][18]; // 16x16 + 2 padding

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    smem[ty][tx] = value;

    // Load borders (only necessary threads)
    if (threadIdx.x == 0 && j > 0) {
        uint16_t left_val = apply_decimation ?
            input[(i << 1) * stride + ((j - 1) << 1)] :
            input[i * stride + (j - 1)];
        smem[ty][0] = left_val;
    }
    if (threadIdx.x == blockDim.x - 1 && j < out_width - 1) {
        uint16_t right_val = apply_decimation ?
            input[(i << 1) * stride + ((j + 1) << 1)] :
            input[i * stride + (j + 1)];
        smem[ty][tx + 1] = right_val;
    }
    if (threadIdx.y == 0 && i > 0) {
        uint16_t top_val = apply_decimation ?
            input[((i - 1) << 1) * stride + (j << 1)] :
            input[(i - 1) * stride + j];
        smem[0][tx] = top_val;
    }
    if (threadIdx.y == blockDim.y - 1 && i < out_height - 1) {
        uint16_t bottom_val = apply_decimation ?
            input[((i + 1) << 1) * stride + (j << 1)] :
            input[(i + 1) * stride + j];
        smem[ty + 1][tx] = bottom_val;
    }

    __syncthreads();

    // Helper: mode3 function
    auto mode3 = [](uint16_t a, uint16_t b, uint16_t c) -> uint16_t {
        if (a == b || a == c) return a;
        if (b == c) return b;
        return min(a, min(b, c));
    };

    // Apply 3x3 mode filter (horizontal then vertical)
    uint16_t h_filtered;
    if (j == 0 || j == out_width - 1) {
        h_filtered = smem[ty][tx];
    } else {
        h_filtered = mode3(smem[ty][tx - 1], smem[ty][tx], smem[ty][tx + 1]);
    }

    uint16_t result;
    if (i == 0 || i == out_height - 1) {
        result = h_filtered;
    } else {
        uint16_t h_top = (j == 0 || j == out_width - 1) ?
            smem[ty - 1][tx] :
            mode3(smem[ty - 1][tx - 1], smem[ty - 1][tx], smem[ty - 1][tx + 1]);
        uint16_t h_bottom = (j == 0 || j == out_width - 1) ?
            smem[ty + 1][tx] :
            mode3(smem[ty + 1][tx - 1], smem[ty + 1][tx], smem[ty + 1][tx + 1]);
        result = mode3(h_top, h_filtered, h_bottom);
    }

    output[i * stride + j] = result;
}

#define TEMPALTE_DECIMATE_FILTER(decimate)                                                                                                                                                       \
__global__ void decimate_and_filter_kernel_##decimate(const uint16_t* __restrict__ input, uint16_t* __restrict__ output, \
                                                      int in_width, int in_height, int out_width, int out_height, int stride) {  \
    decimate_and_filter_kernel<decimate>(input, output,in_width,  in_height,out_width, out_height,stride);                      \
}

extern "C" {
    TEMPALTE_PREPROCESS(uint8_t, 4, true);
    TEMPALTE_PREPROCESS(uint16_t, 2, true);
    TEMPALTE_PREPROCESS(uint8_t, 4, false);
    TEMPALTE_PREPROCESS(uint16_t, 2, false);

    TEMPALTE_DECIMATE_FILTER(true);
    TEMPALTE_DECIMATE_FILTER(false);
}

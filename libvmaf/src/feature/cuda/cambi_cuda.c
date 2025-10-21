/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <common.h>
#include <errno.h>
#include <picture_cuda.h>
#include <stdio.h>
#include <string.h>

#include "common/macros.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "mem.h"
#include "mkdirp.h"
#include "picture.h"
#include "cambi.h"

extern const unsigned char cambi_ptx[];

typedef struct CambiCudaBuffers {
    VmafCudaBuffer *c_values;
    VmafCudaBuffer *mask_dp;
    VmafCudaBuffer *c_values_histograms;
    VmafCudaBuffer *filter_mode_buffer;
    VmafCudaBuffer *diffs_to_consider;
    VmafCudaBuffer *tvi_for_diff;
    VmafCudaBuffer *derivative_buffer;
    VmafCudaBuffer *diff_weights;
    VmafCudaBuffer *all_diffs;
} CambiCudaBuffers;


typedef struct CambiCudaState {
    CUevent event, finished;
    CUstream str, host_stream;
    CUfunction preprocess_u8_s, preprocess_u16_s;
    CUfunction preprocess_u8, preprocess_u16;
    CambiCudaBuffers device_buffers;
    void *write_score_parameters;
} CambiCudaState;

typedef struct write_score_parameters_cambi {
    VmafFeatureCollector *feature_collector;
    CambiCudaState *s;
    unsigned index;
} write_score_parameters_cambi;

extern VmafFeatureExtractor vmaf_fex_cambi;

static int allocate_aligned_and_upload_buffer(void *host_ptr, size_t size,
                                              VmafCudaBuffer *device_ptr,
                                              VmafCudaState *cuda_state) {
    int err = vmaf_cuda_buffer_alloc(cuda_state, &device_ptr, size);
    if (err)
        CHECK_CUDA(cuda_state->f, cuMemcpyHtoDAsync(device_ptr->data, host_ptr, size,
                   cuda_state->str));
    return 0;
}

static int init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h) {
    (void) pix_fmt;
    int err = vmaf_fex_cambi.init(fex, pix_fmt, bpc, w, h);
    if (err) return err;

    CambiState *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    s->cambi_cuda_state = calloc(1, sizeof(CambiCudaState));
    CambiCudaState *cu_s = s->cambi_cuda_state;

    CHECK_CUDA(cu_f, cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuStreamCreateWithPriority(&cu_s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cu_f, cuEventCreate(&cu_s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cu_f, cuEventCreate(&cu_s->finished, CU_EVENT_DEFAULT));
    CUmodule module;
    CHECK_CUDA(cu_f, cuModuleLoadData(&module, cambi_ptx));

    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u8, module, "preprocess_uint8_t_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u16, module, "preprocess_uint16_t_false"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u8_s, module, "preprocess_uint8_t_true"));
    CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_s->preprocess_u16_s, module, "preprocess_uint16_t_true"));
    //CHECK_CUDA(cu_f, cuModuleGetFunction(&cu_state->preprocess, module, "preprocess"));

    CHECK_CUDA(cu_f, cuCtxPopCurrent(NULL));

    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++)
        err |= vmaf_picture_unref(&s->pics[i]);
    if (err) return err;

    int alloc_w = s->full_ref ? MAX(s->src_width, s->enc_width) : s->enc_width;
    int alloc_h = s->full_ref ? MAX(s->src_height, s->enc_height) : s->enc_height;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        VmafPicture pic;
        VmafCudaCookie cookie;
        cookie.bpc = 10;
        // only allocate the luma plane
        cookie.pix_fmt = VMAF_PIX_FMT_YUV400P;
        cookie.h = alloc_w;
        cookie.w = alloc_h;
        cookie.state = fex->cu_state;
        err |= vmaf_cuda_picture_alloc(&pic, &cookie);
        err |= vmaf_picture_ref(&s->pics[i], &pic);
    }
    if (err) return err;

    // upload CambiBuffers to constant CUDA memory
    err |= allocate_aligned_and_upload_buffer(s->buffers.c_values, s->buffers.c_values_size,
                                              cu_s->device_buffers.c_values, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.mask_dp, s->buffers.mask_dp_size,
                                              cu_s->device_buffers.mask_dp, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.c_values_histograms, s->buffers.c_values_histograms_size,
                                              cu_s->device_buffers.c_values_histograms, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.filter_mode_buffer, s->buffers.filter_mode_buffer_size,
                                              cu_s->device_buffers.filter_mode_buffer, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.diffs_to_consider, s->buffers.diffs_to_consider_size,
                                              cu_s->device_buffers.diffs_to_consider, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.tvi_for_diff, s->buffers.tvi_for_diff_size,
                                              cu_s->device_buffers.tvi_for_diff, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.derivative_buffer, s->buffers.derivative_buffer_size,
                                              cu_s->device_buffers.derivative_buffer, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.diff_weights, s->buffers.diff_weights_size,
                                              cu_s->device_buffers.diff_weights, fex->cu_state);
    err |= allocate_aligned_and_upload_buffer(s->buffers.all_diffs, s->buffers.all_diffs_size,
                                              cu_s->device_buffers.all_diffs, fex->cu_state);

    cu_s->write_score_parameters = malloc(sizeof(write_score_parameters_cambi));
    if (!cu_s->write_score_parameters) return -ENOMEM;

    ((write_score_parameters_cambi *) cu_s->write_score_parameters)->s = s;
    return err;
}

static void write_scores(write_score_parameters_cambi *params) {
    VmafFeatureCollector *feature_collector = params->feature_collector;
    CambiState *s = params->s;
    // TODO get the scores from device memory
    double dist_score = 0;
    double src_score = 0;
    double combined_score = 0;

    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "Cambi_feature_cambi_score",
        MIN(dist_score, s->cambi_max_val), params->index
    );
    if (err) return;

    if (s->full_ref) {
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_source",
            MIN(src_score, s->cambi_max_val), params->index
        );
        if (err) return;

        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_full_reference",
            MIN(combined_score, s->cambi_max_val), params->index
        );
        if (err) return;
    }
}


static int preprocess(VmafFeatureExtractor *fex, CambiState *s, VmafPicture *pic, bool is_src) {
    CudaFunctions *cu_f = fex->cu_state->f;
    CambiCudaState *cuda_state = s->cambi_cuda_state;

    int width = is_src ? s->src_width : s->enc_width;
    int height = is_src ? s->src_height : s->enc_height;

    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cu_f, cuEventSynchronize(cuda_state->finished));

    {
        // preprocess
        int block_dim_x = 32;
        int block_dim_y = 8;
        void *kernelParams[] = {
            pic, &s->pics[0], &width, &height, &s->enc_bitdepth
        };
        bool is_scaling = !(pic->w[0] == width && pic->h[0] == height);

        if (s->enc_bitdepth <= 8) {
            const int values_per_thread = 4;
            // overlap by one to accommodate for filters
            int grid_dim_x = DIV_ROUND_UP(width, (block_dim_x - 1) * values_per_thread);
            int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

            if (is_scaling) {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u8_s, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            } else {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u8, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            }
        } else {
            const int values_per_thread = 2;
            // overlap by one to accommodate for filters
            int grid_dim_x = DIV_ROUND_UP(width, (block_dim_x - 1) * values_per_thread);
            int grid_dim_y = DIV_ROUND_UP(height, block_dim_y);

            if (is_scaling) {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u16_s, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            } else {
                CHECK_CUDA(cu_f, cuLaunchKernel(cuda_state->preprocess_u16, grid_dim_x, grid_dim_y, 1,
                               block_dim_x, block_dim_y, 1,
                               0, vmaf_cuda_picture_get_stream(pic), kernelParams, NULL));
            }
        }
    }

    // download device scores
    CHECK_CUDA(cu_f, cuStreamSynchronize(cuda_state->host_stream));
    //CHECK_CUDA(cu_f, cuMemcpyDtoHAsync(buf->results_host, buf->tmp_res->data, sizeof(int64_t) * RES_BUFFER_SIZE, s->str));
    CHECK_CUDA(cu_f, cuEventRecord(cuda_state->finished, cuda_state->str));

    return 0;
}

static double combine_dist_src_scores(double dist_score, double src_score) {
    return MAX(0, dist_score - src_score);
}

static int extract(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector) {
    (void) ref_pic_90;
    (void) dist_pic_90;

    CambiState *s = fex->priv;
    CudaFunctions *cu_f = fex->cu_state->f;
    CambiCudaState *cu_s = s->cambi_cuda_state;
    double dist_score;
    int err = preprocess(fex, s, dist_pic, false);
    if (err) return err;


    if (s->full_ref) {
        double src_score;
        int err = preprocess(fex, s, ref_pic, true);
        if (err) return err;
    }
    write_score_parameters_cambi *data = cu_s->write_score_parameters;
    data->feature_collector = feature_collector;
    data->s = s;
    data->index = index;
    CHECK_CUDA(cu_f, cuStreamWaitEvent(cu_s->host_stream, cu_s->finished, CU_EVENT_WAIT_DEFAULT));
    CHECK_CUDA(cu_f, cuLaunchHostFunc(cu_s->host_stream, (CUhostFn*)write_scores, data));

    return 0;
}


static int flush(VmafFeatureExtractor *fex,
        VmafFeatureCollector *feature_collector)
{
    CambiState *s = fex->priv;
    CambiCudaState *cu_s = s->cambi_cuda_state;
    CudaFunctions *cu_f = fex->cu_state->f;

    CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->str));
    CHECK_CUDA(cu_f, cuStreamSynchronize(cu_s->host_stream));
    return 1;
}

static int close(VmafFeatureExtractor *fex) {
    CambiState *s = fex->priv;
    CambiCudaState *cu_s = s->cambi_cuda_state;

    int err = 0;
    for (unsigned i = 0; i < PICS_BUFFER_SIZE; i++) {
        err |= vmaf_picture_unref(&s->pics[i]);
    }

    aligned_free(s->buffers.tvi_for_diff);
    aligned_free(s->buffers.c_values);
    aligned_free(s->buffers.c_values_histograms);
    aligned_free(s->buffers.mask_dp);
    aligned_free(s->buffers.filter_mode_buffer);
    aligned_free(s->buffers.diffs_to_consider);
    aligned_free(s->buffers.diff_weights);
    aligned_free(s->buffers.all_diffs);
    aligned_free(s->buffers.derivative_buffer);

    if (s->heatmaps_path) {
        for (int scale = 0; scale < NUM_SCALES; scale++) {
            fclose(s->heatmaps_files[scale]);
        }
    }

    if (s->feature_name_dict)
        vmaf_dictionary_free(&s->feature_name_dict);

    if (cu_s->write_score_parameters) free(cu_s->write_score_parameters);
    if (s->cambi_cuda_state) free(s->cambi_cuda_state);

    return err;
}

VmafFeatureExtractor vmaf_fex_cambi_cuda = {
    .name = "cambi_cuda",
    .init = init,
    .extract = extract,
    .flush = flush,
    .close = close,
    .options = options,
    .priv_size = sizeof(CambiState),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CUDA,
};

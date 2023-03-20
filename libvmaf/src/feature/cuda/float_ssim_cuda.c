/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
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

#include <errno.h>
#include <math.h>
#include <stddef.h>

#include "feature_collector.h"
#include "feature_extractor.h"

#include "mem.h"
#include "ssim.h"
#include "picture_copy.h"

typedef struct SsimStateCuda {
    CUevent event, finished;
    CUfunction func_ssim;
    CUstream str, host_stream;
    VmafCudaBuffer *lcs_device;
    double *lcs_host;
    void* write_score_parameters;
    size_t float_stride;
    float *ref;
    float *dist;
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    double max_db;
} SsimStateCuda;

typedef struct write_score_parameters_ssim {
    VmafFeatureCollector *feature_collector;
    SsimStateCuda *s;
    unsigned h[3], w[3];
    unsigned index;
} write_score_parameters_ssim;

static const VmafOption options[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(SsimStateCuda, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(SsimStateCuda, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(SsimStateCuda, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

static int init_fex_cuda(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    (void) pix_fmt;

    SsimStateCuda *s = fex->priv;


    CHECK_CUDA(cuCtxPushCurrent(fex->cu_state->ctx));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->str, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuStreamCreateWithPriority(&s->host_stream, CU_STREAM_NON_BLOCKING, 0));
    CHECK_CUDA(cuEventCreate(&s->event, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&s->finished, CU_EVENT_DEFAULT));

    CUmodule module;
    CHECK_CUDA(cuModuleLoadData(&module, src_psnr_ptx));
    if (bpc > 8) {
        CHECK_CUDA(cuModuleGetFunction(&s->func_ssim, module, "ssim_hbd"));
    } else {
        CHECK_CUDA(cuModuleGetFunction(&s->func_ssim, module, "ssim"));
    }
    CHECK_CUDA(cuCtxPopCurrent(NULL));

    s->write_score_parameters = malloc(sizeof(write_score_parameters_ssim));
    ((write_score_parameters_ssim*)s->write_score_parameters)->s = s;

    int ret = 0; 
    ret |= vmaf_cuda_buffer_alloc(fex->cu_state, &s->lcs_device, sizeof(double) * 3);
    if (ret) goto free_ref;
    ret |= vmaf_cuda_buffer_host_alloc(fex->cu_state, &s->lcs_host, sizeof(double) * 3);
    if (ret) goto free_ref;

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    s->float_stride = ALIGN_CEIL(w * sizeof(float));
    s->ref = aligned_malloc(s->float_stride * h, 32);
    if (!s->ref) goto fail;
    s->dist = aligned_malloc(s->float_stride * h, 32);
    if (!s->dist) goto free_ref;

    return 0;

free_ref:
    free(s->ref);
fail:
    return -ENOMEM;
}

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}


static int write_scores(write_score_parameters_psnr* params)
{
    if (s->enable_db)
        score = convert_to_db(score, s->max_db);

    err = vmaf_feature_collector_append(feature_collector, "float_ssim",
                                        score, index);
    if (s->enable_lcs) {
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_l",
                                            l_score, index);
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_c",
                                            c_score, index);
        err |= vmaf_feature_collector_append(feature_collector, "float_ssim_s",
                                            s_score, index);
    }
}

static int extract_fex_cuda(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    SsimStateCuda *s = fex->priv;
    int err = 0;

    (void) ref_pic_90;
    (void) dist_pic_90;


    // this is done to ensure that the CPU does not overwrite the buffer params for 'write_scores
    CHECK_CUDA(cuEventSynchronize(s->finished));

    // Reset device SSE
    CHECK_CUDA(cuMemsetD8Async(s->sse_device->data, 0, sizeof(uint64_t) * 3, s->str));\

    picture_copy(s->ref, s->float_stride, ref_pic, 0, ref_pic->bpc);
    picture_copy(s->dist, s->float_stride, dist_pic, 0, dist_pic->bpc);

    double score, l_score, c_score, s_score;
    err = compute_ssim(s->ref, s->dist, ref_pic->w[0], ref_pic->h[0],
                       s->float_stride, s->float_stride,
                       &score, &l_score, &c_score, &s_score);
    if (err) return err;

    CHECK_CUDA(cuMemcpyDtoHAsync(s->sse_host, (CUdeviceptr)s->sse_device->data,
                sizeof(uint64_t) * 3, s->str));
    CHECK_CUDA(cuEventRecord(s->finished, s->str));
    CHECK_CUDA(cuStreamWaitEvent(s->host_stream, s->finished, CU_EVENT_WAIT_DEFAULT));
    write_score_parameters_ssim* params = s->write_score_parameters;
    params->feature_collector = feature_collector;
    for (unsigned p = 0; p < grid_dim_z; p++) {
        params->h[p] = ref_pic->h[p];
        params->w[p] = ref_pic->w[p];
    }
    params->index = index;
    CHECK_CUDA(cuLaunchHostFunc(s->host_stream, write_scores, s->write_score_parameters));


    return err;
}


static int flush_fex_cuda(VmafFeatureExtractor *fex,
                 VmafFeatureCollector *feature_collector)
{
    SsimStateCuda *s = fex->priv;
    CHECK_CUDA(cuStreamSynchronize(s->str));
    CHECK_CUDA(cuStreamSynchronize(s->host_stream));

    return (err < 0) ? err : !err;
}

static int close_fex_cuda(VmafFeatureExtractor *fex)
{
    SsimStateCuda *s = fex->priv;
    if (s->ref) aligned_free(s->ref);
    if (s->dist) aligned_free(s->dist);

    CHECK_CUDA(cuStreamSynchronize(s->host_stream));
    CHECK_CUDA(cuStreamSynchronize(s->str));
    int ret = 0;

    if (s->sse_host) {
        ret |= vmaf_cuda_buffer_host_free(fex->cu_state, s->lcs_host);
    }
    if (s->sse_device) {
        ret |= vmaf_cuda_buffer_free(fex->cu_state, s->lcs_device);
        free(s->lcs_device);
    }
    if(s->write_score_parameters) {
        free(s->write_score_parameters);
    }

    return 0;
}

static const char *provided_features[] = {
    "float_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ssim = {
    .name = "float_ssim",
    .init = init_fex_cuda,
    .extract = extract_fex_cuda,
    .options = options,
    .flush = flush_fex_cuda,
    .close = close_fex_cuda,
    .priv_size = sizeof(SsimStateCuda),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_CONTEXT_DO_NOT_OVERWRITE
};

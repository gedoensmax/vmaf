/**
*
 *  Copyright 2016-2025 Netflix, Inc.
 *  Copyright 2025 NVIDIA Corporation
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

#pragma once

/* Ratio of pixels for computation, must be 0 < topk <= 1.0 */
#define DEFAULT_CAMBI_TOPK_POOLING (0.6)

/* Window size to compute CAMBI: 65 corresponds to approximately 1 degree at 4k scale */
#define DEFAULT_CAMBI_WINDOW_SIZE (65)

/* Visibility threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886 */
#define DEFAULT_CAMBI_TVI (0.019)

/* Luminance value below which we assume any banding is not visible */
#define DEFAULT_CAMBI_VLT (0.0)

/* Max log contrast luma levels */
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)
#define MAX_CAMBI_MAX_LOG_CONTRAST (5)

/* If true, CAMBI will be run in full-reference mode and will use both the reference and distorted inputs */
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)

/* EOTF to use for the visibility threshold calculations. One of ['bt1886', 'pq']. Default: 'bt1886'. */
#define DEFAULT_CAMBI_EOTF ("bt1886")

/* CAMBI speed-up for resolutions >=1080p by down-scaling right after the sptial mask */
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)

#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)

/* Default maximum value allowed for CAMBI */
#define DEFAULT_CAMBI_MAX_VAL (1000.0)

#define NUM_SCALES 5
static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};

/* Suprathreshold contrast response */
static const int g_contrast_weights[32] = {
    1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
    8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
};

#define PICS_BUFFER_SIZE 2
#define MASK_FILTER_SIZE 7


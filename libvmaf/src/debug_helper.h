/*#
# Copyright 2026 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
*/
#ifndef LIBVMAF_DEBUG_HELPER_H
#define LIBVMAF_DEBUG_HELPER_H

#include <stdint.h>

#define DIV_ROUND_UP(x, y) (((x) + (y)-1) / (y))

static void write_buffer_to_file(const char *filename, const void *buffer, int h, int width, int stride, int bpc) {
  FILE *file = fopen(filename, "wb");
  if (!file) {
    perror("fopen");
    return;
  }
  width *= DIV_ROUND_UP(bpc, 8); // convert width to bytes
  for (int row =0 ; row < h; row++) {
    if (width != fwrite(buffer, 1, width, file)) {
      perror("fwrite not all bytes are written ");
      return;
    }
    buffer += stride;
  }
  fclose(file);
}

static void write_image(char* filename_base, int index, VmafPicture* pic) {
  char name_buffer[1024];
  snprintf(name_buffer, sizeof(name_buffer), "%s_%d_%dx%d_b%d.yuv",filename_base, index, pic->w[0], pic->h[0], pic->bpc);
  write_buffer_to_file(name_buffer, pic->data[0], pic->h[0], pic->w[0], pic->stride[0], pic->bpc);
}

static void write_buffer_as_image(char* filename_base, int index, int width, int height, int stride, int bpc, const void* image_buffer) {
  char name_buffer[1024];
  snprintf(name_buffer, sizeof(name_buffer), "%s_%d_%dx%d_b%d.yuv",filename_base, index, width, height, bpc);
  write_buffer_to_file(name_buffer, image_buffer, height, width, stride, bpc);
}

#endif // LIBVMAF_DEBUG_HELPER_H

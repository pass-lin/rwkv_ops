/* 在 glibc 2.41+ 系统上，为 CUDA 13.1 的 math_functions.h 禁用冲突的 IEC 60559 C23 rsqrt/rsqrtf 声明。 */
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>

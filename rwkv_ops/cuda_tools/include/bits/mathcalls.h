/* Wrapper to disable IEC 60559 C23 rsqrt/rsqrtf declarations that conflict
   with CUDA 13.1's math_functions.h on glibc 2.41+ systems.
   We only undefine the feature-test macro for this header and then include
   the real glibc bits/mathcalls.h. */
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#include_next <bits/mathcalls.h>

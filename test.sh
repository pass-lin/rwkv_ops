python test_script/test_jax_cuda_kernel.py
python test_script/test_tf_cuda_kernel.py
python test_script/test_torch_cuda_kernel.py

python test_script/test_rwkv6_kernel.py --backend torch --kernel-type cuda
python test_script/test_rwkv6_kernel.py --backend torch --kernel-type native
#python test_script/test_rwkv6_kernel.py --backend jax --kernel-type cuda
python test_script/test_rwkv6_kernel.py --backend jax --kernel-type native
python test_script/test_rwkv6_kernel.py --backend tensorflow --kernel-type native
python test_script/test_rwkv6_kernel.py --backend numpy --kernel-type native

python test_script/test_rwkv7_kernel.py --backend jax --kernel-type triton
python test_script/test_rwkv7_kernel.py --backend jax --kernel-type native
python test_script/test_rwkv7_kernel.py --backend torch --kernel-type native
python test_script/test_rwkv7_kernel.py --backend torch --kernel-type triton
python test_script/test_rwkv7_kernel.py --backend tensorflow --kernel-type native
python test_script/test_rwkv7_kernel.py --backend numpy --kernel-type native
python test_script/test_rwkv7_value.py

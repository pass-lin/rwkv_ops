pip install -e .
python test_script/test_rwkv7_kernel.py --backend jax --kernel-type triton
python test_script/test_rwkv7_kernel.py --backend jax --kernel-type native
python test_script/test_rwkv7_kernel.py --backend torch --kernel-type native
python test_script/test_rwkv7_kernel.py --backend torch --kernel-type triton
python test_script/test_rwkv7_kernel.py --backend tensorflow --kernel-type native
python test_script/test_rwkv7_kernel.py --backend numpy --kernel-type native
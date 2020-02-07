# $LLVM_HOME/bin/clang++ -v -g $1 --cuda-path=$CUDA_HOME -L$CUDA_LIB --cuda-gpu-arch=sm_52 -lcudart -ldl -lrt -pthread
rm -f *.fatbin *.ll *.o *.ptx *.bc

echo "[+] Host/Device LLVM IR..."
echo 

$LLVM_HOME/bin/clang++ -x cuda -S $1 -emit-llvm \
  --cuda-gpu-arch=sm_52 --cuda-path=$CUDA_HOME  \
  -no-integrated-as
#step 0.A - generate host/device llvm IR

#step 1 - Compile .bc to .ptx
echo "[+] Device IR to PTX..."
echo

$LLVM_HOME/bin/llc input-cuda-nvptx64-nvidia-cuda-sm_52.ll \
-o input-cuda-nvptx64-nvidia-cuda-sm_52.ptx -march=nvptx64 -mcpu=sm_52

echo "[+] Device PTX to Cubin"
echo
nvcc --cubin input-cuda-nvptx64-nvidia-cuda-sm_52.ptx -o input-cuda-nvptx64-nvidia-cuda-sm_52.cubin -arch=sm_52


# #step 2 - Create SAAS file
# /usr/local/cuda/bin/ptxas -m64 -g --dont-merge-basicblocks --return-at-end -v --gpu-name sm_52 --output-file /tmp/input-bae51e.o /tmp/input-fac573.s
echo "[+] PTX to SAAS..."
echo
/usr/local/cuda/bin/ptxas -m64 -g -v --gpu-name sm_52 \
--output-file input-cuda-nvptx64-nvidia-cuda-sm_52.o input-cuda-nvptx64-nvidia-cuda-sm_52.ptx


# #step 3 - Make the SAAS a fat binary
# /usr/local/cuda/bin/fatbinary -64 --create /tmp/input-06f271.fatbin -g --image=profile=sm_52,file=/tmp/input-bae51e.o --image=profile=compute_52,file=/tmp/input-fac573.s
echo
echo "[+] SAAS to FatBinary..."
echo #GOOD
/usr/local/cuda/bin/fatbinary -64 -g --create input-cuda-nvptx64-nvidia-cuda-sm_52.fatbin \
--image=profile=sm_52,file=input-cuda-nvptx64-nvidia-cuda-sm_52.o\
--image=profile=compute_52,file=input-cuda-nvptx64-nvidia-cuda-sm_52.ptx


$LLVM_HOME/bin/clang++ -cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -emit-obj -no-integrated-as -fcuda-is-device
-main-file-name input.cu -mrelocation-model static -target-cpu x86-64 -fcuda-include-gpubinary input-cuda-nvptx64-nvidia-cuda-sm_52.fatbin -o input.o -x cuda input.cu

# #step 4 - Link the fat binary to the host code
# /usr/bin/ld -z relro --hash-style=gnu --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o a.out \
# /usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../x86_64-linux-gnu/crt1.o \
# /usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../x86_64-linux-gnu/crti.o \
# /usr/lib/gcc/x86_64-linux-gnu/7.4.0/crtbegin.o \
# -L/usr/local/cuda/lib64 
# -L/usr/lib/gcc/x86_64-linux-gnu/7.4.0 
# -L/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../x86_64-linux-gnu 
# -L/lib/x86_64-linux-gnu 
# -L/lib/../lib64 -L/usr/lib/x86_64-linux-gnu 
# -L/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../.. 
# -L/home/gkarlos/s/llvm/10/bin/../lib -L/lib 
# -L/usr/lib 
# /tmp/input-f353de.o 
# -lcudart -ldl -lrt -lstdc++ -lm -lgcc_s -lgcc -lpthread -lc -lgcc_s -lgcc 
# /usr/lib/gcc/x86_64-linux-gnu/7.4.0/crtend.o 
# /usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../x86_64-linux-gnu/crtn.o
# /usr/bin/ld -hash-style=gnu --eh-frame-hdr -m elf_x86_64 -o input input.o input-cuda-nvptx64-nvidia-cuda-sm_52.fatbin -L$CUDA_LIB -lcudart -ldl -lrt -lpthread 
# nvlink -arch=sm_52 -cpu-arch=X86_64 -g -o input input.o input-cuda-nvptx64-nvidia-cuda-sm_52.fatbin
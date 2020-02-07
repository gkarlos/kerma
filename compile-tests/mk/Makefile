SM   = sm_52
CP   = compute_52
SRC  = input.cu
EXE  = input
DEBUG = -g 
OPTS = -O2 -std=c++11
CFLAGS =
CLANG = ${LLVM_BIN}/clang++
CLANG_LIB = ${LLVM_LIB}/clang/10.0.0
LLC = ${LLVM_BIN}/llc
PTXAS = ptxas
FATBIN = fatbinary

PASS_LIB=

clean :
	rm -f *.cubin *.ll *.o *.ptx *.s *.bc *.fatbin *.cui

device.bc :
	$(info ## Generating device.bc )
	$(CLANG) $(DEBUG) $(OPTS) $(CFLAGS) -x cuda -c --cuda-device-only -emit-llvm $(SRC) -o device.bc --cuda-path=${CUDA_HOME} --cuda-gpu-arch=sm_52

device.ptx : device.bc
	$(info ## Generating device.ptx )
	$(LLC) device.bc -filetype=asm -o device.ptx

device.o : device.ptx
	$(info ## Generating device.o )
	$(PTXAS) --gpu-name $(SM) device.ptx -o device.o -v

device.fatbin : device.o
	$(info ## Generating device.fatbin )
	$(FATBIN) -64 --create device.fatbin --image=profile=$(SM),file=device.o --image=profile=$(CP),file=device.ptx  

host.cui : $(SRC)
	$(info ## Generating host.cui )
	$(CLANG) $(DEBUG) $(OPTS) $(CFLAGS) -E --cuda-host-only -g $(SRC) -o host.cui

host.bc : device.fatbin host.cui
	$(info ## Generating host.bc )
	$(CLANG) \
		-cc1 -v -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -main-file-name $(SRC) $(OPTS) -o host.bc \
		-x cuda-cpp-output host.cui -fcuda-include-gpubinary device.fatbin -emit-llvm-bc -emit-llvm-uselists\
		-target-sdk-version=10.1 -disable-free -resource-dir $(CLANG_LIB) \
		-mrelocation-model static -mthread-model posix -fmath-errno -mframe-pointer=all -masm-verbose -mconstructor-aliases \
		-munwind-tables -fuse-init-array -target-cpu x86-64 -v -dwarf-column-info -debug-info-kind=limited -dwarf-version=4 \
		-debugger-tuning=gdb -fdeprecated-macro -ferror-limit 19 -fmessage-length 0 -fcxx-exceptions -fexceptions \
		-fdiagnostics-show-option -pthread -fobjc-runtime=gcc
	 
	# -fdebug-compilation-dir /home/shen956/llvm/llvm/lib/Transforms/Hello/compile
host.o : host.bc
	$(info ## Generating host.o )
	$(CLANG) host.bc -c -o host.o

all : host.o
	$(CLANG) $(DEBUG) host.o -o $(EXE) -L${CUDA_LIB} -lcudart -ldl -lrt -lm -pthread $(PASS_LIB)
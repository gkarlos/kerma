SM   = sm_52
CP   = compute_52
SRC  = input.cu
EXE  = input
DEBUG = -g
OPTS = -O0 -std=c++11
CFLAGS = #-fno-discard-value-names
CLANG = ${LLVM_BIN}/clang++
CLANG_LIB = ${LLVM_HOME}/lib/clang/10.0.0
OPT = ${LLVM_BIN}/opt
LLC = ${LLVM_BIN}/llc
PTXAS = ptxas
FATBIN = fatbinary
KERMA_HOME = /home/gkarlos/Projects/kerma/build
RT_LIB=/home/gkarlos/Projects/kerma/build/lib/libKermaRT.bc
RT_LINK=/home/gkarlos/Projects/kerma/build/tools/rt-link/rt-link

PASS_LIB=

.PHONY: clean all

clean :
	rm -f *.cubin *.ll *.o *.ptx *.s *.bc *.fatbin *.cui

device.bc :
	$(info ## Generating device.bc )
	@$(CLANG) $(DEBUG) $(OPTS) $(CFLAGS) -x cuda -c -S --cuda-device-only -emit-llvm $(SRC) -o device.bc --cuda-gpu-arch=$(SM) -Xclang -disable-O0-optnone -fno-discard-value-names
	@$(CLANG) $(DEBUG) $(OPTS) $(CFLAGS) -x cuda -c -S --cuda-host-only -emit-llvm $(SRC) -o host.ll --cuda-gpu-arch=$(SM) -Xclang -disable-O0-optnone -fno-discard-value-names
	@echo

device.rt.bc: device.bc
	$(info ## Generating device.rt.bc)
	# cat device.bc > device.rt.bc
	@$(RT_LINK) device.bc --dump > device.rt.bc
	@echo

device.canonical.bc: device.rt.bc
	$(info ## Generating device.canonical.bc)
	@$(OPT) -S -mem2reg  device.rt.bc > device.canonical.bc
	@echo

device.instr.bc: device.canonical.bc
	$(info ## Generating device.instr.bc)
	@$(OPT) -S -O0 -load $(KERMA_HOME)/lib/Transforms/libKermaTransformsPlugin.so -kerma-instru --kerma-instru-mode=warp device.canonical.bc > device.instr.bc
	@echo

device.ptx : device.instr.bc
	$(info ## Generating device.ptx )
	@$(LLC) --march=nvptx64 device.instr.bc --mcpu=$(SM) -mattr=+ptx60  --filetype=asm -o device.ptx
	@echo

device.o : device.ptx
	$(info ## Generating device.o )
	@$(PTXAS) --gpu-name $(SM) device.ptx -o device.o -v
	@echo

device.fatbin : device.o
	$(info ## Generating device.fatbin )
	@$(FATBIN) -64 --create device.fatbin --image=profile=$(SM),file=device.o --image=profile=$(CP),file=device.ptx
	@echo

host.cui : $(SRC)
	$(info ## Generating host.cui )
	@$(CLANG) $(DEBUG) $(OPTS) $(CFLAGS) -E --cuda-host-only -g $(SRC) -o host.cui
	@echo

host.bc : device.fatbin host.cui
	$(info ## Generating host.bc )
	@$(CLANG) \
		-cc1 -v -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda -main-file-name $(SRC) $(OPTS) -o host.bc \
		-x cuda-cpp-output host.cui -fcuda-include-gpubinary device.fatbin -emit-llvm-bc -emit-llvm-uselists\
		-target-sdk-version=10.1 -disable-free -resource-dir $(CLANG_LIB) \
		-mrelocation-model static -mthread-model posix -fmath-errno -mframe-pointer=all -masm-verbose -mconstructor-aliases \
		-munwind-tables -fuse-init-array -target-cpu x86-64 -v -dwarf-column-info -debug-info-kind=limited -dwarf-version=4 \
		-debugger-tuning=gdb -fdeprecated-macro -ferror-limit 19 -fmessage-length 0 -fcxx-exceptions -fexceptions \
		-fdiagnostics-show-option -pthread -fobjc-runtime=gcc
	@echo

# host.instr.bc : host.bc
# 	$(info ## Generating host.instr.bc)
# 	$(OPT) -S -O0 -load $(KERMA_HOME)/lib/Transforms/libInstruCuMemtraceHostPlugin.so --memtrace-host host.bc > host.instr.bc
# 	@echo

host.o : host.bc
	$(info ## Generating host.o )
	@$(CLANG) host.bc -c -o host.o
	@echo

all : clean host.o
	$(info ## Linking final executable)
	@$(CLANG) $(DEBUG) host.o -o $(EXE) -L${CUDA_LIB} -lcudart_static -ldl -lrt -lm -pthread $(PASS_LIB)
	@echo
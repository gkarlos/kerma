[![Build Status](https://travis-ci.com/gkarlos/kerma.svg?branch=master)](https://travis-ci.com/gkarlos/kerma) [![Documentation](https://codedocs.xyz/gkarlos/kerma.svg)](https://codedocs.xyz/gkarlos/kerma/)

## KERnel Memory-access Analysis

### Installation
- Ubuntu 18.04 LTS

- LLVM 10
    ```bash
    # Download precompiled LLVM 10 binaries
    wget -c https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
    tar xf clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz -C /my/llvm/install/dir

    # Create `$LLVM_HOME` env var pointing to the llvm installation
    echo -e export LLVM_HOME=$(realpath /my/llvm/install/dir) >> ~/.bashrc 
    source ~/.bashrc
    ```
- cxxtools
    ```
    sudo apt update
    sudo apt install libcxxtools-dev
    ```
    

* * * 

### Cuda Instrumentation pipeline

```
                   +---------------+
                   |   Device IR   |
                   |Instrumentation|
                   +----+----+-----+
                        ^    |
                        |    |
                        |    |
                        |    |
          clang         +    v     llc                  ptxas             fatbinary
     +---------------> device.bc +-------> device.ptx +-------> device.o +---------> device.fatbin
     |                                                                                  +
     +                                                                                  |
program.cu                                                                              |
     +                                                                                  |
     |    clang                      clang             clang                            |
     +---------------> host.cu.inc +-------> host.bc +-------> host.o +---------------->+ clang (link)
                       (host src w.           +   ^                                     |
                        Cuda wrappers)        |   |                                     |
                                              |   |                                     v
                                              |   |                                  program.exe
                                              v   |
                                         +----+---+------+
                                         |    Host IR    |
                                         |Instrumentation|
                                         +----+---+------+
```

TODO: Add flags for each step

### Notes

#### Clang Flags
- `-fno-discard-value-names`

#### Passes to check
- https://llvm.org/doxygen/TargetLibraryInfo_8h_source.html
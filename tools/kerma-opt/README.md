# kerma-opt

This tool is responsible for running Kerma LLVM passes on a single Cuda module.

It is meant to be used instead of LLVM opt because (some) kerma passes require
both the host and the device IR files. Running those passes with LLVM opt would
require an argument for either the host or device IR file (depending on which
one we use as the default positional argument i.e which IR we run LLVM opt on.

Its main task is to:
- Read the two IR files (host/device) for the command line
- Initialize an LLVM pass manager to run the required Kerma Passes


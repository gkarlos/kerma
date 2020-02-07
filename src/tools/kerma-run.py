import sys, os, logging, argparse

from argparse import RawTextHelpFormatter

#settings
LLVM_HOME_ENV_VAR = 'LLVM_HOME' # We expect $<LLVM_HOME_ENV_VAR> to be defined
CUDA_HOME_ENV_VAR = 'CUDA_HOME' # We expect $<CUDA_HOME_ENV_VAR> to be defined
DEFAULT_CUDA      = '/usr/local/cuda' # Fallback CUDA location
DEFAULT_OUTDIR    = os.getcwd()
DEFAULT_OUTEXE    = 'a.out'

#globals
PROGRAM_NAME  = 'kerma-run'
LLVM_HOME     = None
LLVM_LIB      = None
LLVM_INC      = None
LLVM_BIN      = None
CUDA_HOME     = DEFAULT_CUDA
CUDA_LIB      = None
CUDA_INC      = None
CUDA_BIN      = None
CLANG         = 'clang++'
LLC           = 'llc'
OPT           = 'opt'
PTXAS         = 'ptxas'
FATBIN        = 'fatbinary'
DEBUG         = False
CWD           = os.getcwd()
OUTDIR        = CWD
OUTEXE        = 'a.out'
OUT           = OUTDIR + '/' + OUTEXE

Log = None

class Toolchain():
    def host_bc(self, filename, preserve_clean_copy=False):
        pass

    def device_bc(self, filename, preserve_clean_copy=False):
        pass

    def host_obj(self, host_bc, out_name):
        pass

    def link_host_binary(self, host_o, out_name):
        pass

def init_tools(args):
    global LLVM_HOME, LLVM_LIB, LLVM_INC, LLVM_BIN, CLANG, LLC, OPT
    global CUDA_HOME, CUDA_LIB, CUDA_INC, CUDA_BIN, PTXAS, FATBIN

    if LLVM_HOME_ENV_VAR in os.environ:
        LLVM_HOME = os.environ[LLVM_HOME_ENV_VAR]
    else:
        Log.error('$LLVM_HOME not found. Manual LLVM install required (https://llvm.org/docs/CMake.html)')
        sys.exit(1)

    if CUDA_HOME_ENV_VAR in os.environ:
        CUDA_HOME = os.environ[CUDA_HOME_ENV_VAR]

    LLVM_LIB  = '{0}/{1}'.format(LLVM_HOME, 'lib')
    LLVM_INC  = '{0}/{1}'.format(LLVM_HOME, 'include')
    LLVM_BIN  = '{0}/{1}'.format(LLVM_HOME, 'bin')
    CLANG     = '{0}/{1}'.format(LLVM_BIN, CLANG)
    LLC       = '{0}/{1}'.format(LLVM_BIN, LLC)
    OPT       = '{0}/{1}'.format(LLVM_BIN, OPT)
    CUDA_LIB  = '{0}/{1}'.format(CUDA_HOME, 'lib')
    CUDA_INC  = '{0}/{1}'.format(CUDA_HOME, 'include')
    CUDA_BIN  = '{0}/{1}'.format(CUDA_HOME, 'bin')
    PTXAS     = '{0}/{1}'.format(CUDA_BIN, PTXAS)
    FATBIN    = "{0}/{1}".format(CUDA_BIN, FATBIN)

    Log.info('${} - {}'.format('LLVM_HOME', LLVM_HOME))
    Log.info('${} - {}'.format('CUDA_HOME', CUDA_HOME if CUDA_HOME_ENV_VAR in os.environ else ('Not Found (Assuming {0})'.format(DEFAULT_CUDA))))
    Log.info('Using tools:\n{0}\n{1}'.format(
        "{:>12} - {}\n{:>12} - {}\n{:>12} - {}".format('clang++', CLANG, 'llc', LLC, 'opt', OPT) if LLVM_HOME else
        "\t{}".format("'clang++', 'llc', 'opt' - assumed in $PATH")
        ,
        "{:>12} - {}\n{:>12} - {}".format('ptxas', PTXAS, 'fatbinary', FATBIN) if CUDA_HOME else
        "\t{}".format("'ptxas', 'fatbinary' - assumed in $PATH"))
    )

def setup_arg_parser(parser):
    parser.add_argument("-v", "--verbose", action='count' , help="Verbose output", default=0)
    parser.add_argument('input', nargs='+', help='Input files. Currently only accept a single .cu file')


    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("-full", action="store_true", help="Run the full pipeline", default=True)
    mode_group.add_argument("-step", metavar="<step>", type=str, default=None, 
                                     choices=["device:bc", "device:ptx", "device:cubin", "device:fatbin",
                                              "host:cui", "host:bc", "host:o"], 
                                     help="Run up to a spectific step. Choose <step> from:\n"
                                          "\tdevice:bc     - Generate device LLVM bitcode\n"
                                          "\tdevice:ptx    - Generate device ptx\n"
                                          "\tdevice:cubin  - Generate device cubin\n"
                                          "\tdevice:fatbin - Generate device fatbinary\n"
                                          "\thost:cui      - Generate host cui\n"
                                          "\thost:bc       - Generate host LLVM bitcode\n"
                                          "\thost:o        - Generate host object")

    parser.add_argument("-pass", type=str, default="all", choices=["all", "mdg", "b", "c"], metavar="<pass>",
                                 help="Select an LLVM pass to run. Choose from:\n"
                                      "\tall           - Run all available passes\n"
                                      "\tmdg           - Build the Memory Dependence Graph\n"
                                      "\tkern-detect   - Detect kernel functions\n"
                                      )
    parser.add_argument('-keep', action='store_true', help="Preserve all intermediate files", default=False)
    parser.add_argument('-o', type=str, metavar='<file>', required=False, help="Write resulting binary to <file> (Defaults to $PWD/a.out)")
    parser.add_argument('-od', type=str, metavar='<outdir>', required=False, help="Dump analysis results in <outdir> (Defaults to $PWD)")
    parser.add_argument('-cxxflags', type=str, metavar='<flags>', required=False, help="Flags to be passed to Clang")

    parser.add_argument('--debug', action='store_true', help="Run in debug mode", default=False)

    return parser

def process_args(args):
    def arg_error(msg):
        print(msg)
        sys.exit(1)

    if args.verbose:
        Log.setLevel(logging.INFO)
    if args.debug or DEBUG:
        Log.setLevel(logging.DEBUG)
    
    init_tools(args)

    if len(args.input) > 1:
        arg_error("{}: error: {}".format(PROGRAM_NAME, "Multiple input files not supported. Please use a single .cu file"))
    if args.input[0].split('.')[-1] != 'cu':
        arg_error("{}: error: {}".format(PROGRAM_NAME, "Only .cu files currently supported"))

    global OUTDIR, OUTEXE

    if args.od:
        OUTDIR = args.od
    
    if args.o:
        OUTEXE = args.o

    Log.info("Output Directory: '{}'".format(OUTDIR))

    if args.step:
        args.full = False

    if args.full:
        Log.info("Executable: {}/{}".format(OUTDIR, OUTEXE))

    return args

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(message)s')
    Log    = logging.getLogger()
    Log.setLevel(logging.WARN)
    parser = setup_arg_parser(argparse.ArgumentParser(description="Kerma Static Analysis Runner", formatter_class=RawTextHelpFormatter, prog=PROGRAM_NAME))
    args   = process_args(parser.parse_args())
    

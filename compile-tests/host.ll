; ModuleID = 'input.cu'
source_filename = "input.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZSt3expf = comdat any

$_ZN4dim3C2Ejjj = comdat any

@stderr = external dso_local global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [67 x i8] c"Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\0A\00", align 1
@.str.1 = private unnamed_addr constant [28 x i8] c"\09<rows>   - number of rows\0A\00", align 1
@.str.2 = private unnamed_addr constant [29 x i8] c"\09<cols>    - number of cols\0A\00", align 1
@.str.3 = private unnamed_addr constant [35 x i8] c"\09<y1> \09 - y1 value of the speckle\0A\00", align 1
@.str.4 = private unnamed_addr constant [38 x i8] c"\09<y2>      - y2 value of the speckle\0A\00", align 1
@.str.5 = private unnamed_addr constant [39 x i8] c"\09<x1>       - x1 value of the speckle\0A\00", align 1
@.str.6 = private unnamed_addr constant [39 x i8] c"\09<x2>       - x2 value of the speckle\0A\00", align 1
@.str.7 = private unnamed_addr constant [27 x i8] c"\09<lamda>   - lambda (0,1)\0A\00", align 1
@.str.8 = private unnamed_addr constant [41 x i8] c"\09<no. of iter>   - number of iterations\0A\00", align 1
@.str.9 = private unnamed_addr constant [29 x i8] c"WG size of kernel = %d X %d\0A\00", align 1
@.str.10 = private unnamed_addr constant [39 x i8] c"rows and cols must be multiples of 16\0A\00", align 1
@.str.11 = private unnamed_addr constant [30 x i8] c"Randomizing the input matrix\0A\00", align 1
@.str.12 = private unnamed_addr constant [26 x i8] c"Start the SRAD main loop\0A\00", align 1
@.str.13 = private unnamed_addr constant [18 x i8] c"Computation Done\0A\00", align 1

; Function Attrs: noinline uwtable
define dso_local void @_Z11srad_cuda_1PfS_S_S_S_S_iif(float* %E_C, float* %W_C, float* %N_C, float* %S_C, float* %J_cuda, float* %C_cuda, i32 %cols, i32 %rows, float %q0sqr) #0 !dbg !864 {
entry:
  %E_C.addr = alloca float*, align 8
  %W_C.addr = alloca float*, align 8
  %N_C.addr = alloca float*, align 8
  %S_C.addr = alloca float*, align 8
  %J_cuda.addr = alloca float*, align 8
  %C_cuda.addr = alloca float*, align 8
  %cols.addr = alloca i32, align 4
  %rows.addr = alloca i32, align 4
  %q0sqr.addr = alloca float, align 4
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca i8*, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  store float* %E_C, float** %E_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %E_C.addr, metadata !868, metadata !DIExpression()), !dbg !869
  store float* %W_C, float** %W_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %W_C.addr, metadata !870, metadata !DIExpression()), !dbg !871
  store float* %N_C, float** %N_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %N_C.addr, metadata !872, metadata !DIExpression()), !dbg !873
  store float* %S_C, float** %S_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %S_C.addr, metadata !874, metadata !DIExpression()), !dbg !875
  store float* %J_cuda, float** %J_cuda.addr, align 8
  call void @llvm.dbg.declare(metadata float** %J_cuda.addr, metadata !876, metadata !DIExpression()), !dbg !877
  store float* %C_cuda, float** %C_cuda.addr, align 8
  call void @llvm.dbg.declare(metadata float** %C_cuda.addr, metadata !878, metadata !DIExpression()), !dbg !879
  store i32 %cols, i32* %cols.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %cols.addr, metadata !880, metadata !DIExpression()), !dbg !881
  store i32 %rows, i32* %rows.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %rows.addr, metadata !882, metadata !DIExpression()), !dbg !883
  store float %q0sqr, float* %q0sqr.addr, align 4
  call void @llvm.dbg.declare(metadata float* %q0sqr.addr, metadata !884, metadata !DIExpression()), !dbg !885
  %kernel_args = alloca i8*, i64 9, align 16, !dbg !886
  %0 = bitcast float** %E_C.addr to i8*, !dbg !886
  %1 = getelementptr i8*, i8** %kernel_args, i32 0, !dbg !886
  store i8* %0, i8** %1, !dbg !886
  %2 = bitcast float** %W_C.addr to i8*, !dbg !886
  %3 = getelementptr i8*, i8** %kernel_args, i32 1, !dbg !886
  store i8* %2, i8** %3, !dbg !886
  %4 = bitcast float** %N_C.addr to i8*, !dbg !886
  %5 = getelementptr i8*, i8** %kernel_args, i32 2, !dbg !886
  store i8* %4, i8** %5, !dbg !886
  %6 = bitcast float** %S_C.addr to i8*, !dbg !886
  %7 = getelementptr i8*, i8** %kernel_args, i32 3, !dbg !886
  store i8* %6, i8** %7, !dbg !886
  %8 = bitcast float** %J_cuda.addr to i8*, !dbg !886
  %9 = getelementptr i8*, i8** %kernel_args, i32 4, !dbg !886
  store i8* %8, i8** %9, !dbg !886
  %10 = bitcast float** %C_cuda.addr to i8*, !dbg !886
  %11 = getelementptr i8*, i8** %kernel_args, i32 5, !dbg !886
  store i8* %10, i8** %11, !dbg !886
  %12 = bitcast i32* %cols.addr to i8*, !dbg !886
  %13 = getelementptr i8*, i8** %kernel_args, i32 6, !dbg !886
  store i8* %12, i8** %13, !dbg !886
  %14 = bitcast i32* %rows.addr to i8*, !dbg !886
  %15 = getelementptr i8*, i8** %kernel_args, i32 7, !dbg !886
  store i8* %14, i8** %15, !dbg !886
  %16 = bitcast float* %q0sqr.addr to i8*, !dbg !886
  %17 = getelementptr i8*, i8** %kernel_args, i32 8, !dbg !886
  store i8* %16, i8** %17, !dbg !886
  %18 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %grid_dim, %struct.dim3* %block_dim, i64* %shmem_size, i8** %stream), !dbg !886
  %19 = load i64, i64* %shmem_size, align 8, !dbg !886
  %20 = load i8*, i8** %stream, align 8, !dbg !886
  %21 = bitcast { i64, i32 }* %grid_dim.coerce to i8*, !dbg !886
  %22 = bitcast %struct.dim3* %grid_dim to i8*, !dbg !886
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %21, i8* align 8 %22, i64 12, i1 false), !dbg !886
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 0, !dbg !886
  %24 = load i64, i64* %23, align 8, !dbg !886
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 1, !dbg !886
  %26 = load i32, i32* %25, align 8, !dbg !886
  %27 = bitcast { i64, i32 }* %block_dim.coerce to i8*, !dbg !886
  %28 = bitcast %struct.dim3* %block_dim to i8*, !dbg !886
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %27, i8* align 8 %28, i64 12, i1 false), !dbg !886
  %29 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 0, !dbg !886
  %30 = load i64, i64* %29, align 8, !dbg !886
  %31 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 1, !dbg !886
  %32 = load i32, i32* %31, align 8, !dbg !886
  %33 = bitcast i8* %20 to %struct.CUstream_st*, !dbg !886
  %call = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float*, float*, float*, float*, i32, i32, float)* @_Z11srad_cuda_1PfS_S_S_S_S_iif to i8*), i64 %24, i32 %26, i64 %30, i32 %32, i8** %kernel_args, i64 %19, %struct.CUstream_st* %33), !dbg !886
  br label %setup.end, !dbg !886

setup.end:                                        ; preds = %entry
  ret void, !dbg !887
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**)

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

; Function Attrs: noinline uwtable
define dso_local void @_Z11srad_cuda_2PfS_S_S_S_S_iiff(float* %E_C, float* %W_C, float* %N_C, float* %S_C, float* %J_cuda, float* %C_cuda, i32 %cols, i32 %rows, float %lambda, float %q0sqr) #0 !dbg !888 {
entry:
  %E_C.addr = alloca float*, align 8
  %W_C.addr = alloca float*, align 8
  %N_C.addr = alloca float*, align 8
  %S_C.addr = alloca float*, align 8
  %J_cuda.addr = alloca float*, align 8
  %C_cuda.addr = alloca float*, align 8
  %cols.addr = alloca i32, align 4
  %rows.addr = alloca i32, align 4
  %lambda.addr = alloca float, align 4
  %q0sqr.addr = alloca float, align 4
  %grid_dim = alloca %struct.dim3, align 8
  %block_dim = alloca %struct.dim3, align 8
  %shmem_size = alloca i64, align 8
  %stream = alloca i8*, align 8
  %grid_dim.coerce = alloca { i64, i32 }, align 8
  %block_dim.coerce = alloca { i64, i32 }, align 8
  store float* %E_C, float** %E_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %E_C.addr, metadata !891, metadata !DIExpression()), !dbg !892
  store float* %W_C, float** %W_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %W_C.addr, metadata !893, metadata !DIExpression()), !dbg !894
  store float* %N_C, float** %N_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %N_C.addr, metadata !895, metadata !DIExpression()), !dbg !896
  store float* %S_C, float** %S_C.addr, align 8
  call void @llvm.dbg.declare(metadata float** %S_C.addr, metadata !897, metadata !DIExpression()), !dbg !898
  store float* %J_cuda, float** %J_cuda.addr, align 8
  call void @llvm.dbg.declare(metadata float** %J_cuda.addr, metadata !899, metadata !DIExpression()), !dbg !900
  store float* %C_cuda, float** %C_cuda.addr, align 8
  call void @llvm.dbg.declare(metadata float** %C_cuda.addr, metadata !901, metadata !DIExpression()), !dbg !902
  store i32 %cols, i32* %cols.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %cols.addr, metadata !903, metadata !DIExpression()), !dbg !904
  store i32 %rows, i32* %rows.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %rows.addr, metadata !905, metadata !DIExpression()), !dbg !906
  store float %lambda, float* %lambda.addr, align 4
  call void @llvm.dbg.declare(metadata float* %lambda.addr, metadata !907, metadata !DIExpression()), !dbg !908
  store float %q0sqr, float* %q0sqr.addr, align 4
  call void @llvm.dbg.declare(metadata float* %q0sqr.addr, metadata !909, metadata !DIExpression()), !dbg !910
  %kernel_args = alloca i8*, i64 10, align 16, !dbg !911
  %0 = bitcast float** %E_C.addr to i8*, !dbg !911
  %1 = getelementptr i8*, i8** %kernel_args, i32 0, !dbg !911
  store i8* %0, i8** %1, !dbg !911
  %2 = bitcast float** %W_C.addr to i8*, !dbg !911
  %3 = getelementptr i8*, i8** %kernel_args, i32 1, !dbg !911
  store i8* %2, i8** %3, !dbg !911
  %4 = bitcast float** %N_C.addr to i8*, !dbg !911
  %5 = getelementptr i8*, i8** %kernel_args, i32 2, !dbg !911
  store i8* %4, i8** %5, !dbg !911
  %6 = bitcast float** %S_C.addr to i8*, !dbg !911
  %7 = getelementptr i8*, i8** %kernel_args, i32 3, !dbg !911
  store i8* %6, i8** %7, !dbg !911
  %8 = bitcast float** %J_cuda.addr to i8*, !dbg !911
  %9 = getelementptr i8*, i8** %kernel_args, i32 4, !dbg !911
  store i8* %8, i8** %9, !dbg !911
  %10 = bitcast float** %C_cuda.addr to i8*, !dbg !911
  %11 = getelementptr i8*, i8** %kernel_args, i32 5, !dbg !911
  store i8* %10, i8** %11, !dbg !911
  %12 = bitcast i32* %cols.addr to i8*, !dbg !911
  %13 = getelementptr i8*, i8** %kernel_args, i32 6, !dbg !911
  store i8* %12, i8** %13, !dbg !911
  %14 = bitcast i32* %rows.addr to i8*, !dbg !911
  %15 = getelementptr i8*, i8** %kernel_args, i32 7, !dbg !911
  store i8* %14, i8** %15, !dbg !911
  %16 = bitcast float* %lambda.addr to i8*, !dbg !911
  %17 = getelementptr i8*, i8** %kernel_args, i32 8, !dbg !911
  store i8* %16, i8** %17, !dbg !911
  %18 = bitcast float* %q0sqr.addr to i8*, !dbg !911
  %19 = getelementptr i8*, i8** %kernel_args, i32 9, !dbg !911
  store i8* %18, i8** %19, !dbg !911
  %20 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %grid_dim, %struct.dim3* %block_dim, i64* %shmem_size, i8** %stream), !dbg !911
  %21 = load i64, i64* %shmem_size, align 8, !dbg !911
  %22 = load i8*, i8** %stream, align 8, !dbg !911
  %23 = bitcast { i64, i32 }* %grid_dim.coerce to i8*, !dbg !911
  %24 = bitcast %struct.dim3* %grid_dim to i8*, !dbg !911
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %23, i8* align 8 %24, i64 12, i1 false), !dbg !911
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 0, !dbg !911
  %26 = load i64, i64* %25, align 8, !dbg !911
  %27 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %grid_dim.coerce, i32 0, i32 1, !dbg !911
  %28 = load i32, i32* %27, align 8, !dbg !911
  %29 = bitcast { i64, i32 }* %block_dim.coerce to i8*, !dbg !911
  %30 = bitcast %struct.dim3* %block_dim to i8*, !dbg !911
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %29, i8* align 8 %30, i64 12, i1 false), !dbg !911
  %31 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 0, !dbg !911
  %32 = load i64, i64* %31, align 8, !dbg !911
  %33 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %block_dim.coerce, i32 0, i32 1, !dbg !911
  %34 = load i32, i32* %33, align 8, !dbg !911
  %35 = bitcast i8* %22 to %struct.CUstream_st*, !dbg !911
  %call = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float*, float*, float*, float*, i32, i32, float, float)* @_Z11srad_cuda_2PfS_S_S_S_S_iiff to i8*), i64 %26, i32 %28, i64 %32, i32 %34, i8** %kernel_args, i64 %21, %struct.CUstream_st* %35), !dbg !911
  br label %setup.end, !dbg !911

setup.end:                                        ; preds = %entry
  ret void, !dbg !912
}

; Function Attrs: noinline uwtable
define dso_local void @_Z5usageiPPc(i32 %argc, i8** %argv) #0 !dbg !913 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !916, metadata !DIExpression()), !dbg !917
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !918, metadata !DIExpression()), !dbg !919
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !920
  %1 = load i8**, i8*** %argv.addr, align 8, !dbg !921
  %arrayidx = getelementptr inbounds i8*, i8** %1, i64 0, !dbg !921
  %2 = load i8*, i8** %arrayidx, align 8, !dbg !921
  %call = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %0, i8* getelementptr inbounds ([67 x i8], [67 x i8]* @.str, i64 0, i64 0), i8* %2), !dbg !922
  %3 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !923
  %call1 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %3, i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.1, i64 0, i64 0)), !dbg !924
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !925
  %call2 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %4, i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.2, i64 0, i64 0)), !dbg !926
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !927
  %call3 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.3, i64 0, i64 0)), !dbg !928
  %6 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !929
  %call4 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %6, i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.4, i64 0, i64 0)), !dbg !930
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !931
  %call5 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %7, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.5, i64 0, i64 0)), !dbg !932
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !933
  %call6 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %8, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.6, i64 0, i64 0)), !dbg !934
  %9 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !935
  %call7 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %9, i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.7, i64 0, i64 0)), !dbg !936
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !937
  %call8 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %10, i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.8, i64 0, i64 0)), !dbg !938
  call void @exit(i32 1) #9, !dbg !939
  unreachable, !dbg !939
}

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #3

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #4

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #5 !dbg !940 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !943, metadata !DIExpression()), !dbg !944
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !945, metadata !DIExpression()), !dbg !946
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.9, i64 0, i64 0), i32 16, i32 16), !dbg !947
  %0 = load i32, i32* %argc.addr, align 4, !dbg !948
  %1 = load i8**, i8*** %argv.addr, align 8, !dbg !949
  call void @_Z7runTestiPPc(i32 %0, i8** %1), !dbg !950
  ret i32 0, !dbg !951
}

declare dso_local i32 @printf(i8*, ...) #3

; Function Attrs: noinline uwtable
define dso_local void @_Z7runTestiPPc(i32 %argc, i8** %argv) #0 !dbg !952 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %rows = alloca i32, align 4
  %cols = alloca i32, align 4
  %size_I = alloca i32, align 4
  %size_R = alloca i32, align 4
  %niter = alloca i32, align 4
  %iter = alloca i32, align 4
  %I = alloca float*, align 8
  %J = alloca float*, align 8
  %lambda = alloca float, align 4
  %q0sqr = alloca float, align 4
  %sum = alloca float, align 4
  %sum2 = alloca float, align 4
  %tmp = alloca float, align 4
  %meanROI = alloca float, align 4
  %varROI = alloca float, align 4
  %J_cuda = alloca float*, align 8
  %C_cuda = alloca float*, align 8
  %E_C = alloca float*, align 8
  %W_C = alloca float*, align 8
  %N_C = alloca float*, align 8
  %S_C = alloca float*, align 8
  %r1 = alloca i32, align 4
  %r2 = alloca i32, align 4
  %c1 = alloca i32, align 4
  %c2 = alloca i32, align 4
  %c = alloca float*, align 8
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %block_x = alloca i32, align 4
  %block_y = alloca i32, align 4
  %dimBlock = alloca %struct.dim3, align 4
  %dimGrid = alloca %struct.dim3, align 4
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp92 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp92.coerce = alloca { i64, i32 }, align 4
  %agg.tmp94 = alloca %struct.dim3, align 4
  %agg.tmp95 = alloca %struct.dim3, align 4
  %agg.tmp94.coerce = alloca { i64, i32 }, align 4
  %agg.tmp95.coerce = alloca { i64, i32 }, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !953, metadata !DIExpression()), !dbg !954
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !955, metadata !DIExpression()), !dbg !956
  call void @llvm.dbg.declare(metadata i32* %rows, metadata !957, metadata !DIExpression()), !dbg !958
  call void @llvm.dbg.declare(metadata i32* %cols, metadata !959, metadata !DIExpression()), !dbg !960
  call void @llvm.dbg.declare(metadata i32* %size_I, metadata !961, metadata !DIExpression()), !dbg !962
  call void @llvm.dbg.declare(metadata i32* %size_R, metadata !963, metadata !DIExpression()), !dbg !964
  call void @llvm.dbg.declare(metadata i32* %niter, metadata !965, metadata !DIExpression()), !dbg !966
  store i32 10, i32* %niter, align 4, !dbg !966
  call void @llvm.dbg.declare(metadata i32* %iter, metadata !967, metadata !DIExpression()), !dbg !968
  call void @llvm.dbg.declare(metadata float** %I, metadata !969, metadata !DIExpression()), !dbg !970
  call void @llvm.dbg.declare(metadata float** %J, metadata !971, metadata !DIExpression()), !dbg !972
  call void @llvm.dbg.declare(metadata float* %lambda, metadata !973, metadata !DIExpression()), !dbg !974
  call void @llvm.dbg.declare(metadata float* %q0sqr, metadata !975, metadata !DIExpression()), !dbg !976
  call void @llvm.dbg.declare(metadata float* %sum, metadata !977, metadata !DIExpression()), !dbg !978
  call void @llvm.dbg.declare(metadata float* %sum2, metadata !979, metadata !DIExpression()), !dbg !980
  call void @llvm.dbg.declare(metadata float* %tmp, metadata !981, metadata !DIExpression()), !dbg !982
  call void @llvm.dbg.declare(metadata float* %meanROI, metadata !983, metadata !DIExpression()), !dbg !984
  call void @llvm.dbg.declare(metadata float* %varROI, metadata !985, metadata !DIExpression()), !dbg !986
  call void @llvm.dbg.declare(metadata float** %J_cuda, metadata !987, metadata !DIExpression()), !dbg !988
  call void @llvm.dbg.declare(metadata float** %C_cuda, metadata !989, metadata !DIExpression()), !dbg !990
  call void @llvm.dbg.declare(metadata float** %E_C, metadata !991, metadata !DIExpression()), !dbg !992
  call void @llvm.dbg.declare(metadata float** %W_C, metadata !993, metadata !DIExpression()), !dbg !994
  call void @llvm.dbg.declare(metadata float** %N_C, metadata !995, metadata !DIExpression()), !dbg !996
  call void @llvm.dbg.declare(metadata float** %S_C, metadata !997, metadata !DIExpression()), !dbg !998
  call void @llvm.dbg.declare(metadata i32* %r1, metadata !999, metadata !DIExpression()), !dbg !1000
  call void @llvm.dbg.declare(metadata i32* %r2, metadata !1001, metadata !DIExpression()), !dbg !1002
  call void @llvm.dbg.declare(metadata i32* %c1, metadata !1003, metadata !DIExpression()), !dbg !1004
  call void @llvm.dbg.declare(metadata i32* %c2, metadata !1005, metadata !DIExpression()), !dbg !1006
  call void @llvm.dbg.declare(metadata float** %c, metadata !1007, metadata !DIExpression()), !dbg !1008
  %0 = load i32, i32* %argc.addr, align 4, !dbg !1009
  %cmp = icmp eq i32 %0, 9, !dbg !1011
  br i1 %cmp, label %if.then, label %if.else, !dbg !1012

if.then:                                          ; preds = %entry
  %1 = load i8**, i8*** %argv.addr, align 8, !dbg !1013
  %arrayidx = getelementptr inbounds i8*, i8** %1, i64 1, !dbg !1013
  %2 = load i8*, i8** %arrayidx, align 8, !dbg !1013
  %call = call i32 @atoi(i8* %2) #10, !dbg !1015
  store i32 %call, i32* %rows, align 4, !dbg !1016
  %3 = load i8**, i8*** %argv.addr, align 8, !dbg !1017
  %arrayidx1 = getelementptr inbounds i8*, i8** %3, i64 2, !dbg !1017
  %4 = load i8*, i8** %arrayidx1, align 8, !dbg !1017
  %call2 = call i32 @atoi(i8* %4) #10, !dbg !1018
  store i32 %call2, i32* %cols, align 4, !dbg !1019
  %5 = load i32, i32* %rows, align 4, !dbg !1020
  %rem = srem i32 %5, 16, !dbg !1022
  %cmp3 = icmp ne i32 %rem, 0, !dbg !1023
  br i1 %cmp3, label %if.then6, label %lor.lhs.false, !dbg !1024

lor.lhs.false:                                    ; preds = %if.then
  %6 = load i32, i32* %cols, align 4, !dbg !1025
  %rem4 = srem i32 %6, 16, !dbg !1026
  %cmp5 = icmp ne i32 %rem4, 0, !dbg !1027
  br i1 %cmp5, label %if.then6, label %if.end, !dbg !1028

if.then6:                                         ; preds = %lor.lhs.false, %if.then
  %7 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !1029
  %call7 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %7, i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0)), !dbg !1031
  call void @exit(i32 1) #9, !dbg !1032
  unreachable, !dbg !1032

if.end:                                           ; preds = %lor.lhs.false
  %8 = load i8**, i8*** %argv.addr, align 8, !dbg !1033
  %arrayidx8 = getelementptr inbounds i8*, i8** %8, i64 3, !dbg !1033
  %9 = load i8*, i8** %arrayidx8, align 8, !dbg !1033
  %call9 = call i32 @atoi(i8* %9) #10, !dbg !1034
  store i32 %call9, i32* %r1, align 4, !dbg !1035
  %10 = load i8**, i8*** %argv.addr, align 8, !dbg !1036
  %arrayidx10 = getelementptr inbounds i8*, i8** %10, i64 4, !dbg !1036
  %11 = load i8*, i8** %arrayidx10, align 8, !dbg !1036
  %call11 = call i32 @atoi(i8* %11) #10, !dbg !1037
  store i32 %call11, i32* %r2, align 4, !dbg !1038
  %12 = load i8**, i8*** %argv.addr, align 8, !dbg !1039
  %arrayidx12 = getelementptr inbounds i8*, i8** %12, i64 5, !dbg !1039
  %13 = load i8*, i8** %arrayidx12, align 8, !dbg !1039
  %call13 = call i32 @atoi(i8* %13) #10, !dbg !1040
  store i32 %call13, i32* %c1, align 4, !dbg !1041
  %14 = load i8**, i8*** %argv.addr, align 8, !dbg !1042
  %arrayidx14 = getelementptr inbounds i8*, i8** %14, i64 6, !dbg !1042
  %15 = load i8*, i8** %arrayidx14, align 8, !dbg !1042
  %call15 = call i32 @atoi(i8* %15) #10, !dbg !1043
  store i32 %call15, i32* %c2, align 4, !dbg !1044
  %16 = load i8**, i8*** %argv.addr, align 8, !dbg !1045
  %arrayidx16 = getelementptr inbounds i8*, i8** %16, i64 7, !dbg !1045
  %17 = load i8*, i8** %arrayidx16, align 8, !dbg !1045
  %call17 = call double @atof(i8* %17) #10, !dbg !1046
  %conv = fptrunc double %call17 to float, !dbg !1046
  store float %conv, float* %lambda, align 4, !dbg !1047
  %18 = load i8**, i8*** %argv.addr, align 8, !dbg !1048
  %arrayidx18 = getelementptr inbounds i8*, i8** %18, i64 8, !dbg !1048
  %19 = load i8*, i8** %arrayidx18, align 8, !dbg !1048
  %call19 = call i32 @atoi(i8* %19) #10, !dbg !1049
  store i32 %call19, i32* %niter, align 4, !dbg !1050
  br label %if.end20, !dbg !1051

if.else:                                          ; preds = %entry
  %20 = load i32, i32* %argc.addr, align 4, !dbg !1052
  %21 = load i8**, i8*** %argv.addr, align 8, !dbg !1054
  call void @_Z5usageiPPc(i32 %20, i8** %21), !dbg !1055
  br label %if.end20

if.end20:                                         ; preds = %if.else, %if.end
  %22 = load i32, i32* %cols, align 4, !dbg !1056
  %23 = load i32, i32* %rows, align 4, !dbg !1057
  %mul = mul nsw i32 %22, %23, !dbg !1058
  store i32 %mul, i32* %size_I, align 4, !dbg !1059
  %24 = load i32, i32* %r2, align 4, !dbg !1060
  %25 = load i32, i32* %r1, align 4, !dbg !1061
  %sub = sub i32 %24, %25, !dbg !1062
  %add = add i32 %sub, 1, !dbg !1063
  %26 = load i32, i32* %c2, align 4, !dbg !1064
  %27 = load i32, i32* %c1, align 4, !dbg !1065
  %sub21 = sub i32 %26, %27, !dbg !1066
  %add22 = add i32 %sub21, 1, !dbg !1067
  %mul23 = mul i32 %add, %add22, !dbg !1068
  store i32 %mul23, i32* %size_R, align 4, !dbg !1069
  %28 = load i32, i32* %size_I, align 4, !dbg !1070
  %conv24 = sext i32 %28 to i64, !dbg !1070
  %mul25 = mul i64 %conv24, 4, !dbg !1071
  %call26 = call noalias i8* @malloc(i64 %mul25) #11, !dbg !1072
  %29 = bitcast i8* %call26 to float*, !dbg !1073
  store float* %29, float** %I, align 8, !dbg !1074
  %30 = load i32, i32* %size_I, align 4, !dbg !1075
  %conv27 = sext i32 %30 to i64, !dbg !1075
  %mul28 = mul i64 %conv27, 4, !dbg !1076
  %call29 = call noalias i8* @malloc(i64 %mul28) #11, !dbg !1077
  %31 = bitcast i8* %call29 to float*, !dbg !1078
  store float* %31, float** %J, align 8, !dbg !1079
  %32 = load i32, i32* %size_I, align 4, !dbg !1080
  %conv30 = sext i32 %32 to i64, !dbg !1080
  %mul31 = mul i64 4, %conv30, !dbg !1081
  %call32 = call noalias i8* @malloc(i64 %mul31) #11, !dbg !1082
  %33 = bitcast i8* %call32 to float*, !dbg !1083
  store float* %33, float** %c, align 8, !dbg !1084
  %34 = bitcast float** %J_cuda to i8**, !dbg !1085
  %35 = load i32, i32* %size_I, align 4, !dbg !1086
  %conv33 = sext i32 %35 to i64, !dbg !1086
  %mul34 = mul i64 4, %conv33, !dbg !1087
  %call35 = call i32 @cudaMalloc(i8** %34, i64 %mul34), !dbg !1088
  %36 = bitcast float** %C_cuda to i8**, !dbg !1089
  %37 = load i32, i32* %size_I, align 4, !dbg !1090
  %conv36 = sext i32 %37 to i64, !dbg !1090
  %mul37 = mul i64 4, %conv36, !dbg !1091
  %call38 = call i32 @cudaMalloc(i8** %36, i64 %mul37), !dbg !1092
  %38 = bitcast float** %E_C to i8**, !dbg !1093
  %39 = load i32, i32* %size_I, align 4, !dbg !1094
  %conv39 = sext i32 %39 to i64, !dbg !1094
  %mul40 = mul i64 4, %conv39, !dbg !1095
  %call41 = call i32 @cudaMalloc(i8** %38, i64 %mul40), !dbg !1096
  %40 = bitcast float** %W_C to i8**, !dbg !1097
  %41 = load i32, i32* %size_I, align 4, !dbg !1098
  %conv42 = sext i32 %41 to i64, !dbg !1098
  %mul43 = mul i64 4, %conv42, !dbg !1099
  %call44 = call i32 @cudaMalloc(i8** %40, i64 %mul43), !dbg !1100
  %42 = bitcast float** %S_C to i8**, !dbg !1101
  %43 = load i32, i32* %size_I, align 4, !dbg !1102
  %conv45 = sext i32 %43 to i64, !dbg !1102
  %mul46 = mul i64 4, %conv45, !dbg !1103
  %call47 = call i32 @cudaMalloc(i8** %42, i64 %mul46), !dbg !1104
  %44 = bitcast float** %N_C to i8**, !dbg !1105
  %45 = load i32, i32* %size_I, align 4, !dbg !1106
  %conv48 = sext i32 %45 to i64, !dbg !1106
  %mul49 = mul i64 4, %conv48, !dbg !1107
  %call50 = call i32 @cudaMalloc(i8** %44, i64 %mul49), !dbg !1108
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.11, i64 0, i64 0)), !dbg !1109
  %46 = load float*, float** %I, align 8, !dbg !1110
  %47 = load i32, i32* %rows, align 4, !dbg !1111
  %48 = load i32, i32* %cols, align 4, !dbg !1112
  call void @_Z13random_matrixPfii(float* %46, i32 %47, i32 %48), !dbg !1113
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1114, metadata !DIExpression()), !dbg !1116
  store i32 0, i32* %k, align 4, !dbg !1116
  br label %for.cond, !dbg !1117

for.cond:                                         ; preds = %for.inc, %if.end20
  %49 = load i32, i32* %k, align 4, !dbg !1118
  %50 = load i32, i32* %size_I, align 4, !dbg !1120
  %cmp52 = icmp slt i32 %49, %50, !dbg !1121
  br i1 %cmp52, label %for.body, label %for.end, !dbg !1122

for.body:                                         ; preds = %for.cond
  %51 = load float*, float** %I, align 8, !dbg !1123
  %52 = load i32, i32* %k, align 4, !dbg !1125
  %idxprom = sext i32 %52 to i64, !dbg !1123
  %arrayidx53 = getelementptr inbounds float, float* %51, i64 %idxprom, !dbg !1123
  %53 = load float, float* %arrayidx53, align 4, !dbg !1123
  %call54 = call float @_ZSt3expf(float %53), !dbg !1126
  %54 = load float*, float** %J, align 8, !dbg !1127
  %55 = load i32, i32* %k, align 4, !dbg !1128
  %idxprom55 = sext i32 %55 to i64, !dbg !1127
  %arrayidx56 = getelementptr inbounds float, float* %54, i64 %idxprom55, !dbg !1127
  store float %call54, float* %arrayidx56, align 4, !dbg !1129
  br label %for.inc, !dbg !1130

for.inc:                                          ; preds = %for.body
  %56 = load i32, i32* %k, align 4, !dbg !1131
  %inc = add nsw i32 %56, 1, !dbg !1131
  store i32 %inc, i32* %k, align 4, !dbg !1131
  br label %for.cond, !dbg !1132, !llvm.loop !1133

for.end:                                          ; preds = %for.cond
  %call57 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.12, i64 0, i64 0)), !dbg !1135
  store i32 0, i32* %iter, align 4, !dbg !1136
  br label %for.cond58, !dbg !1138

for.cond58:                                       ; preds = %for.inc103, %for.end
  %57 = load i32, i32* %iter, align 4, !dbg !1139
  %58 = load i32, i32* %niter, align 4, !dbg !1141
  %cmp59 = icmp slt i32 %57, %58, !dbg !1142
  br i1 %cmp59, label %for.body60, label %for.end105, !dbg !1143

for.body60:                                       ; preds = %for.cond58
  store float 0.000000e+00, float* %sum, align 4, !dbg !1144
  store float 0.000000e+00, float* %sum2, align 4, !dbg !1146
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1147, metadata !DIExpression()), !dbg !1149
  %59 = load i32, i32* %r1, align 4, !dbg !1150
  store i32 %59, i32* %i, align 4, !dbg !1149
  br label %for.cond61, !dbg !1151

for.cond61:                                       ; preds = %for.inc77, %for.body60
  %60 = load i32, i32* %i, align 4, !dbg !1152
  %61 = load i32, i32* %r2, align 4, !dbg !1154
  %cmp62 = icmp ule i32 %60, %61, !dbg !1155
  br i1 %cmp62, label %for.body63, label %for.end79, !dbg !1156

for.body63:                                       ; preds = %for.cond61
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1157, metadata !DIExpression()), !dbg !1160
  %62 = load i32, i32* %c1, align 4, !dbg !1161
  store i32 %62, i32* %j, align 4, !dbg !1160
  br label %for.cond64, !dbg !1162

for.cond64:                                       ; preds = %for.inc74, %for.body63
  %63 = load i32, i32* %j, align 4, !dbg !1163
  %64 = load i32, i32* %c2, align 4, !dbg !1165
  %cmp65 = icmp ule i32 %63, %64, !dbg !1166
  br i1 %cmp65, label %for.body66, label %for.end76, !dbg !1167

for.body66:                                       ; preds = %for.cond64
  %65 = load float*, float** %J, align 8, !dbg !1168
  %66 = load i32, i32* %i, align 4, !dbg !1170
  %67 = load i32, i32* %cols, align 4, !dbg !1171
  %mul67 = mul nsw i32 %66, %67, !dbg !1172
  %68 = load i32, i32* %j, align 4, !dbg !1173
  %add68 = add nsw i32 %mul67, %68, !dbg !1174
  %idxprom69 = sext i32 %add68 to i64, !dbg !1168
  %arrayidx70 = getelementptr inbounds float, float* %65, i64 %idxprom69, !dbg !1168
  %69 = load float, float* %arrayidx70, align 4, !dbg !1168
  store float %69, float* %tmp, align 4, !dbg !1175
  %70 = load float, float* %tmp, align 4, !dbg !1176
  %71 = load float, float* %sum, align 4, !dbg !1177
  %add71 = fadd contract float %71, %70, !dbg !1177
  store float %add71, float* %sum, align 4, !dbg !1177
  %72 = load float, float* %tmp, align 4, !dbg !1178
  %73 = load float, float* %tmp, align 4, !dbg !1179
  %mul72 = fmul contract float %72, %73, !dbg !1180
  %74 = load float, float* %sum2, align 4, !dbg !1181
  %add73 = fadd contract float %74, %mul72, !dbg !1181
  store float %add73, float* %sum2, align 4, !dbg !1181
  br label %for.inc74, !dbg !1182

for.inc74:                                        ; preds = %for.body66
  %75 = load i32, i32* %j, align 4, !dbg !1183
  %inc75 = add nsw i32 %75, 1, !dbg !1183
  store i32 %inc75, i32* %j, align 4, !dbg !1183
  br label %for.cond64, !dbg !1184, !llvm.loop !1185

for.end76:                                        ; preds = %for.cond64
  br label %for.inc77, !dbg !1187

for.inc77:                                        ; preds = %for.end76
  %76 = load i32, i32* %i, align 4, !dbg !1188
  %inc78 = add nsw i32 %76, 1, !dbg !1188
  store i32 %inc78, i32* %i, align 4, !dbg !1188
  br label %for.cond61, !dbg !1189, !llvm.loop !1190

for.end79:                                        ; preds = %for.cond61
  %77 = load float, float* %sum, align 4, !dbg !1192
  %78 = load i32, i32* %size_R, align 4, !dbg !1193
  %conv80 = sitofp i32 %78 to float, !dbg !1193
  %div = fdiv float %77, %conv80, !dbg !1194
  store float %div, float* %meanROI, align 4, !dbg !1195
  %79 = load float, float* %sum2, align 4, !dbg !1196
  %80 = load i32, i32* %size_R, align 4, !dbg !1197
  %conv81 = sitofp i32 %80 to float, !dbg !1197
  %div82 = fdiv float %79, %conv81, !dbg !1198
  %81 = load float, float* %meanROI, align 4, !dbg !1199
  %82 = load float, float* %meanROI, align 4, !dbg !1200
  %mul83 = fmul contract float %81, %82, !dbg !1201
  %sub84 = fsub contract float %div82, %mul83, !dbg !1202
  store float %sub84, float* %varROI, align 4, !dbg !1203
  %83 = load float, float* %varROI, align 4, !dbg !1204
  %84 = load float, float* %meanROI, align 4, !dbg !1205
  %85 = load float, float* %meanROI, align 4, !dbg !1206
  %mul85 = fmul contract float %84, %85, !dbg !1207
  %div86 = fdiv float %83, %mul85, !dbg !1208
  store float %div86, float* %q0sqr, align 4, !dbg !1209
  call void @llvm.dbg.declare(metadata i32* %block_x, metadata !1210, metadata !DIExpression()), !dbg !1211
  %86 = load i32, i32* %cols, align 4, !dbg !1212
  %div87 = sdiv i32 %86, 16, !dbg !1213
  store i32 %div87, i32* %block_x, align 4, !dbg !1211
  call void @llvm.dbg.declare(metadata i32* %block_y, metadata !1214, metadata !DIExpression()), !dbg !1215
  %87 = load i32, i32* %rows, align 4, !dbg !1216
  %div88 = sdiv i32 %87, 16, !dbg !1217
  store i32 %div88, i32* %block_y, align 4, !dbg !1215
  call void @llvm.dbg.declare(metadata %struct.dim3* %dimBlock, metadata !1218, metadata !DIExpression()), !dbg !1242
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %dimBlock, i32 16, i32 16, i32 1), !dbg !1242
  call void @llvm.dbg.declare(metadata %struct.dim3* %dimGrid, metadata !1243, metadata !DIExpression()), !dbg !1244
  %88 = load i32, i32* %block_x, align 4, !dbg !1245
  %89 = load i32, i32* %block_y, align 4, !dbg !1246
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %dimGrid, i32 %88, i32 %89, i32 1), !dbg !1244
  %90 = load float*, float** %J_cuda, align 8, !dbg !1247
  %91 = bitcast float* %90 to i8*, !dbg !1247
  %92 = load float*, float** %J, align 8, !dbg !1248
  %93 = bitcast float* %92 to i8*, !dbg !1248
  %94 = load i32, i32* %size_I, align 4, !dbg !1249
  %conv89 = sext i32 %94 to i64, !dbg !1249
  %mul90 = mul i64 4, %conv89, !dbg !1250
  %call91 = call i32 @cudaMemcpy(i8* %91, i8* %93, i64 %mul90, i32 1), !dbg !1251
  %95 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1252
  %96 = bitcast %struct.dim3* %dimGrid to i8*, !dbg !1252
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %95, i8* align 4 %96, i64 12, i1 false), !dbg !1252
  %97 = bitcast %struct.dim3* %agg.tmp92 to i8*, !dbg !1253
  %98 = bitcast %struct.dim3* %dimBlock to i8*, !dbg !1253
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %97, i8* align 4 %98, i64 12, i1 false), !dbg !1253
  %99 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1254
  %100 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1254
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %99, i8* align 4 %100, i64 12, i1 false), !dbg !1254
  %101 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1254
  %102 = load i64, i64* %101, align 4, !dbg !1254
  %103 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1254
  %104 = load i32, i32* %103, align 4, !dbg !1254
  %105 = bitcast { i64, i32 }* %agg.tmp92.coerce to i8*, !dbg !1254
  %106 = bitcast %struct.dim3* %agg.tmp92 to i8*, !dbg !1254
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %105, i8* align 4 %106, i64 12, i1 false), !dbg !1254
  %107 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp92.coerce, i32 0, i32 0, !dbg !1254
  %108 = load i64, i64* %107, align 4, !dbg !1254
  %109 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp92.coerce, i32 0, i32 1, !dbg !1254
  %110 = load i32, i32* %109, align 4, !dbg !1254
  %call93 = call i32 @__cudaPushCallConfiguration(i64 %102, i32 %104, i64 %108, i32 %110, i64 0, i8* null), !dbg !1254
  %tobool = icmp ne i32 %call93, 0, !dbg !1254
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1255

kcall.configok:                                   ; preds = %for.end79
  %111 = load float*, float** %E_C, align 8, !dbg !1256
  %112 = load float*, float** %W_C, align 8, !dbg !1257
  %113 = load float*, float** %N_C, align 8, !dbg !1258
  %114 = load float*, float** %S_C, align 8, !dbg !1259
  %115 = load float*, float** %J_cuda, align 8, !dbg !1260
  %116 = load float*, float** %C_cuda, align 8, !dbg !1261
  %117 = load i32, i32* %cols, align 4, !dbg !1262
  %118 = load i32, i32* %rows, align 4, !dbg !1263
  %119 = load float, float* %q0sqr, align 4, !dbg !1264
  call void @_Z11srad_cuda_1PfS_S_S_S_S_iif(float* %111, float* %112, float* %113, float* %114, float* %115, float* %116, i32 %117, i32 %118, float %119), !dbg !1255
  br label %kcall.end, !dbg !1255

kcall.end:                                        ; preds = %kcall.configok, %for.end79
  %120 = bitcast %struct.dim3* %agg.tmp94 to i8*, !dbg !1265
  %121 = bitcast %struct.dim3* %dimGrid to i8*, !dbg !1265
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %120, i8* align 4 %121, i64 12, i1 false), !dbg !1265
  %122 = bitcast %struct.dim3* %agg.tmp95 to i8*, !dbg !1266
  %123 = bitcast %struct.dim3* %dimBlock to i8*, !dbg !1266
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %122, i8* align 4 %123, i64 12, i1 false), !dbg !1266
  %124 = bitcast { i64, i32 }* %agg.tmp94.coerce to i8*, !dbg !1267
  %125 = bitcast %struct.dim3* %agg.tmp94 to i8*, !dbg !1267
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %124, i8* align 4 %125, i64 12, i1 false), !dbg !1267
  %126 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp94.coerce, i32 0, i32 0, !dbg !1267
  %127 = load i64, i64* %126, align 4, !dbg !1267
  %128 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp94.coerce, i32 0, i32 1, !dbg !1267
  %129 = load i32, i32* %128, align 4, !dbg !1267
  %130 = bitcast { i64, i32 }* %agg.tmp95.coerce to i8*, !dbg !1267
  %131 = bitcast %struct.dim3* %agg.tmp95 to i8*, !dbg !1267
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %130, i8* align 4 %131, i64 12, i1 false), !dbg !1267
  %132 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp95.coerce, i32 0, i32 0, !dbg !1267
  %133 = load i64, i64* %132, align 4, !dbg !1267
  %134 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp95.coerce, i32 0, i32 1, !dbg !1267
  %135 = load i32, i32* %134, align 4, !dbg !1267
  %call96 = call i32 @__cudaPushCallConfiguration(i64 %127, i32 %129, i64 %133, i32 %135, i64 0, i8* null), !dbg !1267
  %tobool97 = icmp ne i32 %call96, 0, !dbg !1267
  br i1 %tobool97, label %kcall.end99, label %kcall.configok98, !dbg !1268

kcall.configok98:                                 ; preds = %kcall.end
  %136 = load float*, float** %E_C, align 8, !dbg !1269
  %137 = load float*, float** %W_C, align 8, !dbg !1270
  %138 = load float*, float** %N_C, align 8, !dbg !1271
  %139 = load float*, float** %S_C, align 8, !dbg !1272
  %140 = load float*, float** %J_cuda, align 8, !dbg !1273
  %141 = load float*, float** %C_cuda, align 8, !dbg !1274
  %142 = load i32, i32* %cols, align 4, !dbg !1275
  %143 = load i32, i32* %rows, align 4, !dbg !1276
  %144 = load float, float* %lambda, align 4, !dbg !1277
  %145 = load float, float* %q0sqr, align 4, !dbg !1278
  call void @_Z11srad_cuda_2PfS_S_S_S_S_iiff(float* %136, float* %137, float* %138, float* %139, float* %140, float* %141, i32 %142, i32 %143, float %144, float %145), !dbg !1268
  br label %kcall.end99, !dbg !1268

kcall.end99:                                      ; preds = %kcall.configok98, %kcall.end
  %146 = load float*, float** %J, align 8, !dbg !1279
  %147 = bitcast float* %146 to i8*, !dbg !1279
  %148 = load float*, float** %J_cuda, align 8, !dbg !1280
  %149 = bitcast float* %148 to i8*, !dbg !1280
  %150 = load i32, i32* %size_I, align 4, !dbg !1281
  %conv100 = sext i32 %150 to i64, !dbg !1281
  %mul101 = mul i64 4, %conv100, !dbg !1282
  %call102 = call i32 @cudaMemcpy(i8* %147, i8* %149, i64 %mul101, i32 2), !dbg !1283
  br label %for.inc103, !dbg !1284

for.inc103:                                       ; preds = %kcall.end99
  %151 = load i32, i32* %iter, align 4, !dbg !1285
  %inc104 = add nsw i32 %151, 1, !dbg !1285
  store i32 %inc104, i32* %iter, align 4, !dbg !1285
  br label %for.cond58, !dbg !1286, !llvm.loop !1287

for.end105:                                       ; preds = %for.cond58
  %call106 = call i32 @cudaDeviceSynchronize(), !dbg !1289
  %call107 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.13, i64 0, i64 0)), !dbg !1290
  %152 = load float*, float** %I, align 8, !dbg !1291
  %153 = bitcast float* %152 to i8*, !dbg !1291
  call void @free(i8* %153) #11, !dbg !1292
  %154 = load float*, float** %J, align 8, !dbg !1293
  %155 = bitcast float* %154 to i8*, !dbg !1293
  call void @free(i8* %155) #11, !dbg !1294
  %156 = load float*, float** %C_cuda, align 8, !dbg !1295
  %157 = bitcast float* %156 to i8*, !dbg !1295
  %call108 = call i32 @cudaFree(i8* %157), !dbg !1296
  %158 = load float*, float** %J_cuda, align 8, !dbg !1297
  %159 = bitcast float* %158 to i8*, !dbg !1297
  %call109 = call i32 @cudaFree(i8* %159), !dbg !1298
  %160 = load float*, float** %E_C, align 8, !dbg !1299
  %161 = bitcast float* %160 to i8*, !dbg !1299
  %call110 = call i32 @cudaFree(i8* %161), !dbg !1300
  %162 = load float*, float** %W_C, align 8, !dbg !1301
  %163 = bitcast float* %162 to i8*, !dbg !1301
  %call111 = call i32 @cudaFree(i8* %163), !dbg !1302
  %164 = load float*, float** %N_C, align 8, !dbg !1303
  %165 = bitcast float* %164 to i8*, !dbg !1303
  %call112 = call i32 @cudaFree(i8* %165), !dbg !1304
  %166 = load float*, float** %S_C, align 8, !dbg !1305
  %167 = bitcast float* %166 to i8*, !dbg !1305
  %call113 = call i32 @cudaFree(i8* %167), !dbg !1306
  %168 = load float*, float** %c, align 8, !dbg !1307
  %169 = bitcast float* %168 to i8*, !dbg !1307
  call void @free(i8* %169) #11, !dbg !1308
  ret void, !dbg !1309
}

; Function Attrs: nounwind readonly
declare dso_local i32 @atoi(i8*) #6

; Function Attrs: nounwind readonly
declare dso_local double @atof(i8*) #6

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #7

declare dso_local i32 @cudaMalloc(i8**, i64) #3

; Function Attrs: noinline nounwind uwtable
define dso_local void @_Z13random_matrixPfii(float* %I, i32 %rows, i32 %cols) #8 !dbg !1310 {
entry:
  %I.addr = alloca float*, align 8
  %rows.addr = alloca i32, align 4
  %cols.addr = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store float* %I, float** %I.addr, align 8
  call void @llvm.dbg.declare(metadata float** %I.addr, metadata !1313, metadata !DIExpression()), !dbg !1314
  store i32 %rows, i32* %rows.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %rows.addr, metadata !1315, metadata !DIExpression()), !dbg !1316
  store i32 %cols, i32* %cols.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %cols.addr, metadata !1317, metadata !DIExpression()), !dbg !1318
  call void @srand(i32 7) #11, !dbg !1319
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1320, metadata !DIExpression()), !dbg !1322
  store i32 0, i32* %i, align 4, !dbg !1322
  br label %for.cond, !dbg !1323

for.cond:                                         ; preds = %for.inc4, %entry
  %0 = load i32, i32* %i, align 4, !dbg !1324
  %1 = load i32, i32* %rows.addr, align 4, !dbg !1326
  %cmp = icmp slt i32 %0, %1, !dbg !1327
  br i1 %cmp, label %for.body, label %for.end6, !dbg !1328

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1329, metadata !DIExpression()), !dbg !1332
  store i32 0, i32* %j, align 4, !dbg !1332
  br label %for.cond1, !dbg !1333

for.cond1:                                        ; preds = %for.inc, %for.body
  %2 = load i32, i32* %j, align 4, !dbg !1334
  %3 = load i32, i32* %cols.addr, align 4, !dbg !1336
  %cmp2 = icmp slt i32 %2, %3, !dbg !1337
  br i1 %cmp2, label %for.body3, label %for.end, !dbg !1338

for.body3:                                        ; preds = %for.cond1
  %call = call i32 @rand() #11, !dbg !1339
  %conv = sitofp i32 %call to float, !dbg !1339
  %div = fdiv float %conv, 0x41E0000000000000, !dbg !1341
  %4 = load float*, float** %I.addr, align 8, !dbg !1342
  %5 = load i32, i32* %i, align 4, !dbg !1343
  %6 = load i32, i32* %cols.addr, align 4, !dbg !1344
  %mul = mul nsw i32 %5, %6, !dbg !1345
  %7 = load i32, i32* %j, align 4, !dbg !1346
  %add = add nsw i32 %mul, %7, !dbg !1347
  %idxprom = sext i32 %add to i64, !dbg !1342
  %arrayidx = getelementptr inbounds float, float* %4, i64 %idxprom, !dbg !1342
  store float %div, float* %arrayidx, align 4, !dbg !1348
  br label %for.inc, !dbg !1349

for.inc:                                          ; preds = %for.body3
  %8 = load i32, i32* %j, align 4, !dbg !1350
  %inc = add nsw i32 %8, 1, !dbg !1350
  store i32 %inc, i32* %j, align 4, !dbg !1350
  br label %for.cond1, !dbg !1351, !llvm.loop !1352

for.end:                                          ; preds = %for.cond1
  br label %for.inc4, !dbg !1354

for.inc4:                                         ; preds = %for.end
  %9 = load i32, i32* %i, align 4, !dbg !1355
  %inc5 = add nsw i32 %9, 1, !dbg !1355
  store i32 %inc5, i32* %i, align 4, !dbg !1355
  br label %for.cond, !dbg !1356, !llvm.loop !1357

for.end6:                                         ; preds = %for.cond
  ret void, !dbg !1359
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local float @_ZSt3expf(float %__x) #8 comdat !dbg !1360 {
entry:
  %__x.addr = alloca float, align 4
  store float %__x, float* %__x.addr, align 4
  call void @llvm.dbg.declare(metadata float* %__x.addr, metadata !1361, metadata !DIExpression()), !dbg !1362
  %0 = load float, float* %__x.addr, align 4, !dbg !1363
  %call = call float @expf(float %0) #11, !dbg !1364
  ret float %call, !dbg !1365
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #8 comdat align 2 !dbg !1366 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !1367, metadata !DIExpression()), !dbg !1369
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !1370, metadata !DIExpression()), !dbg !1371
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !1372, metadata !DIExpression()), !dbg !1373
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !1374, metadata !DIExpression()), !dbg !1375
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !1376
  %0 = load i32, i32* %vx.addr, align 4, !dbg !1377
  store i32 %0, i32* %x, align 4, !dbg !1376
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !1378
  %1 = load i32, i32* %vy.addr, align 4, !dbg !1379
  store i32 %1, i32* %y, align 4, !dbg !1378
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !1380
  %2 = load i32, i32* %vz.addr, align 4, !dbg !1381
  store i32 %2, i32* %z, align 4, !dbg !1380
  ret void, !dbg !1382
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) #3

declare dso_local i32 @cudaDeviceSynchronize() #3

; Function Attrs: nounwind
declare dso_local void @free(i8*) #7

declare dso_local i32 @cudaFree(i8*) #3

; Function Attrs: nounwind
declare dso_local void @srand(i32) #7

; Function Attrs: nounwind
declare dso_local i32 @rand() #7

; Function Attrs: nounwind
declare dso_local float @expf(float) #7

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { noreturn nounwind }
attributes #10 = { nounwind readonly }
attributes #11 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.dbg.cu = !{!4}
!llvm.ident = !{!863}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !5, producer: "clang version 10.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !16, imports: !21, splitDebugInlining: false, nameTableKind: None)
!5 = !DIFile(filename: "input.cu", directory: "/home/gkarlos/Projects/kerma/compile-tests")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !8, line: 1020, baseType: !9, size: 32, elements: !10, identifier: "_ZTS14cudaMemcpyKind")
!8 = !DIFile(filename: "s/cuda/10/include/driver_types.h", directory: "/home/gkarlos")
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11, !12, !13, !14, !15}
!11 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0, isUnsigned: true)
!12 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1, isUnsigned: true)
!13 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2, isUnsigned: true)
!14 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3, isUnsigned: true)
!15 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4, isUnsigned: true)
!16 = !{!17, !19, !18}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!21 = !{!22, !29, !33, !35, !37, !39, !41, !45, !47, !49, !51, !53, !55, !57, !59, !61, !63, !65, !67, !69, !71, !73, !77, !79, !81, !83, !87, !92, !94, !96, !101, !105, !107, !109, !111, !113, !115, !117, !119, !121, !126, !130, !132, !137, !141, !143, !145, !147, !149, !151, !155, !157, !159, !163, !171, !175, !177, !179, !181, !183, !187, !189, !191, !195, !197, !199, !201, !203, !205, !207, !209, !211, !213, !217, !223, !225, !227, !231, !233, !235, !237, !239, !241, !243, !245, !249, !253, !255, !257, !262, !264, !266, !268, !270, !272, !274, !277, !279, !281, !283, !288, !290, !292, !294, !296, !298, !300, !302, !304, !306, !308, !310, !314, !316, !318, !320, !322, !324, !326, !328, !330, !332, !334, !336, !338, !340, !342, !344, !348, !350, !354, !356, !358, !360, !362, !364, !366, !368, !370, !372, !376, !378, !382, !384, !386, !388, !392, !394, !398, !400, !402, !404, !406, !408, !410, !412, !414, !416, !418, !420, !422, !426, !428, !432, !434, !436, !438, !440, !442, !446, !448, !450, !452, !454, !456, !458, !462, !466, !468, !470, !472, !474, !478, !480, !484, !486, !488, !490, !492, !494, !496, !500, !502, !506, !508, !510, !514, !516, !518, !520, !522, !524, !526, !530, !536, !540, !545, !547, !549, !553, !557, !570, !574, !578, !582, !586, !591, !593, !597, !601, !605, !613, !617, !621, !623, !627, !631, !635, !641, !645, !649, !651, !659, !663, !670, !672, !674, !678, !682, !686, !691, !695, !699, !700, !701, !702, !704, !705, !706, !707, !708, !709, !710, !712, !713, !714, !715, !716, !717, !718, !720, !721, !722, !723, !724, !725, !726, !727, !728, !729, !730, !731, !732, !733, !734, !735, !736, !737, !738, !739, !740, !741, !742, !743, !744, !748, !750, !752, !754, !756, !758, !760, !762, !764, !766, !768, !770, !772, !774, !776, !778, !780, !782, !784, !786, !788, !790, !792, !794, !796, !798, !800, !802, !804, !806, !808, !810, !812, !814, !816, !818, !820, !822, !824, !826, !828, !830, !832, !834, !836, !838, !840, !842, !844, !846, !848, !850, !852, !854, !856, !858}
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !24, file: !25, line: 223)
!23 = !DINamespace(name: "std", scope: null)
!24 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !25, file: !25, line: 53, type: !26, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!25 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/home/gkarlos")
!26 = !DISubroutineType(types: !27)
!27 = !{!28, !28}
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!29 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !30, file: !25, line: 224)
!30 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !25, file: !25, line: 55, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!31 = !DISubroutineType(types: !32)
!32 = !{!18, !18}
!33 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !34, file: !25, line: 225)
!34 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !25, file: !25, line: 57, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!35 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !36, file: !25, line: 226)
!36 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !25, file: !25, line: 59, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!37 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !38, file: !25, line: 227)
!38 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !25, file: !25, line: 61, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !40, file: !25, line: 228)
!40 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !25, file: !25, line: 65, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !42, file: !25, line: 229)
!42 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !25, file: !25, line: 63, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!43 = !DISubroutineType(types: !44)
!44 = !{!18, !18, !18}
!45 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !46, file: !25, line: 230)
!46 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !25, file: !25, line: 67, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!47 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !48, file: !25, line: 231)
!48 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !25, file: !25, line: 69, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!49 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !50, file: !25, line: 232)
!50 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !25, file: !25, line: 71, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!51 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !52, file: !25, line: 233)
!52 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !25, file: !25, line: 73, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!53 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !54, file: !25, line: 234)
!54 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !25, file: !25, line: 75, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!55 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !56, file: !25, line: 235)
!56 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !25, file: !25, line: 77, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!57 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !58, file: !25, line: 236)
!58 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !25, file: !25, line: 81, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!59 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !60, file: !25, line: 237)
!60 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !25, file: !25, line: 79, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!61 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !62, file: !25, line: 238)
!62 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !25, file: !25, line: 85, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!63 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !64, file: !25, line: 239)
!64 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !25, file: !25, line: 83, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!65 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !66, file: !25, line: 240)
!66 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !25, file: !25, line: 87, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!67 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !68, file: !25, line: 241)
!68 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !25, file: !25, line: 89, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!69 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !70, file: !25, line: 242)
!70 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !25, file: !25, line: 91, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !72, file: !25, line: 243)
!72 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !25, file: !25, line: 93, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!73 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !74, file: !25, line: 244)
!74 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !25, file: !25, line: 95, type: !75, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!75 = !DISubroutineType(types: !76)
!76 = !{!18, !18, !18, !18}
!77 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !78, file: !25, line: 245)
!78 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !25, file: !25, line: 97, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!79 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !80, file: !25, line: 246)
!80 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !25, file: !25, line: 99, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!81 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !82, file: !25, line: 247)
!82 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !25, file: !25, line: 101, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!83 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !84, file: !25, line: 248)
!84 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !25, file: !25, line: 103, type: !85, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!85 = !DISubroutineType(types: !86)
!86 = !{!28, !18}
!87 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !88, file: !25, line: 249)
!88 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !25, file: !25, line: 105, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!89 = !DISubroutineType(types: !90)
!90 = !{!18, !18, !91}
!91 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !28, size: 64)
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !93, file: !25, line: 250)
!93 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !25, file: !25, line: 107, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !95, file: !25, line: 251)
!95 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !25, file: !25, line: 109, type: !85, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!96 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !97, file: !25, line: 252)
!97 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !25, file: !25, line: 114, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!98 = !DISubroutineType(types: !99)
!99 = !{!100, !18}
!100 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!101 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !102, file: !25, line: 253)
!102 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !25, file: !25, line: 118, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!103 = !DISubroutineType(types: !104)
!104 = !{!100, !18, !18}
!105 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !106, file: !25, line: 254)
!106 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !25, file: !25, line: 117, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !108, file: !25, line: 255)
!108 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !25, file: !25, line: 123, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!109 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !110, file: !25, line: 256)
!110 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !25, file: !25, line: 127, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!111 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !112, file: !25, line: 257)
!112 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !25, file: !25, line: 126, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!113 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !114, file: !25, line: 258)
!114 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !25, file: !25, line: 129, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!115 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !116, file: !25, line: 259)
!116 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !25, file: !25, line: 134, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!117 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !118, file: !25, line: 260)
!118 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !25, file: !25, line: 136, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!119 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !120, file: !25, line: 261)
!120 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !25, file: !25, line: 138, type: !103, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!121 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !122, file: !25, line: 262)
!122 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !25, file: !25, line: 139, type: !123, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!123 = !DISubroutineType(types: !124)
!124 = !{!125, !125}
!125 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !127, file: !25, line: 263)
!127 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !25, file: !25, line: 141, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!128 = !DISubroutineType(types: !129)
!129 = !{!18, !18, !28}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !131, file: !25, line: 264)
!131 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !25, file: !25, line: 143, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !133, file: !25, line: 265)
!133 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !25, file: !25, line: 144, type: !134, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!134 = !DISubroutineType(types: !135)
!135 = !{!136, !136}
!136 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!137 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !138, file: !25, line: 266)
!138 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !25, file: !25, line: 146, type: !139, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!139 = !DISubroutineType(types: !140)
!140 = !{!136, !18}
!141 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !142, file: !25, line: 267)
!142 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !25, file: !25, line: 159, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!143 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !144, file: !25, line: 268)
!144 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !25, file: !25, line: 148, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !146, file: !25, line: 269)
!146 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !25, file: !25, line: 150, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!147 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !148, file: !25, line: 270)
!148 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !25, file: !25, line: 152, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !150, file: !25, line: 271)
!150 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !25, file: !25, line: 154, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !152, file: !25, line: 272)
!152 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !25, file: !25, line: 161, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!153 = !DISubroutineType(types: !154)
!154 = !{!125, !18}
!155 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !156, file: !25, line: 273)
!156 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !25, file: !25, line: 163, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!157 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !158, file: !25, line: 274)
!158 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !25, file: !25, line: 164, type: !139, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !160, file: !25, line: 275)
!160 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !25, file: !25, line: 166, type: !161, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!161 = !DISubroutineType(types: !162)
!162 = !{!18, !18, !17}
!163 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !164, file: !25, line: 276)
!164 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !25, file: !25, line: 167, type: !165, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!165 = !DISubroutineType(types: !166)
!166 = !{!167, !168}
!167 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!168 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !169, size: 64)
!169 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !170)
!170 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !172, file: !25, line: 277)
!172 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !25, file: !25, line: 168, type: !173, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!173 = !DISubroutineType(types: !174)
!174 = !{!18, !168}
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !176, file: !25, line: 278)
!176 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !25, file: !25, line: 170, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !178, file: !25, line: 279)
!178 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !25, file: !25, line: 172, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !180, file: !25, line: 280)
!180 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !25, file: !25, line: 176, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !182, file: !25, line: 281)
!182 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !25, file: !25, line: 178, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !184, file: !25, line: 282)
!184 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !25, file: !25, line: 180, type: !185, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!185 = !DISubroutineType(types: !186)
!186 = !{!18, !18, !18, !91}
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !188, file: !25, line: 283)
!188 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !25, file: !25, line: 182, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !190, file: !25, line: 284)
!190 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !25, file: !25, line: 184, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !192, file: !25, line: 285)
!192 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !25, file: !25, line: 186, type: !193, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!193 = !DISubroutineType(types: !194)
!194 = !{!18, !18, !125}
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !196, file: !25, line: 286)
!196 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !25, file: !25, line: 188, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !198, file: !25, line: 287)
!198 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !25, file: !25, line: 190, type: !98, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !200, file: !25, line: 288)
!200 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !25, file: !25, line: 192, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !202, file: !25, line: 289)
!202 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !25, file: !25, line: 194, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !204, file: !25, line: 290)
!204 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !25, file: !25, line: 196, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !206, file: !25, line: 291)
!206 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !25, file: !25, line: 198, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !208, file: !25, line: 292)
!208 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !25, file: !25, line: 200, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !210, file: !25, line: 293)
!210 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !25, file: !25, line: 202, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !212, file: !25, line: 294)
!212 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !25, file: !25, line: 204, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !214, file: !216, line: 52)
!214 = !DISubprogram(name: "abs", scope: !215, file: !215, line: 837, type: !26, flags: DIFlagPrototyped, spFlags: 0)
!215 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!216 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/bits/std_abs.h", directory: "")
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !218, file: !222, line: 83)
!218 = !DISubprogram(name: "acos", scope: !219, file: !219, line: 53, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!219 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!220 = !DISubroutineType(types: !221)
!221 = !{!167, !167}
!222 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/cmath", directory: "")
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !224, file: !222, line: 102)
!224 = !DISubprogram(name: "asin", scope: !219, file: !219, line: 55, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !226, file: !222, line: 121)
!226 = !DISubprogram(name: "atan", scope: !219, file: !219, line: 57, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !228, file: !222, line: 140)
!228 = !DISubprogram(name: "atan2", scope: !219, file: !219, line: 59, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!229 = !DISubroutineType(types: !230)
!230 = !{!167, !167, !167}
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !232, file: !222, line: 161)
!232 = !DISubprogram(name: "ceil", scope: !219, file: !219, line: 159, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !234, file: !222, line: 180)
!234 = !DISubprogram(name: "cos", scope: !219, file: !219, line: 62, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !236, file: !222, line: 199)
!236 = !DISubprogram(name: "cosh", scope: !219, file: !219, line: 71, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !238, file: !222, line: 218)
!238 = !DISubprogram(name: "exp", scope: !219, file: !219, line: 95, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !240, file: !222, line: 237)
!240 = !DISubprogram(name: "fabs", scope: !219, file: !219, line: 162, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !242, file: !222, line: 256)
!242 = !DISubprogram(name: "floor", scope: !219, file: !219, line: 165, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !244, file: !222, line: 275)
!244 = !DISubprogram(name: "fmod", scope: !219, file: !219, line: 168, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !246, file: !222, line: 296)
!246 = !DISubprogram(name: "frexp", scope: !219, file: !219, line: 98, type: !247, flags: DIFlagPrototyped, spFlags: 0)
!247 = !DISubroutineType(types: !248)
!248 = !{!167, !167, !91}
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !250, file: !222, line: 315)
!250 = !DISubprogram(name: "ldexp", scope: !219, file: !219, line: 101, type: !251, flags: DIFlagPrototyped, spFlags: 0)
!251 = !DISubroutineType(types: !252)
!252 = !{!167, !167, !28}
!253 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !254, file: !222, line: 334)
!254 = !DISubprogram(name: "log", scope: !219, file: !219, line: 104, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!255 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !256, file: !222, line: 353)
!256 = !DISubprogram(name: "log10", scope: !219, file: !219, line: 107, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !258, file: !222, line: 372)
!258 = !DISubprogram(name: "modf", scope: !219, file: !219, line: 110, type: !259, flags: DIFlagPrototyped, spFlags: 0)
!259 = !DISubroutineType(types: !260)
!260 = !{!167, !167, !261}
!261 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !167, size: 64)
!262 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !263, file: !222, line: 384)
!263 = !DISubprogram(name: "pow", scope: !219, file: !219, line: 140, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!264 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !265, file: !222, line: 421)
!265 = !DISubprogram(name: "sin", scope: !219, file: !219, line: 64, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!266 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !267, file: !222, line: 440)
!267 = !DISubprogram(name: "sinh", scope: !219, file: !219, line: 73, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!268 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !269, file: !222, line: 459)
!269 = !DISubprogram(name: "sqrt", scope: !219, file: !219, line: 143, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!270 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !271, file: !222, line: 478)
!271 = !DISubprogram(name: "tan", scope: !219, file: !219, line: 66, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!272 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !273, file: !222, line: 497)
!273 = !DISubprogram(name: "tanh", scope: !219, file: !219, line: 75, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!274 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !275, file: !222, line: 1080)
!275 = !DIDerivedType(tag: DW_TAG_typedef, name: "double_t", file: !276, line: 150, baseType: !167)
!276 = !DIFile(filename: "/usr/include/math.h", directory: "")
!277 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !278, file: !222, line: 1081)
!278 = !DIDerivedType(tag: DW_TAG_typedef, name: "float_t", file: !276, line: 149, baseType: !18)
!279 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !280, file: !222, line: 1084)
!280 = !DISubprogram(name: "acosh", scope: !219, file: !219, line: 85, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!281 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !282, file: !222, line: 1085)
!282 = !DISubprogram(name: "acoshf", scope: !219, file: !219, line: 85, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !284, file: !222, line: 1086)
!284 = !DISubprogram(name: "acoshl", scope: !219, file: !219, line: 85, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!285 = !DISubroutineType(types: !286)
!286 = !{!287, !287}
!287 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!288 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !289, file: !222, line: 1088)
!289 = !DISubprogram(name: "asinh", scope: !219, file: !219, line: 87, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!290 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !291, file: !222, line: 1089)
!291 = !DISubprogram(name: "asinhf", scope: !219, file: !219, line: 87, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!292 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !293, file: !222, line: 1090)
!293 = !DISubprogram(name: "asinhl", scope: !219, file: !219, line: 87, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!294 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !295, file: !222, line: 1092)
!295 = !DISubprogram(name: "atanh", scope: !219, file: !219, line: 89, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!296 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !297, file: !222, line: 1093)
!297 = !DISubprogram(name: "atanhf", scope: !219, file: !219, line: 89, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!298 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !299, file: !222, line: 1094)
!299 = !DISubprogram(name: "atanhl", scope: !219, file: !219, line: 89, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!300 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !301, file: !222, line: 1096)
!301 = !DISubprogram(name: "cbrt", scope: !219, file: !219, line: 152, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!302 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !303, file: !222, line: 1097)
!303 = !DISubprogram(name: "cbrtf", scope: !219, file: !219, line: 152, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!304 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !305, file: !222, line: 1098)
!305 = !DISubprogram(name: "cbrtl", scope: !219, file: !219, line: 152, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!306 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !307, file: !222, line: 1100)
!307 = !DISubprogram(name: "copysign", scope: !219, file: !219, line: 196, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!308 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !309, file: !222, line: 1101)
!309 = !DISubprogram(name: "copysignf", scope: !219, file: !219, line: 196, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!310 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !311, file: !222, line: 1102)
!311 = !DISubprogram(name: "copysignl", scope: !219, file: !219, line: 196, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!312 = !DISubroutineType(types: !313)
!313 = !{!287, !287, !287}
!314 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !315, file: !222, line: 1104)
!315 = !DISubprogram(name: "erf", scope: !219, file: !219, line: 228, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!316 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !317, file: !222, line: 1105)
!317 = !DISubprogram(name: "erff", scope: !219, file: !219, line: 228, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!318 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !319, file: !222, line: 1106)
!319 = !DISubprogram(name: "erfl", scope: !219, file: !219, line: 228, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!320 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !321, file: !222, line: 1108)
!321 = !DISubprogram(name: "erfc", scope: !219, file: !219, line: 229, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!322 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !323, file: !222, line: 1109)
!323 = !DISubprogram(name: "erfcf", scope: !219, file: !219, line: 229, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!324 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !325, file: !222, line: 1110)
!325 = !DISubprogram(name: "erfcl", scope: !219, file: !219, line: 229, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!326 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !327, file: !222, line: 1112)
!327 = !DISubprogram(name: "exp2", scope: !219, file: !219, line: 130, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!328 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !329, file: !222, line: 1113)
!329 = !DISubprogram(name: "exp2f", scope: !219, file: !219, line: 130, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!330 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !331, file: !222, line: 1114)
!331 = !DISubprogram(name: "exp2l", scope: !219, file: !219, line: 130, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!332 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !333, file: !222, line: 1116)
!333 = !DISubprogram(name: "expm1", scope: !219, file: !219, line: 119, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!334 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !335, file: !222, line: 1117)
!335 = !DISubprogram(name: "expm1f", scope: !219, file: !219, line: 119, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!336 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !337, file: !222, line: 1118)
!337 = !DISubprogram(name: "expm1l", scope: !219, file: !219, line: 119, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !339, file: !222, line: 1120)
!339 = !DISubprogram(name: "fdim", scope: !219, file: !219, line: 326, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!340 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !341, file: !222, line: 1121)
!341 = !DISubprogram(name: "fdimf", scope: !219, file: !219, line: 326, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !343, file: !222, line: 1122)
!343 = !DISubprogram(name: "fdiml", scope: !219, file: !219, line: 326, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!344 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !345, file: !222, line: 1124)
!345 = !DISubprogram(name: "fma", scope: !219, file: !219, line: 335, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!346 = !DISubroutineType(types: !347)
!347 = !{!167, !167, !167, !167}
!348 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !349, file: !222, line: 1125)
!349 = !DISubprogram(name: "fmaf", scope: !219, file: !219, line: 335, type: !75, flags: DIFlagPrototyped, spFlags: 0)
!350 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !351, file: !222, line: 1126)
!351 = !DISubprogram(name: "fmal", scope: !219, file: !219, line: 335, type: !352, flags: DIFlagPrototyped, spFlags: 0)
!352 = !DISubroutineType(types: !353)
!353 = !{!287, !287, !287, !287}
!354 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !355, file: !222, line: 1128)
!355 = !DISubprogram(name: "fmax", scope: !219, file: !219, line: 329, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!356 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !357, file: !222, line: 1129)
!357 = !DISubprogram(name: "fmaxf", scope: !219, file: !219, line: 329, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !359, file: !222, line: 1130)
!359 = !DISubprogram(name: "fmaxl", scope: !219, file: !219, line: 329, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!360 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !361, file: !222, line: 1132)
!361 = !DISubprogram(name: "fmin", scope: !219, file: !219, line: 332, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !363, file: !222, line: 1133)
!363 = !DISubprogram(name: "fminf", scope: !219, file: !219, line: 332, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!364 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !365, file: !222, line: 1134)
!365 = !DISubprogram(name: "fminl", scope: !219, file: !219, line: 332, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !367, file: !222, line: 1136)
!367 = !DISubprogram(name: "hypot", scope: !219, file: !219, line: 147, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!368 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !369, file: !222, line: 1137)
!369 = !DISubprogram(name: "hypotf", scope: !219, file: !219, line: 147, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!370 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !371, file: !222, line: 1138)
!371 = !DISubprogram(name: "hypotl", scope: !219, file: !219, line: 147, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!372 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !373, file: !222, line: 1140)
!373 = !DISubprogram(name: "ilogb", scope: !219, file: !219, line: 280, type: !374, flags: DIFlagPrototyped, spFlags: 0)
!374 = !DISubroutineType(types: !375)
!375 = !{!28, !167}
!376 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !377, file: !222, line: 1141)
!377 = !DISubprogram(name: "ilogbf", scope: !219, file: !219, line: 280, type: !85, flags: DIFlagPrototyped, spFlags: 0)
!378 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !379, file: !222, line: 1142)
!379 = !DISubprogram(name: "ilogbl", scope: !219, file: !219, line: 280, type: !380, flags: DIFlagPrototyped, spFlags: 0)
!380 = !DISubroutineType(types: !381)
!381 = !{!28, !287}
!382 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !383, file: !222, line: 1144)
!383 = !DISubprogram(name: "lgamma", scope: !219, file: !219, line: 230, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!384 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !385, file: !222, line: 1145)
!385 = !DISubprogram(name: "lgammaf", scope: !219, file: !219, line: 230, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!386 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !387, file: !222, line: 1146)
!387 = !DISubprogram(name: "lgammal", scope: !219, file: !219, line: 230, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !389, file: !222, line: 1149)
!389 = !DISubprogram(name: "llrint", scope: !219, file: !219, line: 316, type: !390, flags: DIFlagPrototyped, spFlags: 0)
!390 = !DISubroutineType(types: !391)
!391 = !{!136, !167}
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !393, file: !222, line: 1150)
!393 = !DISubprogram(name: "llrintf", scope: !219, file: !219, line: 316, type: !139, flags: DIFlagPrototyped, spFlags: 0)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !395, file: !222, line: 1151)
!395 = !DISubprogram(name: "llrintl", scope: !219, file: !219, line: 316, type: !396, flags: DIFlagPrototyped, spFlags: 0)
!396 = !DISubroutineType(types: !397)
!397 = !{!136, !287}
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !399, file: !222, line: 1153)
!399 = !DISubprogram(name: "llround", scope: !219, file: !219, line: 322, type: !390, flags: DIFlagPrototyped, spFlags: 0)
!400 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !401, file: !222, line: 1154)
!401 = !DISubprogram(name: "llroundf", scope: !219, file: !219, line: 322, type: !139, flags: DIFlagPrototyped, spFlags: 0)
!402 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !403, file: !222, line: 1155)
!403 = !DISubprogram(name: "llroundl", scope: !219, file: !219, line: 322, type: !396, flags: DIFlagPrototyped, spFlags: 0)
!404 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !405, file: !222, line: 1158)
!405 = !DISubprogram(name: "log1p", scope: !219, file: !219, line: 122, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!406 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !407, file: !222, line: 1159)
!407 = !DISubprogram(name: "log1pf", scope: !219, file: !219, line: 122, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!408 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !409, file: !222, line: 1160)
!409 = !DISubprogram(name: "log1pl", scope: !219, file: !219, line: 122, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !411, file: !222, line: 1162)
!411 = !DISubprogram(name: "log2", scope: !219, file: !219, line: 133, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!412 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !413, file: !222, line: 1163)
!413 = !DISubprogram(name: "log2f", scope: !219, file: !219, line: 133, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !415, file: !222, line: 1164)
!415 = !DISubprogram(name: "log2l", scope: !219, file: !219, line: 133, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !417, file: !222, line: 1166)
!417 = !DISubprogram(name: "logb", scope: !219, file: !219, line: 125, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !419, file: !222, line: 1167)
!419 = !DISubprogram(name: "logbf", scope: !219, file: !219, line: 125, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!420 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !421, file: !222, line: 1168)
!421 = !DISubprogram(name: "logbl", scope: !219, file: !219, line: 125, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !423, file: !222, line: 1170)
!423 = !DISubprogram(name: "lrint", scope: !219, file: !219, line: 314, type: !424, flags: DIFlagPrototyped, spFlags: 0)
!424 = !DISubroutineType(types: !425)
!425 = !{!125, !167}
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !427, file: !222, line: 1171)
!427 = !DISubprogram(name: "lrintf", scope: !219, file: !219, line: 314, type: !153, flags: DIFlagPrototyped, spFlags: 0)
!428 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !429, file: !222, line: 1172)
!429 = !DISubprogram(name: "lrintl", scope: !219, file: !219, line: 314, type: !430, flags: DIFlagPrototyped, spFlags: 0)
!430 = !DISubroutineType(types: !431)
!431 = !{!125, !287}
!432 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !433, file: !222, line: 1174)
!433 = !DISubprogram(name: "lround", scope: !219, file: !219, line: 320, type: !424, flags: DIFlagPrototyped, spFlags: 0)
!434 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !435, file: !222, line: 1175)
!435 = !DISubprogram(name: "lroundf", scope: !219, file: !219, line: 320, type: !153, flags: DIFlagPrototyped, spFlags: 0)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !437, file: !222, line: 1176)
!437 = !DISubprogram(name: "lroundl", scope: !219, file: !219, line: 320, type: !430, flags: DIFlagPrototyped, spFlags: 0)
!438 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !439, file: !222, line: 1178)
!439 = !DISubprogram(name: "nan", scope: !219, file: !219, line: 201, type: !165, flags: DIFlagPrototyped, spFlags: 0)
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !441, file: !222, line: 1179)
!441 = !DISubprogram(name: "nanf", scope: !219, file: !219, line: 201, type: !173, flags: DIFlagPrototyped, spFlags: 0)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !443, file: !222, line: 1180)
!443 = !DISubprogram(name: "nanl", scope: !219, file: !219, line: 201, type: !444, flags: DIFlagPrototyped, spFlags: 0)
!444 = !DISubroutineType(types: !445)
!445 = !{!287, !168}
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !447, file: !222, line: 1182)
!447 = !DISubprogram(name: "nearbyint", scope: !219, file: !219, line: 294, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !449, file: !222, line: 1183)
!449 = !DISubprogram(name: "nearbyintf", scope: !219, file: !219, line: 294, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !451, file: !222, line: 1184)
!451 = !DISubprogram(name: "nearbyintl", scope: !219, file: !219, line: 294, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !453, file: !222, line: 1186)
!453 = !DISubprogram(name: "nextafter", scope: !219, file: !219, line: 259, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !455, file: !222, line: 1187)
!455 = !DISubprogram(name: "nextafterf", scope: !219, file: !219, line: 259, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !457, file: !222, line: 1188)
!457 = !DISubprogram(name: "nextafterl", scope: !219, file: !219, line: 259, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !459, file: !222, line: 1190)
!459 = !DISubprogram(name: "nexttoward", scope: !219, file: !219, line: 261, type: !460, flags: DIFlagPrototyped, spFlags: 0)
!460 = !DISubroutineType(types: !461)
!461 = !{!167, !167, !287}
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !463, file: !222, line: 1191)
!463 = !DISubprogram(name: "nexttowardf", scope: !219, file: !219, line: 261, type: !464, flags: DIFlagPrototyped, spFlags: 0)
!464 = !DISubroutineType(types: !465)
!465 = !{!18, !18, !287}
!466 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !467, file: !222, line: 1192)
!467 = !DISubprogram(name: "nexttowardl", scope: !219, file: !219, line: 261, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !469, file: !222, line: 1194)
!469 = !DISubprogram(name: "remainder", scope: !219, file: !219, line: 272, type: !229, flags: DIFlagPrototyped, spFlags: 0)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !471, file: !222, line: 1195)
!471 = !DISubprogram(name: "remainderf", scope: !219, file: !219, line: 272, type: !43, flags: DIFlagPrototyped, spFlags: 0)
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !473, file: !222, line: 1196)
!473 = !DISubprogram(name: "remainderl", scope: !219, file: !219, line: 272, type: !312, flags: DIFlagPrototyped, spFlags: 0)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !475, file: !222, line: 1198)
!475 = !DISubprogram(name: "remquo", scope: !219, file: !219, line: 307, type: !476, flags: DIFlagPrototyped, spFlags: 0)
!476 = !DISubroutineType(types: !477)
!477 = !{!167, !167, !167, !91}
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !479, file: !222, line: 1199)
!479 = !DISubprogram(name: "remquof", scope: !219, file: !219, line: 307, type: !185, flags: DIFlagPrototyped, spFlags: 0)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !481, file: !222, line: 1200)
!481 = !DISubprogram(name: "remquol", scope: !219, file: !219, line: 307, type: !482, flags: DIFlagPrototyped, spFlags: 0)
!482 = !DISubroutineType(types: !483)
!483 = !{!287, !287, !287, !91}
!484 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !485, file: !222, line: 1202)
!485 = !DISubprogram(name: "rint", scope: !219, file: !219, line: 256, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!486 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !487, file: !222, line: 1203)
!487 = !DISubprogram(name: "rintf", scope: !219, file: !219, line: 256, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!488 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !489, file: !222, line: 1204)
!489 = !DISubprogram(name: "rintl", scope: !219, file: !219, line: 256, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!490 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !491, file: !222, line: 1206)
!491 = !DISubprogram(name: "round", scope: !219, file: !219, line: 298, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!492 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !493, file: !222, line: 1207)
!493 = !DISubprogram(name: "roundf", scope: !219, file: !219, line: 298, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!494 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !495, file: !222, line: 1208)
!495 = !DISubprogram(name: "roundl", scope: !219, file: !219, line: 298, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!496 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !497, file: !222, line: 1210)
!497 = !DISubprogram(name: "scalbln", scope: !219, file: !219, line: 290, type: !498, flags: DIFlagPrototyped, spFlags: 0)
!498 = !DISubroutineType(types: !499)
!499 = !{!167, !167, !125}
!500 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !501, file: !222, line: 1211)
!501 = !DISubprogram(name: "scalblnf", scope: !219, file: !219, line: 290, type: !193, flags: DIFlagPrototyped, spFlags: 0)
!502 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !503, file: !222, line: 1212)
!503 = !DISubprogram(name: "scalblnl", scope: !219, file: !219, line: 290, type: !504, flags: DIFlagPrototyped, spFlags: 0)
!504 = !DISubroutineType(types: !505)
!505 = !{!287, !287, !125}
!506 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !507, file: !222, line: 1214)
!507 = !DISubprogram(name: "scalbn", scope: !219, file: !219, line: 276, type: !251, flags: DIFlagPrototyped, spFlags: 0)
!508 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !509, file: !222, line: 1215)
!509 = !DISubprogram(name: "scalbnf", scope: !219, file: !219, line: 276, type: !128, flags: DIFlagPrototyped, spFlags: 0)
!510 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !511, file: !222, line: 1216)
!511 = !DISubprogram(name: "scalbnl", scope: !219, file: !219, line: 276, type: !512, flags: DIFlagPrototyped, spFlags: 0)
!512 = !DISubroutineType(types: !513)
!513 = !{!287, !287, !28}
!514 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !515, file: !222, line: 1218)
!515 = !DISubprogram(name: "tgamma", scope: !219, file: !219, line: 235, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!516 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !517, file: !222, line: 1219)
!517 = !DISubprogram(name: "tgammaf", scope: !219, file: !219, line: 235, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!518 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !519, file: !222, line: 1220)
!519 = !DISubprogram(name: "tgammal", scope: !219, file: !219, line: 235, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!520 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !521, file: !222, line: 1222)
!521 = !DISubprogram(name: "trunc", scope: !219, file: !219, line: 302, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!522 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !523, file: !222, line: 1223)
!523 = !DISubprogram(name: "truncf", scope: !219, file: !219, line: 302, type: !31, flags: DIFlagPrototyped, spFlags: 0)
!524 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !525, file: !222, line: 1224)
!525 = !DISubprogram(name: "truncl", scope: !219, file: !219, line: 302, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!526 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !527, file: !529, line: 127)
!527 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !215, line: 62, baseType: !528)
!528 = !DICompositeType(tag: DW_TAG_structure_type, file: !215, line: 58, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!529 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/cstdlib", directory: "")
!530 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !531, file: !529, line: 128)
!531 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !215, line: 70, baseType: !532)
!532 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !215, line: 66, size: 128, flags: DIFlagTypePassByValue, elements: !533, identifier: "_ZTS6ldiv_t")
!533 = !{!534, !535}
!534 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !532, file: !215, line: 68, baseType: !125, size: 64)
!535 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !532, file: !215, line: 69, baseType: !125, size: 64, offset: 64)
!536 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !537, file: !529, line: 130)
!537 = !DISubprogram(name: "abort", scope: !215, file: !215, line: 588, type: !538, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!538 = !DISubroutineType(types: !539)
!539 = !{null}
!540 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !541, file: !529, line: 134)
!541 = !DISubprogram(name: "atexit", scope: !215, file: !215, line: 592, type: !542, flags: DIFlagPrototyped, spFlags: 0)
!542 = !DISubroutineType(types: !543)
!543 = !{!28, !544}
!544 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !538, size: 64)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !546, file: !529, line: 137)
!546 = !DISubprogram(name: "at_quick_exit", scope: !215, file: !215, line: 597, type: !542, flags: DIFlagPrototyped, spFlags: 0)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !548, file: !529, line: 140)
!548 = !DISubprogram(name: "atof", scope: !215, file: !215, line: 101, type: !165, flags: DIFlagPrototyped, spFlags: 0)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !550, file: !529, line: 141)
!550 = !DISubprogram(name: "atoi", scope: !215, file: !215, line: 104, type: !551, flags: DIFlagPrototyped, spFlags: 0)
!551 = !DISubroutineType(types: !552)
!552 = !{!28, !168}
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !554, file: !529, line: 142)
!554 = !DISubprogram(name: "atol", scope: !215, file: !215, line: 107, type: !555, flags: DIFlagPrototyped, spFlags: 0)
!555 = !DISubroutineType(types: !556)
!556 = !{!125, !168}
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !558, file: !529, line: 143)
!558 = !DISubprogram(name: "bsearch", scope: !215, file: !215, line: 817, type: !559, flags: DIFlagPrototyped, spFlags: 0)
!559 = !DISubroutineType(types: !560)
!560 = !{!20, !561, !561, !563, !563, !566}
!561 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !562, size: 64)
!562 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!563 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !564, line: 46, baseType: !565)
!564 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/stddef.h", directory: "/home/gkarlos")
!565 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!566 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !215, line: 805, baseType: !567)
!567 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !568, size: 64)
!568 = !DISubroutineType(types: !569)
!569 = !{!28, !561, !561}
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !571, file: !529, line: 144)
!571 = !DISubprogram(name: "calloc", scope: !215, file: !215, line: 541, type: !572, flags: DIFlagPrototyped, spFlags: 0)
!572 = !DISubroutineType(types: !573)
!573 = !{!20, !563, !563}
!574 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !575, file: !529, line: 145)
!575 = !DISubprogram(name: "div", scope: !215, file: !215, line: 849, type: !576, flags: DIFlagPrototyped, spFlags: 0)
!576 = !DISubroutineType(types: !577)
!577 = !{!527, !28, !28}
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !579, file: !529, line: 146)
!579 = !DISubprogram(name: "exit", scope: !215, file: !215, line: 614, type: !580, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!580 = !DISubroutineType(types: !581)
!581 = !{null, !28}
!582 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !583, file: !529, line: 147)
!583 = !DISubprogram(name: "free", scope: !215, file: !215, line: 563, type: !584, flags: DIFlagPrototyped, spFlags: 0)
!584 = !DISubroutineType(types: !585)
!585 = !{null, !20}
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !587, file: !529, line: 148)
!587 = !DISubprogram(name: "getenv", scope: !215, file: !215, line: 631, type: !588, flags: DIFlagPrototyped, spFlags: 0)
!588 = !DISubroutineType(types: !589)
!589 = !{!590, !168}
!590 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !170, size: 64)
!591 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !592, file: !529, line: 149)
!592 = !DISubprogram(name: "labs", scope: !215, file: !215, line: 838, type: !123, flags: DIFlagPrototyped, spFlags: 0)
!593 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !594, file: !529, line: 150)
!594 = !DISubprogram(name: "ldiv", scope: !215, file: !215, line: 851, type: !595, flags: DIFlagPrototyped, spFlags: 0)
!595 = !DISubroutineType(types: !596)
!596 = !{!531, !125, !125}
!597 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !598, file: !529, line: 151)
!598 = !DISubprogram(name: "malloc", scope: !215, file: !215, line: 539, type: !599, flags: DIFlagPrototyped, spFlags: 0)
!599 = !DISubroutineType(types: !600)
!600 = !{!20, !563}
!601 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !602, file: !529, line: 153)
!602 = !DISubprogram(name: "mblen", scope: !215, file: !215, line: 919, type: !603, flags: DIFlagPrototyped, spFlags: 0)
!603 = !DISubroutineType(types: !604)
!604 = !{!28, !168, !563}
!605 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !606, file: !529, line: 154)
!606 = !DISubprogram(name: "mbstowcs", scope: !215, file: !215, line: 930, type: !607, flags: DIFlagPrototyped, spFlags: 0)
!607 = !DISubroutineType(types: !608)
!608 = !{!563, !609, !612, !563}
!609 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !610)
!610 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !611, size: 64)
!611 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!612 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !168)
!613 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !614, file: !529, line: 155)
!614 = !DISubprogram(name: "mbtowc", scope: !215, file: !215, line: 922, type: !615, flags: DIFlagPrototyped, spFlags: 0)
!615 = !DISubroutineType(types: !616)
!616 = !{!28, !609, !612, !563}
!617 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !618, file: !529, line: 157)
!618 = !DISubprogram(name: "qsort", scope: !215, file: !215, line: 827, type: !619, flags: DIFlagPrototyped, spFlags: 0)
!619 = !DISubroutineType(types: !620)
!620 = !{null, !20, !563, !563, !566}
!621 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !622, file: !529, line: 160)
!622 = !DISubprogram(name: "quick_exit", scope: !215, file: !215, line: 620, type: !580, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!623 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !624, file: !529, line: 163)
!624 = !DISubprogram(name: "rand", scope: !215, file: !215, line: 453, type: !625, flags: DIFlagPrototyped, spFlags: 0)
!625 = !DISubroutineType(types: !626)
!626 = !{!28}
!627 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !628, file: !529, line: 164)
!628 = !DISubprogram(name: "realloc", scope: !215, file: !215, line: 549, type: !629, flags: DIFlagPrototyped, spFlags: 0)
!629 = !DISubroutineType(types: !630)
!630 = !{!20, !20, !563}
!631 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !632, file: !529, line: 165)
!632 = !DISubprogram(name: "srand", scope: !215, file: !215, line: 455, type: !633, flags: DIFlagPrototyped, spFlags: 0)
!633 = !DISubroutineType(types: !634)
!634 = !{null, !9}
!635 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !636, file: !529, line: 166)
!636 = !DISubprogram(name: "strtod", scope: !215, file: !215, line: 117, type: !637, flags: DIFlagPrototyped, spFlags: 0)
!637 = !DISubroutineType(types: !638)
!638 = !{!167, !612, !639}
!639 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !640)
!640 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !590, size: 64)
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !642, file: !529, line: 167)
!642 = !DISubprogram(name: "strtol", scope: !215, file: !215, line: 176, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!643 = !DISubroutineType(types: !644)
!644 = !{!125, !612, !639, !28}
!645 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !646, file: !529, line: 168)
!646 = !DISubprogram(name: "strtoul", scope: !215, file: !215, line: 180, type: !647, flags: DIFlagPrototyped, spFlags: 0)
!647 = !DISubroutineType(types: !648)
!648 = !{!565, !612, !639, !28}
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !650, file: !529, line: 169)
!650 = !DISubprogram(name: "system", scope: !215, file: !215, line: 781, type: !551, flags: DIFlagPrototyped, spFlags: 0)
!651 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !652, file: !529, line: 171)
!652 = !DISubprogram(name: "wcstombs", scope: !215, file: !215, line: 933, type: !653, flags: DIFlagPrototyped, spFlags: 0)
!653 = !DISubroutineType(types: !654)
!654 = !{!563, !655, !656, !563}
!655 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !590)
!656 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !657)
!657 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !658, size: 64)
!658 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !611)
!659 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !660, file: !529, line: 172)
!660 = !DISubprogram(name: "wctomb", scope: !215, file: !215, line: 926, type: !661, flags: DIFlagPrototyped, spFlags: 0)
!661 = !DISubroutineType(types: !662)
!662 = !{!28, !590, !611}
!663 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !665, file: !529, line: 200)
!664 = !DINamespace(name: "__gnu_cxx", scope: null)
!665 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !215, line: 80, baseType: !666)
!666 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !215, line: 76, size: 128, flags: DIFlagTypePassByValue, elements: !667, identifier: "_ZTS7lldiv_t")
!667 = !{!668, !669}
!668 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !666, file: !215, line: 78, baseType: !136, size: 64)
!669 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !666, file: !215, line: 79, baseType: !136, size: 64, offset: 64)
!670 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !671, file: !529, line: 206)
!671 = !DISubprogram(name: "_Exit", scope: !215, file: !215, line: 626, type: !580, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!672 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !673, file: !529, line: 210)
!673 = !DISubprogram(name: "llabs", scope: !215, file: !215, line: 841, type: !134, flags: DIFlagPrototyped, spFlags: 0)
!674 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !675, file: !529, line: 216)
!675 = !DISubprogram(name: "lldiv", scope: !215, file: !215, line: 855, type: !676, flags: DIFlagPrototyped, spFlags: 0)
!676 = !DISubroutineType(types: !677)
!677 = !{!665, !136, !136}
!678 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !679, file: !529, line: 227)
!679 = !DISubprogram(name: "atoll", scope: !215, file: !215, line: 112, type: !680, flags: DIFlagPrototyped, spFlags: 0)
!680 = !DISubroutineType(types: !681)
!681 = !{!136, !168}
!682 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !683, file: !529, line: 228)
!683 = !DISubprogram(name: "strtoll", scope: !215, file: !215, line: 200, type: !684, flags: DIFlagPrototyped, spFlags: 0)
!684 = !DISubroutineType(types: !685)
!685 = !{!136, !612, !639, !28}
!686 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !687, file: !529, line: 229)
!687 = !DISubprogram(name: "strtoull", scope: !215, file: !215, line: 205, type: !688, flags: DIFlagPrototyped, spFlags: 0)
!688 = !DISubroutineType(types: !689)
!689 = !{!690, !612, !639, !28}
!690 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!691 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !692, file: !529, line: 231)
!692 = !DISubprogram(name: "strtof", scope: !215, file: !215, line: 123, type: !693, flags: DIFlagPrototyped, spFlags: 0)
!693 = !DISubroutineType(types: !694)
!694 = !{!18, !612, !639}
!695 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !664, entity: !696, file: !529, line: 232)
!696 = !DISubprogram(name: "strtold", scope: !215, file: !215, line: 126, type: !697, flags: DIFlagPrototyped, spFlags: 0)
!697 = !DISubroutineType(types: !698)
!698 = !{!287, !612, !639}
!699 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !665, file: !529, line: 240)
!700 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !671, file: !529, line: 242)
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !673, file: !529, line: 244)
!702 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !703, file: !529, line: 245)
!703 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !664, file: !529, line: 213, type: !676, flags: DIFlagPrototyped, spFlags: 0)
!704 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !675, file: !529, line: 246)
!705 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !679, file: !529, line: 248)
!706 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !692, file: !529, line: 249)
!707 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !683, file: !529, line: 250)
!708 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !687, file: !529, line: 251)
!709 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !696, file: !529, line: 252)
!710 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !537, file: !711, line: 38)
!711 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/stdlib.h", directory: "")
!712 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !541, file: !711, line: 39)
!713 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !579, file: !711, line: 40)
!714 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !546, file: !711, line: 43)
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !622, file: !711, line: 46)
!716 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !527, file: !711, line: 51)
!717 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !531, file: !711, line: 52)
!718 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !719, file: !711, line: 54)
!719 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !23, file: !216, line: 78, type: !285, flags: DIFlagPrototyped, spFlags: 0)
!720 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !548, file: !711, line: 55)
!721 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !550, file: !711, line: 56)
!722 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !554, file: !711, line: 57)
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !558, file: !711, line: 58)
!724 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !571, file: !711, line: 59)
!725 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !703, file: !711, line: 60)
!726 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !583, file: !711, line: 61)
!727 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !587, file: !711, line: 62)
!728 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !592, file: !711, line: 63)
!729 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !594, file: !711, line: 64)
!730 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !598, file: !711, line: 65)
!731 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !602, file: !711, line: 67)
!732 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !606, file: !711, line: 68)
!733 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !614, file: !711, line: 69)
!734 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !618, file: !711, line: 71)
!735 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !624, file: !711, line: 72)
!736 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !628, file: !711, line: 73)
!737 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !632, file: !711, line: 74)
!738 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !636, file: !711, line: 75)
!739 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !642, file: !711, line: 76)
!740 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !646, file: !711, line: 77)
!741 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !650, file: !711, line: 78)
!742 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !652, file: !711, line: 80)
!743 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !660, file: !711, line: 81)
!744 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !745, file: !747, line: 414)
!745 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !746, file: !746, line: 1489, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!746 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_device_functions.h", directory: "/home/gkarlos")
!747 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_cmath.h", directory: "/home/gkarlos")
!748 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !749, file: !747, line: 415)
!749 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !746, file: !746, line: 1491, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!750 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !751, file: !747, line: 416)
!751 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !746, file: !746, line: 1493, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!752 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !753, file: !747, line: 417)
!753 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !746, file: !746, line: 1495, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!754 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !755, file: !747, line: 418)
!755 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !746, file: !746, line: 1498, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!756 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !757, file: !747, line: 419)
!757 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !746, file: !746, line: 1499, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!758 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !759, file: !747, line: 420)
!759 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !746, file: !746, line: 1501, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!760 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !761, file: !747, line: 421)
!761 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !746, file: !746, line: 1503, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!762 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !763, file: !747, line: 422)
!763 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !746, file: !746, line: 1505, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!764 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !765, file: !747, line: 423)
!765 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !746, file: !746, line: 1513, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!766 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !767, file: !747, line: 424)
!767 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !746, file: !746, line: 1517, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!768 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !769, file: !747, line: 425)
!769 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !746, file: !746, line: 1521, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !771, file: !747, line: 426)
!771 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !746, file: !746, line: 1530, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!772 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !773, file: !747, line: 427)
!773 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !746, file: !746, line: 1535, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!774 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !775, file: !747, line: 428)
!775 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !746, file: !746, line: 1542, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!776 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !777, file: !747, line: 429)
!777 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !746, file: !746, line: 1543, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!778 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !779, file: !747, line: 430)
!779 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !746, file: !746, line: 1545, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!780 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !781, file: !747, line: 431)
!781 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !746, file: !746, line: 1546, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!782 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !783, file: !747, line: 432)
!783 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !746, file: !746, line: 1548, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!784 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !785, file: !747, line: 433)
!785 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !746, file: !746, line: 1558, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!786 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !787, file: !747, line: 434)
!787 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !746, file: !746, line: 1562, type: !75, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!788 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !789, file: !747, line: 435)
!789 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !746, file: !746, line: 1566, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!790 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !791, file: !747, line: 436)
!791 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !746, file: !746, line: 1568, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!792 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !793, file: !747, line: 437)
!793 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !746, file: !746, line: 1570, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!794 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !795, file: !747, line: 438)
!795 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !746, file: !746, line: 1572, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!796 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !797, file: !747, line: 439)
!797 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !746, file: !746, line: 1574, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!798 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !799, file: !747, line: 440)
!799 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !746, file: !746, line: 1576, type: !85, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!800 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !801, file: !747, line: 441)
!801 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !746, file: !746, line: 1589, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!802 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !803, file: !747, line: 442)
!803 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !746, file: !746, line: 1591, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!804 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !805, file: !747, line: 443)
!805 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !746, file: !746, line: 1600, type: !139, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!806 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !807, file: !747, line: 444)
!807 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !746, file: !746, line: 1602, type: !139, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!808 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !809, file: !747, line: 445)
!809 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !746, file: !746, line: 1605, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!810 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !811, file: !747, line: 446)
!811 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !746, file: !746, line: 1607, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!812 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !813, file: !747, line: 447)
!813 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !746, file: !746, line: 1609, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!814 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !815, file: !747, line: 448)
!815 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !746, file: !746, line: 1613, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!816 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !817, file: !747, line: 449)
!817 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !746, file: !746, line: 1614, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!818 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !819, file: !747, line: 450)
!819 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !746, file: !746, line: 1619, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!820 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !821, file: !747, line: 451)
!821 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !746, file: !746, line: 1621, type: !153, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!822 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !823, file: !747, line: 452)
!823 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !746, file: !746, line: 1641, type: !161, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!824 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !825, file: !747, line: 453)
!825 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !746, file: !746, line: 1643, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!826 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !827, file: !747, line: 454)
!827 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !746, file: !746, line: 1647, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!828 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !829, file: !747, line: 455)
!829 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !746, file: !746, line: 1673, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!830 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !831, file: !747, line: 456)
!831 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !746, file: !746, line: 1681, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!832 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !833, file: !747, line: 457)
!833 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !746, file: !746, line: 1687, type: !185, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!834 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !835, file: !747, line: 458)
!835 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !746, file: !746, line: 1697, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!836 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !837, file: !747, line: 459)
!837 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !746, file: !746, line: 1717, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!838 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !839, file: !747, line: 462)
!839 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !746, file: !746, line: 1731, type: !193, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!840 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !841, file: !747, line: 464)
!841 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !746, file: !746, line: 1721, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!842 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !843, file: !747, line: 465)
!843 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !746, file: !746, line: 1752, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!844 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !845, file: !747, line: 466)
!845 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !746, file: !746, line: 1756, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!846 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !847, file: !747, line: 467)
!847 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !746, file: !746, line: 1760, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!848 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !849, file: !747, line: 468)
!849 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !746, file: !746, line: 1762, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!850 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !851, file: !747, line: 469)
!851 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !746, file: !746, line: 1764, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!852 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !853, file: !747, line: 470)
!853 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !746, file: !746, line: 1766, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!854 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !23, entity: !855, file: !747, line: 471)
!855 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !746, file: !746, line: 1768, type: !31, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!856 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !719, file: !857, line: 38)
!857 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/math.h", directory: "")
!858 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !4, entity: !859, file: !857, line: 54)
!859 = !DISubprogram(name: "modf", linkageName: "_ZSt4modfePe", scope: !23, file: !222, line: 380, type: !860, flags: DIFlagPrototyped, spFlags: 0)
!860 = !DISubroutineType(types: !861)
!861 = !{!287, !287, !862}
!862 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !287, size: 64)
!863 = !{!"clang version 10.0.0 "}
!864 = distinct !DISubprogram(name: "srad_cuda_1", linkageName: "_Z11srad_cuda_1PfS_S_S_S_S_iif", scope: !5, file: !5, line: 31, type: !865, scopeLine: 33, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!865 = !DISubroutineType(types: !866)
!866 = !{null, !17, !17, !17, !17, !17, !17, !28, !28, !18}
!867 = !{}
!868 = !DILocalVariable(name: "E_C", arg: 1, scope: !864, file: !5, line: 31, type: !17)
!869 = !DILocation(line: 31, column: 36, scope: !864)
!870 = !DILocalVariable(name: "W_C", arg: 2, scope: !864, file: !5, line: 31, type: !17)
!871 = !DILocation(line: 31, column: 48, scope: !864)
!872 = !DILocalVariable(name: "N_C", arg: 3, scope: !864, file: !5, line: 31, type: !17)
!873 = !DILocation(line: 31, column: 60, scope: !864)
!874 = !DILocalVariable(name: "S_C", arg: 4, scope: !864, file: !5, line: 31, type: !17)
!875 = !DILocation(line: 31, column: 72, scope: !864)
!876 = !DILocalVariable(name: "J_cuda", arg: 5, scope: !864, file: !5, line: 32, type: !17)
!877 = !DILocation(line: 32, column: 36, scope: !864)
!878 = !DILocalVariable(name: "C_cuda", arg: 6, scope: !864, file: !5, line: 32, type: !17)
!879 = !DILocation(line: 32, column: 51, scope: !864)
!880 = !DILocalVariable(name: "cols", arg: 7, scope: !864, file: !5, line: 32, type: !28)
!881 = !DILocation(line: 32, column: 63, scope: !864)
!882 = !DILocalVariable(name: "rows", arg: 8, scope: !864, file: !5, line: 32, type: !28)
!883 = !DILocation(line: 32, column: 73, scope: !864)
!884 = !DILocalVariable(name: "q0sqr", arg: 9, scope: !864, file: !5, line: 33, type: !18)
!885 = !DILocation(line: 33, column: 35, scope: !864)
!886 = !DILocation(line: 33, column: 42, scope: !864)
!887 = !DILocation(line: 171, column: 1, scope: !864)
!888 = distinct !DISubprogram(name: "srad_cuda_2", linkageName: "_Z11srad_cuda_2PfS_S_S_S_S_iiff", scope: !5, file: !5, line: 173, type: !889, scopeLine: 175, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!889 = !DISubroutineType(types: !890)
!890 = !{null, !17, !17, !17, !17, !17, !17, !28, !28, !18, !18}
!891 = !DILocalVariable(name: "E_C", arg: 1, scope: !888, file: !5, line: 173, type: !17)
!892 = !DILocation(line: 173, column: 36, scope: !888)
!893 = !DILocalVariable(name: "W_C", arg: 2, scope: !888, file: !5, line: 173, type: !17)
!894 = !DILocation(line: 173, column: 48, scope: !888)
!895 = !DILocalVariable(name: "N_C", arg: 3, scope: !888, file: !5, line: 173, type: !17)
!896 = !DILocation(line: 173, column: 60, scope: !888)
!897 = !DILocalVariable(name: "S_C", arg: 4, scope: !888, file: !5, line: 173, type: !17)
!898 = !DILocation(line: 173, column: 72, scope: !888)
!899 = !DILocalVariable(name: "J_cuda", arg: 5, scope: !888, file: !5, line: 174, type: !17)
!900 = !DILocation(line: 174, column: 36, scope: !888)
!901 = !DILocalVariable(name: "C_cuda", arg: 6, scope: !888, file: !5, line: 174, type: !17)
!902 = !DILocation(line: 174, column: 51, scope: !888)
!903 = !DILocalVariable(name: "cols", arg: 7, scope: !888, file: !5, line: 174, type: !28)
!904 = !DILocation(line: 174, column: 63, scope: !888)
!905 = !DILocalVariable(name: "rows", arg: 8, scope: !888, file: !5, line: 174, type: !28)
!906 = !DILocation(line: 174, column: 73, scope: !888)
!907 = !DILocalVariable(name: "lambda", arg: 9, scope: !888, file: !5, line: 175, type: !18)
!908 = !DILocation(line: 175, column: 35, scope: !888)
!909 = !DILocalVariable(name: "q0sqr", arg: 10, scope: !888, file: !5, line: 175, type: !18)
!910 = !DILocation(line: 175, column: 49, scope: !888)
!911 = !DILocation(line: 175, column: 56, scope: !888)
!912 = !DILocation(line: 260, column: 1, scope: !888)
!913 = distinct !DISubprogram(name: "usage", linkageName: "_Z5usageiPPc", scope: !5, file: !5, line: 264, type: !914, scopeLine: 264, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!914 = !DISubroutineType(types: !915)
!915 = !{null, !28, !640}
!916 = !DILocalVariable(name: "argc", arg: 1, scope: !913, file: !5, line: 264, type: !28)
!917 = !DILocation(line: 264, column: 16, scope: !913)
!918 = !DILocalVariable(name: "argv", arg: 2, scope: !913, file: !5, line: 264, type: !640)
!919 = !DILocation(line: 264, column: 29, scope: !913)
!920 = !DILocation(line: 265, column: 11, scope: !913)
!921 = !DILocation(line: 267, column: 11, scope: !913)
!922 = !DILocation(line: 265, column: 3, scope: !913)
!923 = !DILocation(line: 268, column: 11, scope: !913)
!924 = !DILocation(line: 268, column: 3, scope: !913)
!925 = !DILocation(line: 269, column: 11, scope: !913)
!926 = !DILocation(line: 269, column: 3, scope: !913)
!927 = !DILocation(line: 270, column: 11, scope: !913)
!928 = !DILocation(line: 270, column: 3, scope: !913)
!929 = !DILocation(line: 271, column: 11, scope: !913)
!930 = !DILocation(line: 271, column: 3, scope: !913)
!931 = !DILocation(line: 272, column: 11, scope: !913)
!932 = !DILocation(line: 272, column: 3, scope: !913)
!933 = !DILocation(line: 273, column: 11, scope: !913)
!934 = !DILocation(line: 273, column: 3, scope: !913)
!935 = !DILocation(line: 274, column: 11, scope: !913)
!936 = !DILocation(line: 274, column: 3, scope: !913)
!937 = !DILocation(line: 275, column: 11, scope: !913)
!938 = !DILocation(line: 275, column: 3, scope: !913)
!939 = !DILocation(line: 277, column: 3, scope: !913)
!940 = distinct !DISubprogram(name: "main", scope: !5, file: !5, line: 282, type: !941, scopeLine: 282, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!941 = !DISubroutineType(types: !942)
!942 = !{!28, !28, !640}
!943 = !DILocalVariable(name: "argc", arg: 1, scope: !940, file: !5, line: 282, type: !28)
!944 = !DILocation(line: 282, column: 14, scope: !940)
!945 = !DILocalVariable(name: "argv", arg: 2, scope: !940, file: !5, line: 282, type: !640)
!946 = !DILocation(line: 282, column: 27, scope: !940)
!947 = !DILocation(line: 283, column: 3, scope: !940)
!948 = !DILocation(line: 284, column: 11, scope: !940)
!949 = !DILocation(line: 284, column: 17, scope: !940)
!950 = !DILocation(line: 284, column: 3, scope: !940)
!951 = !DILocation(line: 286, column: 3, scope: !940)
!952 = distinct !DISubprogram(name: "runTest", linkageName: "_Z7runTestiPPc", scope: !5, file: !5, line: 289, type: !914, scopeLine: 289, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!953 = !DILocalVariable(name: "argc", arg: 1, scope: !952, file: !5, line: 289, type: !28)
!954 = !DILocation(line: 289, column: 18, scope: !952)
!955 = !DILocalVariable(name: "argv", arg: 2, scope: !952, file: !5, line: 289, type: !640)
!956 = !DILocation(line: 289, column: 31, scope: !952)
!957 = !DILocalVariable(name: "rows", scope: !952, file: !5, line: 290, type: !28)
!958 = !DILocation(line: 290, column: 7, scope: !952)
!959 = !DILocalVariable(name: "cols", scope: !952, file: !5, line: 290, type: !28)
!960 = !DILocation(line: 290, column: 13, scope: !952)
!961 = !DILocalVariable(name: "size_I", scope: !952, file: !5, line: 290, type: !28)
!962 = !DILocation(line: 290, column: 19, scope: !952)
!963 = !DILocalVariable(name: "size_R", scope: !952, file: !5, line: 290, type: !28)
!964 = !DILocation(line: 290, column: 27, scope: !952)
!965 = !DILocalVariable(name: "niter", scope: !952, file: !5, line: 290, type: !28)
!966 = !DILocation(line: 290, column: 35, scope: !952)
!967 = !DILocalVariable(name: "iter", scope: !952, file: !5, line: 290, type: !28)
!968 = !DILocation(line: 290, column: 47, scope: !952)
!969 = !DILocalVariable(name: "I", scope: !952, file: !5, line: 291, type: !17)
!970 = !DILocation(line: 291, column: 10, scope: !952)
!971 = !DILocalVariable(name: "J", scope: !952, file: !5, line: 291, type: !17)
!972 = !DILocation(line: 291, column: 14, scope: !952)
!973 = !DILocalVariable(name: "lambda", scope: !952, file: !5, line: 291, type: !18)
!974 = !DILocation(line: 291, column: 17, scope: !952)
!975 = !DILocalVariable(name: "q0sqr", scope: !952, file: !5, line: 291, type: !18)
!976 = !DILocation(line: 291, column: 25, scope: !952)
!977 = !DILocalVariable(name: "sum", scope: !952, file: !5, line: 291, type: !18)
!978 = !DILocation(line: 291, column: 32, scope: !952)
!979 = !DILocalVariable(name: "sum2", scope: !952, file: !5, line: 291, type: !18)
!980 = !DILocation(line: 291, column: 37, scope: !952)
!981 = !DILocalVariable(name: "tmp", scope: !952, file: !5, line: 291, type: !18)
!982 = !DILocation(line: 291, column: 43, scope: !952)
!983 = !DILocalVariable(name: "meanROI", scope: !952, file: !5, line: 291, type: !18)
!984 = !DILocation(line: 291, column: 48, scope: !952)
!985 = !DILocalVariable(name: "varROI", scope: !952, file: !5, line: 291, type: !18)
!986 = !DILocation(line: 291, column: 57, scope: !952)
!987 = !DILocalVariable(name: "J_cuda", scope: !952, file: !5, line: 302, type: !17)
!988 = !DILocation(line: 302, column: 10, scope: !952)
!989 = !DILocalVariable(name: "C_cuda", scope: !952, file: !5, line: 303, type: !17)
!990 = !DILocation(line: 303, column: 10, scope: !952)
!991 = !DILocalVariable(name: "E_C", scope: !952, file: !5, line: 304, type: !17)
!992 = !DILocation(line: 304, column: 10, scope: !952)
!993 = !DILocalVariable(name: "W_C", scope: !952, file: !5, line: 304, type: !17)
!994 = !DILocation(line: 304, column: 16, scope: !952)
!995 = !DILocalVariable(name: "N_C", scope: !952, file: !5, line: 304, type: !17)
!996 = !DILocation(line: 304, column: 22, scope: !952)
!997 = !DILocalVariable(name: "S_C", scope: !952, file: !5, line: 304, type: !17)
!998 = !DILocation(line: 304, column: 28, scope: !952)
!999 = !DILocalVariable(name: "r1", scope: !952, file: !5, line: 308, type: !9)
!1000 = !DILocation(line: 308, column: 16, scope: !952)
!1001 = !DILocalVariable(name: "r2", scope: !952, file: !5, line: 308, type: !9)
!1002 = !DILocation(line: 308, column: 20, scope: !952)
!1003 = !DILocalVariable(name: "c1", scope: !952, file: !5, line: 308, type: !9)
!1004 = !DILocation(line: 308, column: 24, scope: !952)
!1005 = !DILocalVariable(name: "c2", scope: !952, file: !5, line: 308, type: !9)
!1006 = !DILocation(line: 308, column: 28, scope: !952)
!1007 = !DILocalVariable(name: "c", scope: !952, file: !5, line: 309, type: !17)
!1008 = !DILocation(line: 309, column: 10, scope: !952)
!1009 = !DILocation(line: 311, column: 7, scope: !1010)
!1010 = distinct !DILexicalBlock(scope: !952, file: !5, line: 311, column: 7)
!1011 = !DILocation(line: 311, column: 12, scope: !1010)
!1012 = !DILocation(line: 311, column: 7, scope: !952)
!1013 = !DILocation(line: 312, column: 17, scope: !1014)
!1014 = distinct !DILexicalBlock(scope: !1010, file: !5, line: 311, column: 18)
!1015 = !DILocation(line: 312, column: 12, scope: !1014)
!1016 = !DILocation(line: 312, column: 10, scope: !1014)
!1017 = !DILocation(line: 313, column: 17, scope: !1014)
!1018 = !DILocation(line: 313, column: 12, scope: !1014)
!1019 = !DILocation(line: 313, column: 10, scope: !1014)
!1020 = !DILocation(line: 314, column: 10, scope: !1021)
!1021 = distinct !DILexicalBlock(scope: !1014, file: !5, line: 314, column: 9)
!1022 = !DILocation(line: 314, column: 15, scope: !1021)
!1023 = !DILocation(line: 314, column: 20, scope: !1021)
!1024 = !DILocation(line: 314, column: 26, scope: !1021)
!1025 = !DILocation(line: 314, column: 30, scope: !1021)
!1026 = !DILocation(line: 314, column: 35, scope: !1021)
!1027 = !DILocation(line: 314, column: 40, scope: !1021)
!1028 = !DILocation(line: 314, column: 9, scope: !1014)
!1029 = !DILocation(line: 315, column: 15, scope: !1030)
!1030 = distinct !DILexicalBlock(scope: !1021, file: !5, line: 314, column: 47)
!1031 = !DILocation(line: 315, column: 7, scope: !1030)
!1032 = !DILocation(line: 316, column: 7, scope: !1030)
!1033 = !DILocation(line: 318, column: 15, scope: !1014)
!1034 = !DILocation(line: 318, column: 10, scope: !1014)
!1035 = !DILocation(line: 318, column: 8, scope: !1014)
!1036 = !DILocation(line: 319, column: 15, scope: !1014)
!1037 = !DILocation(line: 319, column: 10, scope: !1014)
!1038 = !DILocation(line: 319, column: 8, scope: !1014)
!1039 = !DILocation(line: 320, column: 15, scope: !1014)
!1040 = !DILocation(line: 320, column: 10, scope: !1014)
!1041 = !DILocation(line: 320, column: 8, scope: !1014)
!1042 = !DILocation(line: 321, column: 15, scope: !1014)
!1043 = !DILocation(line: 321, column: 10, scope: !1014)
!1044 = !DILocation(line: 321, column: 8, scope: !1014)
!1045 = !DILocation(line: 322, column: 19, scope: !1014)
!1046 = !DILocation(line: 322, column: 14, scope: !1014)
!1047 = !DILocation(line: 322, column: 12, scope: !1014)
!1048 = !DILocation(line: 323, column: 18, scope: !1014)
!1049 = !DILocation(line: 323, column: 13, scope: !1014)
!1050 = !DILocation(line: 323, column: 11, scope: !1014)
!1051 = !DILocation(line: 325, column: 3, scope: !1014)
!1052 = !DILocation(line: 326, column: 11, scope: !1053)
!1053 = distinct !DILexicalBlock(scope: !1010, file: !5, line: 325, column: 10)
!1054 = !DILocation(line: 326, column: 17, scope: !1053)
!1055 = !DILocation(line: 326, column: 5, scope: !1053)
!1056 = !DILocation(line: 329, column: 12, scope: !952)
!1057 = !DILocation(line: 329, column: 19, scope: !952)
!1058 = !DILocation(line: 329, column: 17, scope: !952)
!1059 = !DILocation(line: 329, column: 10, scope: !952)
!1060 = !DILocation(line: 330, column: 13, scope: !952)
!1061 = !DILocation(line: 330, column: 18, scope: !952)
!1062 = !DILocation(line: 330, column: 16, scope: !952)
!1063 = !DILocation(line: 330, column: 21, scope: !952)
!1064 = !DILocation(line: 330, column: 29, scope: !952)
!1065 = !DILocation(line: 330, column: 34, scope: !952)
!1066 = !DILocation(line: 330, column: 32, scope: !952)
!1067 = !DILocation(line: 330, column: 37, scope: !952)
!1068 = !DILocation(line: 330, column: 26, scope: !952)
!1069 = !DILocation(line: 330, column: 10, scope: !952)
!1070 = !DILocation(line: 332, column: 23, scope: !952)
!1071 = !DILocation(line: 332, column: 30, scope: !952)
!1072 = !DILocation(line: 332, column: 16, scope: !952)
!1073 = !DILocation(line: 332, column: 7, scope: !952)
!1074 = !DILocation(line: 332, column: 5, scope: !952)
!1075 = !DILocation(line: 333, column: 23, scope: !952)
!1076 = !DILocation(line: 333, column: 30, scope: !952)
!1077 = !DILocation(line: 333, column: 16, scope: !952)
!1078 = !DILocation(line: 333, column: 7, scope: !952)
!1079 = !DILocation(line: 333, column: 5, scope: !952)
!1080 = !DILocation(line: 334, column: 39, scope: !952)
!1081 = !DILocation(line: 334, column: 37, scope: !952)
!1082 = !DILocation(line: 334, column: 16, scope: !952)
!1083 = !DILocation(line: 334, column: 7, scope: !952)
!1084 = !DILocation(line: 334, column: 5, scope: !952)
!1085 = !DILocation(line: 366, column: 14, scope: !952)
!1086 = !DILocation(line: 366, column: 48, scope: !952)
!1087 = !DILocation(line: 366, column: 46, scope: !952)
!1088 = !DILocation(line: 366, column: 3, scope: !952)
!1089 = !DILocation(line: 367, column: 14, scope: !952)
!1090 = !DILocation(line: 367, column: 48, scope: !952)
!1091 = !DILocation(line: 367, column: 46, scope: !952)
!1092 = !DILocation(line: 367, column: 3, scope: !952)
!1093 = !DILocation(line: 368, column: 14, scope: !952)
!1094 = !DILocation(line: 368, column: 45, scope: !952)
!1095 = !DILocation(line: 368, column: 43, scope: !952)
!1096 = !DILocation(line: 368, column: 3, scope: !952)
!1097 = !DILocation(line: 369, column: 14, scope: !952)
!1098 = !DILocation(line: 369, column: 45, scope: !952)
!1099 = !DILocation(line: 369, column: 43, scope: !952)
!1100 = !DILocation(line: 369, column: 3, scope: !952)
!1101 = !DILocation(line: 370, column: 14, scope: !952)
!1102 = !DILocation(line: 370, column: 45, scope: !952)
!1103 = !DILocation(line: 370, column: 43, scope: !952)
!1104 = !DILocation(line: 370, column: 3, scope: !952)
!1105 = !DILocation(line: 371, column: 14, scope: !952)
!1106 = !DILocation(line: 371, column: 45, scope: !952)
!1107 = !DILocation(line: 371, column: 43, scope: !952)
!1108 = !DILocation(line: 371, column: 3, scope: !952)
!1109 = !DILocation(line: 375, column: 3, scope: !952)
!1110 = !DILocation(line: 377, column: 17, scope: !952)
!1111 = !DILocation(line: 377, column: 20, scope: !952)
!1112 = !DILocation(line: 377, column: 26, scope: !952)
!1113 = !DILocation(line: 377, column: 3, scope: !952)
!1114 = !DILocalVariable(name: "k", scope: !1115, file: !5, line: 379, type: !28)
!1115 = distinct !DILexicalBlock(scope: !952, file: !5, line: 379, column: 3)
!1116 = !DILocation(line: 379, column: 12, scope: !1115)
!1117 = !DILocation(line: 379, column: 8, scope: !1115)
!1118 = !DILocation(line: 379, column: 19, scope: !1119)
!1119 = distinct !DILexicalBlock(scope: !1115, file: !5, line: 379, column: 3)
!1120 = !DILocation(line: 379, column: 23, scope: !1119)
!1121 = !DILocation(line: 379, column: 21, scope: !1119)
!1122 = !DILocation(line: 379, column: 3, scope: !1115)
!1123 = !DILocation(line: 380, column: 23, scope: !1124)
!1124 = distinct !DILexicalBlock(scope: !1119, file: !5, line: 379, column: 36)
!1125 = !DILocation(line: 380, column: 25, scope: !1124)
!1126 = !DILocation(line: 380, column: 19, scope: !1124)
!1127 = !DILocation(line: 380, column: 5, scope: !1124)
!1128 = !DILocation(line: 380, column: 7, scope: !1124)
!1129 = !DILocation(line: 380, column: 10, scope: !1124)
!1130 = !DILocation(line: 381, column: 3, scope: !1124)
!1131 = !DILocation(line: 379, column: 32, scope: !1119)
!1132 = !DILocation(line: 379, column: 3, scope: !1119)
!1133 = distinct !{!1133, !1122, !1134}
!1134 = !DILocation(line: 381, column: 3, scope: !1115)
!1135 = !DILocation(line: 382, column: 3, scope: !952)
!1136 = !DILocation(line: 383, column: 13, scope: !1137)
!1137 = distinct !DILexicalBlock(scope: !952, file: !5, line: 383, column: 3)
!1138 = !DILocation(line: 383, column: 8, scope: !1137)
!1139 = !DILocation(line: 383, column: 18, scope: !1140)
!1140 = distinct !DILexicalBlock(scope: !1137, file: !5, line: 383, column: 3)
!1141 = !DILocation(line: 383, column: 25, scope: !1140)
!1142 = !DILocation(line: 383, column: 23, scope: !1140)
!1143 = !DILocation(line: 383, column: 3, scope: !1137)
!1144 = !DILocation(line: 384, column: 9, scope: !1145)
!1145 = distinct !DILexicalBlock(scope: !1140, file: !5, line: 383, column: 40)
!1146 = !DILocation(line: 385, column: 10, scope: !1145)
!1147 = !DILocalVariable(name: "i", scope: !1148, file: !5, line: 386, type: !28)
!1148 = distinct !DILexicalBlock(scope: !1145, file: !5, line: 386, column: 5)
!1149 = !DILocation(line: 386, column: 14, scope: !1148)
!1150 = !DILocation(line: 386, column: 18, scope: !1148)
!1151 = !DILocation(line: 386, column: 10, scope: !1148)
!1152 = !DILocation(line: 386, column: 22, scope: !1153)
!1153 = distinct !DILexicalBlock(scope: !1148, file: !5, line: 386, column: 5)
!1154 = !DILocation(line: 386, column: 27, scope: !1153)
!1155 = !DILocation(line: 386, column: 24, scope: !1153)
!1156 = !DILocation(line: 386, column: 5, scope: !1148)
!1157 = !DILocalVariable(name: "j", scope: !1158, file: !5, line: 387, type: !28)
!1158 = distinct !DILexicalBlock(scope: !1159, file: !5, line: 387, column: 7)
!1159 = distinct !DILexicalBlock(scope: !1153, file: !5, line: 386, column: 36)
!1160 = !DILocation(line: 387, column: 16, scope: !1158)
!1161 = !DILocation(line: 387, column: 20, scope: !1158)
!1162 = !DILocation(line: 387, column: 12, scope: !1158)
!1163 = !DILocation(line: 387, column: 24, scope: !1164)
!1164 = distinct !DILexicalBlock(scope: !1158, file: !5, line: 387, column: 7)
!1165 = !DILocation(line: 387, column: 29, scope: !1164)
!1166 = !DILocation(line: 387, column: 26, scope: !1164)
!1167 = !DILocation(line: 387, column: 7, scope: !1158)
!1168 = !DILocation(line: 388, column: 15, scope: !1169)
!1169 = distinct !DILexicalBlock(scope: !1164, file: !5, line: 387, column: 38)
!1170 = !DILocation(line: 388, column: 17, scope: !1169)
!1171 = !DILocation(line: 388, column: 21, scope: !1169)
!1172 = !DILocation(line: 388, column: 19, scope: !1169)
!1173 = !DILocation(line: 388, column: 28, scope: !1169)
!1174 = !DILocation(line: 388, column: 26, scope: !1169)
!1175 = !DILocation(line: 388, column: 13, scope: !1169)
!1176 = !DILocation(line: 389, column: 16, scope: !1169)
!1177 = !DILocation(line: 389, column: 13, scope: !1169)
!1178 = !DILocation(line: 390, column: 17, scope: !1169)
!1179 = !DILocation(line: 390, column: 23, scope: !1169)
!1180 = !DILocation(line: 390, column: 21, scope: !1169)
!1181 = !DILocation(line: 390, column: 14, scope: !1169)
!1182 = !DILocation(line: 391, column: 7, scope: !1169)
!1183 = !DILocation(line: 387, column: 34, scope: !1164)
!1184 = !DILocation(line: 387, column: 7, scope: !1164)
!1185 = distinct !{!1185, !1167, !1186}
!1186 = !DILocation(line: 391, column: 7, scope: !1158)
!1187 = !DILocation(line: 392, column: 5, scope: !1159)
!1188 = !DILocation(line: 386, column: 32, scope: !1153)
!1189 = !DILocation(line: 386, column: 5, scope: !1153)
!1190 = distinct !{!1190, !1156, !1191}
!1191 = !DILocation(line: 392, column: 5, scope: !1148)
!1192 = !DILocation(line: 393, column: 15, scope: !1145)
!1193 = !DILocation(line: 393, column: 21, scope: !1145)
!1194 = !DILocation(line: 393, column: 19, scope: !1145)
!1195 = !DILocation(line: 393, column: 13, scope: !1145)
!1196 = !DILocation(line: 394, column: 15, scope: !1145)
!1197 = !DILocation(line: 394, column: 22, scope: !1145)
!1198 = !DILocation(line: 394, column: 20, scope: !1145)
!1199 = !DILocation(line: 394, column: 32, scope: !1145)
!1200 = !DILocation(line: 394, column: 42, scope: !1145)
!1201 = !DILocation(line: 394, column: 40, scope: !1145)
!1202 = !DILocation(line: 394, column: 30, scope: !1145)
!1203 = !DILocation(line: 394, column: 12, scope: !1145)
!1204 = !DILocation(line: 395, column: 13, scope: !1145)
!1205 = !DILocation(line: 395, column: 23, scope: !1145)
!1206 = !DILocation(line: 395, column: 33, scope: !1145)
!1207 = !DILocation(line: 395, column: 31, scope: !1145)
!1208 = !DILocation(line: 395, column: 20, scope: !1145)
!1209 = !DILocation(line: 395, column: 11, scope: !1145)
!1210 = !DILocalVariable(name: "block_x", scope: !1145, file: !5, line: 457, type: !28)
!1211 = !DILocation(line: 457, column: 9, scope: !1145)
!1212 = !DILocation(line: 457, column: 19, scope: !1145)
!1213 = !DILocation(line: 457, column: 24, scope: !1145)
!1214 = !DILocalVariable(name: "block_y", scope: !1145, file: !5, line: 458, type: !28)
!1215 = !DILocation(line: 458, column: 9, scope: !1145)
!1216 = !DILocation(line: 458, column: 19, scope: !1145)
!1217 = !DILocation(line: 458, column: 24, scope: !1145)
!1218 = !DILocalVariable(name: "dimBlock", scope: !1145, file: !5, line: 460, type: !1219)
!1219 = !DIDerivedType(tag: DW_TAG_typedef, name: "dim3", file: !1220, line: 430, baseType: !1221)
!1220 = !DIFile(filename: "s/cuda/10/include/vector_types.h", directory: "/home/gkarlos")
!1221 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1220, line: 416, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !1222, identifier: "_ZTS4dim3")
!1222 = !{!1223, !1224, !1225, !1226, !1230, !1239}
!1223 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1221, file: !1220, line: 418, baseType: !9, size: 32)
!1224 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1221, file: !1220, line: 418, baseType: !9, size: 32, offset: 32)
!1225 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1221, file: !1220, line: 418, baseType: !9, size: 32, offset: 64)
!1226 = !DISubprogram(name: "dim3", scope: !1221, file: !1220, line: 421, type: !1227, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!1227 = !DISubroutineType(types: !1228)
!1228 = !{null, !1229, !9, !9, !9}
!1229 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1221, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1230 = !DISubprogram(name: "dim3", scope: !1221, file: !1220, line: 425, type: !1231, scopeLine: 425, flags: DIFlagPrototyped, spFlags: 0)
!1231 = !DISubroutineType(types: !1232)
!1232 = !{null, !1229, !1233}
!1233 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1220, line: 382, baseType: !1234)
!1234 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1220, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !1235, identifier: "_ZTS5uint3")
!1235 = !{!1236, !1237, !1238}
!1236 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1234, file: !1220, line: 192, baseType: !9, size: 32)
!1237 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1234, file: !1220, line: 192, baseType: !9, size: 32, offset: 32)
!1238 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1234, file: !1220, line: 192, baseType: !9, size: 32, offset: 64)
!1239 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1221, file: !1220, line: 426, type: !1240, scopeLine: 426, flags: DIFlagPrototyped, spFlags: 0)
!1240 = !DISubroutineType(types: !1241)
!1241 = !{!1233, !1229}
!1242 = !DILocation(line: 460, column: 10, scope: !1145)
!1243 = !DILocalVariable(name: "dimGrid", scope: !1145, file: !5, line: 461, type: !1219)
!1244 = !DILocation(line: 461, column: 10, scope: !1145)
!1245 = !DILocation(line: 461, column: 18, scope: !1145)
!1246 = !DILocation(line: 461, column: 27, scope: !1145)
!1247 = !DILocation(line: 464, column: 16, scope: !1145)
!1248 = !DILocation(line: 464, column: 24, scope: !1145)
!1249 = !DILocation(line: 464, column: 43, scope: !1145)
!1250 = !DILocation(line: 464, column: 41, scope: !1145)
!1251 = !DILocation(line: 464, column: 5, scope: !1145)
!1252 = !DILocation(line: 467, column: 19, scope: !1145)
!1253 = !DILocation(line: 467, column: 28, scope: !1145)
!1254 = !DILocation(line: 467, column: 16, scope: !1145)
!1255 = !DILocation(line: 467, column: 5, scope: !1145)
!1256 = !DILocation(line: 467, column: 40, scope: !1145)
!1257 = !DILocation(line: 467, column: 45, scope: !1145)
!1258 = !DILocation(line: 467, column: 50, scope: !1145)
!1259 = !DILocation(line: 467, column: 55, scope: !1145)
!1260 = !DILocation(line: 467, column: 60, scope: !1145)
!1261 = !DILocation(line: 467, column: 68, scope: !1145)
!1262 = !DILocation(line: 467, column: 76, scope: !1145)
!1263 = !DILocation(line: 468, column: 40, scope: !1145)
!1264 = !DILocation(line: 468, column: 46, scope: !1145)
!1265 = !DILocation(line: 469, column: 19, scope: !1145)
!1266 = !DILocation(line: 469, column: 28, scope: !1145)
!1267 = !DILocation(line: 469, column: 16, scope: !1145)
!1268 = !DILocation(line: 469, column: 5, scope: !1145)
!1269 = !DILocation(line: 469, column: 40, scope: !1145)
!1270 = !DILocation(line: 469, column: 45, scope: !1145)
!1271 = !DILocation(line: 469, column: 50, scope: !1145)
!1272 = !DILocation(line: 469, column: 55, scope: !1145)
!1273 = !DILocation(line: 469, column: 60, scope: !1145)
!1274 = !DILocation(line: 469, column: 68, scope: !1145)
!1275 = !DILocation(line: 469, column: 76, scope: !1145)
!1276 = !DILocation(line: 470, column: 40, scope: !1145)
!1277 = !DILocation(line: 470, column: 46, scope: !1145)
!1278 = !DILocation(line: 470, column: 54, scope: !1145)
!1279 = !DILocation(line: 473, column: 16, scope: !1145)
!1280 = !DILocation(line: 473, column: 19, scope: !1145)
!1281 = !DILocation(line: 473, column: 43, scope: !1145)
!1282 = !DILocation(line: 473, column: 41, scope: !1145)
!1283 = !DILocation(line: 473, column: 5, scope: !1145)
!1284 = !DILocation(line: 476, column: 3, scope: !1145)
!1285 = !DILocation(line: 383, column: 36, scope: !1140)
!1286 = !DILocation(line: 383, column: 3, scope: !1140)
!1287 = distinct !{!1287, !1143, !1288}
!1288 = !DILocation(line: 476, column: 3, scope: !1137)
!1289 = !DILocation(line: 478, column: 3, scope: !952)
!1290 = !DILocation(line: 491, column: 3, scope: !952)
!1291 = !DILocation(line: 493, column: 8, scope: !952)
!1292 = !DILocation(line: 493, column: 3, scope: !952)
!1293 = !DILocation(line: 494, column: 8, scope: !952)
!1294 = !DILocation(line: 494, column: 3, scope: !952)
!1295 = !DILocation(line: 506, column: 12, scope: !952)
!1296 = !DILocation(line: 506, column: 3, scope: !952)
!1297 = !DILocation(line: 507, column: 12, scope: !952)
!1298 = !DILocation(line: 507, column: 3, scope: !952)
!1299 = !DILocation(line: 508, column: 12, scope: !952)
!1300 = !DILocation(line: 508, column: 3, scope: !952)
!1301 = !DILocation(line: 509, column: 12, scope: !952)
!1302 = !DILocation(line: 509, column: 3, scope: !952)
!1303 = !DILocation(line: 510, column: 12, scope: !952)
!1304 = !DILocation(line: 510, column: 3, scope: !952)
!1305 = !DILocation(line: 511, column: 12, scope: !952)
!1306 = !DILocation(line: 511, column: 3, scope: !952)
!1307 = !DILocation(line: 513, column: 8, scope: !952)
!1308 = !DILocation(line: 513, column: 3, scope: !952)
!1309 = !DILocation(line: 514, column: 1, scope: !952)
!1310 = distinct !DISubprogram(name: "random_matrix", linkageName: "_Z13random_matrixPfii", scope: !5, file: !5, line: 516, type: !1311, scopeLine: 516, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!1311 = !DISubroutineType(types: !1312)
!1312 = !{null, !17, !28, !28}
!1313 = !DILocalVariable(name: "I", arg: 1, scope: !1310, file: !5, line: 516, type: !17)
!1314 = !DILocation(line: 516, column: 27, scope: !1310)
!1315 = !DILocalVariable(name: "rows", arg: 2, scope: !1310, file: !5, line: 516, type: !28)
!1316 = !DILocation(line: 516, column: 34, scope: !1310)
!1317 = !DILocalVariable(name: "cols", arg: 3, scope: !1310, file: !5, line: 516, type: !28)
!1318 = !DILocation(line: 516, column: 44, scope: !1310)
!1319 = !DILocation(line: 518, column: 3, scope: !1310)
!1320 = !DILocalVariable(name: "i", scope: !1321, file: !5, line: 520, type: !28)
!1321 = distinct !DILexicalBlock(scope: !1310, file: !5, line: 520, column: 3)
!1322 = !DILocation(line: 520, column: 12, scope: !1321)
!1323 = !DILocation(line: 520, column: 8, scope: !1321)
!1324 = !DILocation(line: 520, column: 19, scope: !1325)
!1325 = distinct !DILexicalBlock(scope: !1321, file: !5, line: 520, column: 3)
!1326 = !DILocation(line: 520, column: 23, scope: !1325)
!1327 = !DILocation(line: 520, column: 21, scope: !1325)
!1328 = !DILocation(line: 520, column: 3, scope: !1321)
!1329 = !DILocalVariable(name: "j", scope: !1330, file: !5, line: 521, type: !28)
!1330 = distinct !DILexicalBlock(scope: !1331, file: !5, line: 521, column: 5)
!1331 = distinct !DILexicalBlock(scope: !1325, file: !5, line: 520, column: 34)
!1332 = !DILocation(line: 521, column: 14, scope: !1330)
!1333 = !DILocation(line: 521, column: 10, scope: !1330)
!1334 = !DILocation(line: 521, column: 21, scope: !1335)
!1335 = distinct !DILexicalBlock(scope: !1330, file: !5, line: 521, column: 5)
!1336 = !DILocation(line: 521, column: 25, scope: !1335)
!1337 = !DILocation(line: 521, column: 23, scope: !1335)
!1338 = !DILocation(line: 521, column: 5, scope: !1330)
!1339 = !DILocation(line: 522, column: 25, scope: !1340)
!1340 = distinct !DILexicalBlock(scope: !1335, file: !5, line: 521, column: 36)
!1341 = !DILocation(line: 522, column: 32, scope: !1340)
!1342 = !DILocation(line: 522, column: 7, scope: !1340)
!1343 = !DILocation(line: 522, column: 9, scope: !1340)
!1344 = !DILocation(line: 522, column: 13, scope: !1340)
!1345 = !DILocation(line: 522, column: 11, scope: !1340)
!1346 = !DILocation(line: 522, column: 20, scope: !1340)
!1347 = !DILocation(line: 522, column: 18, scope: !1340)
!1348 = !DILocation(line: 522, column: 23, scope: !1340)
!1349 = !DILocation(line: 523, column: 5, scope: !1340)
!1350 = !DILocation(line: 521, column: 32, scope: !1335)
!1351 = !DILocation(line: 521, column: 5, scope: !1335)
!1352 = distinct !{!1352, !1338, !1353}
!1353 = !DILocation(line: 523, column: 5, scope: !1330)
!1354 = !DILocation(line: 524, column: 3, scope: !1331)
!1355 = !DILocation(line: 520, column: 30, scope: !1325)
!1356 = !DILocation(line: 520, column: 3, scope: !1325)
!1357 = distinct !{!1357, !1328, !1358}
!1358 = !DILocation(line: 524, column: 3, scope: !1321)
!1359 = !DILocation(line: 525, column: 1, scope: !1310)
!1360 = distinct !DISubprogram(name: "exp", linkageName: "_ZSt3expf", scope: !23, file: !222, line: 222, type: !31, scopeLine: 223, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !867)
!1361 = !DILocalVariable(name: "__x", arg: 1, scope: !1360, file: !222, line: 222, type: !18)
!1362 = !DILocation(line: 222, column: 13, scope: !1360)
!1363 = !DILocation(line: 223, column: 27, scope: !1360)
!1364 = !DILocation(line: 223, column: 12, scope: !1360)
!1365 = !DILocation(line: 223, column: 5, scope: !1360)
!1366 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1221, file: !1220, line: 421, type: !1227, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, declaration: !1226, retainedNodes: !867)
!1367 = !DILocalVariable(name: "this", arg: 1, scope: !1366, type: !1368, flags: DIFlagArtificial | DIFlagObjectPointer)
!1368 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1221, size: 64)
!1369 = !DILocation(line: 0, scope: !1366)
!1370 = !DILocalVariable(name: "vx", arg: 2, scope: !1366, file: !1220, line: 421, type: !9)
!1371 = !DILocation(line: 421, column: 53, scope: !1366)
!1372 = !DILocalVariable(name: "vy", arg: 3, scope: !1366, file: !1220, line: 421, type: !9)
!1373 = !DILocation(line: 421, column: 74, scope: !1366)
!1374 = !DILocalVariable(name: "vz", arg: 4, scope: !1366, file: !1220, line: 421, type: !9)
!1375 = !DILocation(line: 421, column: 95, scope: !1366)
!1376 = !DILocation(line: 421, column: 105, scope: !1366)
!1377 = !DILocation(line: 421, column: 107, scope: !1366)
!1378 = !DILocation(line: 421, column: 112, scope: !1366)
!1379 = !DILocation(line: 421, column: 114, scope: !1366)
!1380 = !DILocation(line: 421, column: 119, scope: !1366)
!1381 = !DILocation(line: 421, column: 121, scope: !1366)
!1382 = !DILocation(line: 421, column: 126, scope: !1366)

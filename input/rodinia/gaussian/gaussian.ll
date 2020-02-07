; ModuleID = 'gaussian.cu'
source_filename = "gaussian.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.cudaDeviceProp = type { [256 x i8], %struct.CUuuid_st, [8 x i8], i32, i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32 }
%struct.CUuuid_st = type { [16 x i8] }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@Size = dso_local global i32 0, align 4, !dbg !0
@a = dso_local global float* null, align 8, !dbg !130
@b = dso_local global float* null, align 8, !dbg !132
@finalVec = dso_local global float* null, align 8, !dbg !134
@m = dso_local global float* null, align 8, !dbg !136
@fp = dso_local global %struct._IO_FILE* null, align 8, !dbg !138
@totalKernelTime = dso_local global i32 0, align 4, !dbg !145
@.str = private unnamed_addr constant [34 x i8] c"Usage: gaussian matrix.txt [-q]\0A\0A\00", align 1
@.str.1 = private unnamed_addr constant [62 x i8] c"-q (quiet) suppresses printing the matrix and result values.\0A\00", align 1
@.str.2 = private unnamed_addr constant [68 x i8] c"The first line of the file contains the dimension of the matrix, n.\00", align 1
@.str.3 = private unnamed_addr constant [43 x i8] c"The second line of the file is a newline.\0A\00", align 1
@.str.4 = private unnamed_addr constant [64 x i8] c"The next n lines contain n tab separated values for the matrix.\00", align 1
@.str.5 = private unnamed_addr constant [41 x i8] c"The next line of the file is a newline.\0A\00", align 1
@.str.6 = private unnamed_addr constant [70 x i8] c"The next line of the file is a 1xn vector with tab separated values.\0A\00", align 1
@.str.7 = private unnamed_addr constant [52 x i8] c"The next line of the file is a newline. (optional)\0A\00", align 1
@.str.8 = private unnamed_addr constant [69 x i8] c"The final line of the file is the pre-computed solution. (optional)\0A\00", align 1
@.str.9 = private unnamed_addr constant [23 x i8] c"Example: matrix4.txt:\0A\00", align 1
@.str.10 = private unnamed_addr constant [3 x i8] c"4\0A\00", align 1
@.str.11 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.12 = private unnamed_addr constant [19 x i8] c"-0.6\09-0.5\090.7\090.3\0A\00", align 1
@.str.13 = private unnamed_addr constant [19 x i8] c"-0.3\09-0.9\090.3\090.7\0A\00", align 1
@.str.14 = private unnamed_addr constant [21 x i8] c"-0.4\09-0.5\09-0.3\09-0.8\0A\00", align 1
@.str.15 = private unnamed_addr constant [18 x i8] c"0.0\09-0.1\090.2\090.9\0A\00", align 1
@.str.16 = private unnamed_addr constant [24 x i8] c"-0.85\09-0.68\090.24\09-0.53\0A\00", align 1
@.str.17 = private unnamed_addr constant [19 x i8] c"0.7\090.0\09-0.4\09-0.5\0A\00", align 1
@.str.18 = private unnamed_addr constant [3 x i8] c"-q\00", align 1
@.str.19 = private unnamed_addr constant [15 x i8] c"Matrix m is: \0A\00", align 1
@.str.20 = private unnamed_addr constant [15 x i8] c"Matrix a is: \0A\00", align 1
@.str.21 = private unnamed_addr constant [14 x i8] c"Array b is: \0A\00", align 1
@.str.22 = private unnamed_addr constant [25 x i8] c"The final solution is: \0A\00", align 1
@.str.23 = private unnamed_addr constant [49 x i8] c"\0ATime total (including memory transfers)\09%f sec\0A\00", align 1
@.str.24 = private unnamed_addr constant [31 x i8] c"Time for CUDA kernels:\09%f sec\0A\00", align 1
@.str.25 = private unnamed_addr constant [23 x i8] c"Total Device found: %d\00", align 1
@.str.26 = private unnamed_addr constant [22 x i8] c"\0ADevice Name \09\09 - %s \00", align 1
@.str.27 = private unnamed_addr constant [40 x i8] c"\0A**************************************\00", align 1
@.str.28 = private unnamed_addr constant [33 x i8] c"\0ATotal Global Memory\09\09\09 - %lu KB\00", align 1
@.str.29 = private unnamed_addr constant [46 x i8] c"\0AShared memory available per block \09 - %lu KB\00", align 1
@.str.30 = private unnamed_addr constant [45 x i8] c"\0ANumber of registers per thread block \09 - %d\00", align 1
@.str.31 = private unnamed_addr constant [31 x i8] c"\0AWarp size in threads \09\09\09 - %d\00", align 1
@.str.32 = private unnamed_addr constant [31 x i8] c"\0AMemory Pitch \09\09\09\09 - %zu bytes\00", align 1
@.str.33 = private unnamed_addr constant [35 x i8] c"\0AMaximum threads per block \09\09 - %d\00", align 1
@.str.34 = private unnamed_addr constant [47 x i8] c"\0AMaximum Thread Dimension (block) \09 - %d %d %d\00", align 1
@.str.35 = private unnamed_addr constant [46 x i8] c"\0AMaximum Thread Dimension (grid) \09 - %d %d %d\00", align 1
@.str.36 = private unnamed_addr constant [39 x i8] c"\0ATotal constant memory \09\09\09 - %zu bytes\00", align 1
@.str.37 = private unnamed_addr constant [23 x i8] c"\0ACUDA ver \09\09\09\09 - %d.%d\00", align 1
@.str.38 = private unnamed_addr constant [26 x i8] c"\0AClock rate \09\09\09\09 - %d KHz\00", align 1
@.str.39 = private unnamed_addr constant [35 x i8] c"\0ATexture Alignment \09\09\09 - %zu bytes\00", align 1
@.str.40 = private unnamed_addr constant [26 x i8] c"\0ADevice Overlap \09\09\09\09 - %s\00", align 1
@.str.41 = private unnamed_addr constant [8 x i8] c"Allowed\00", align 1
@.str.42 = private unnamed_addr constant [12 x i8] c"Not Allowed\00", align 1
@.str.43 = private unnamed_addr constant [38 x i8] c"\0ANumber of Multi processors \09\09 - %d\0A\0A\00", align 1
@.str.44 = private unnamed_addr constant [4 x i8] c"\0A%s\00", align 1
@.str.45 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.46 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.47 = private unnamed_addr constant [5 x i8] c"Fan2\00", align 1
@.str.48 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.49 = private unnamed_addr constant [7 x i8] c"%8.2f \00", align 1
@.str.50 = private unnamed_addr constant [6 x i8] c"%.2f \00", align 1
@.str.51 = private unnamed_addr constant [3 x i8] c"\0A\0A\00", align 1
@stderr = external dso_local global %struct._IO_FILE*, align 8
@.str.52 = private unnamed_addr constant [21 x i8] c"Cuda error: %s: %s.\0A\00", align 1

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main(i32 %0, i8** %1) #0 !dbg !732 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  %7 = alloca %struct.timeval, align 8
  %8 = alloca %struct.timeval, align 8
  %9 = alloca i32, align 4
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !736, metadata !DIExpression()), !dbg !737
  store i8** %1, i8*** %5, align 8
  call void @llvm.dbg.declare(metadata i8*** %5, metadata !738, metadata !DIExpression()), !dbg !739
  call void @llvm.dbg.declare(metadata i32* %6, metadata !740, metadata !DIExpression()), !dbg !741
  store i32 1, i32* %6, align 4, !dbg !741
  %10 = load i32, i32* %4, align 4, !dbg !742
  %11 = icmp slt i32 %10, 2, !dbg !744
  br i1 %11, label %12, label %33, !dbg !745

12:                                               ; preds = %2
  %13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str, i64 0, i64 0)), !dbg !746
  %14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([62 x i8], [62 x i8]* @.str.1, i64 0, i64 0)), !dbg !748
  %15 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([68 x i8], [68 x i8]* @.str.2, i64 0, i64 0)), !dbg !749
  %16 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.3, i64 0, i64 0)), !dbg !750
  %17 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.4, i64 0, i64 0)), !dbg !751
  %18 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.5, i64 0, i64 0)), !dbg !752
  %19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([70 x i8], [70 x i8]* @.str.6, i64 0, i64 0)), !dbg !753
  %20 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.7, i64 0, i64 0)), !dbg !754
  %21 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str.8, i64 0, i64 0)), !dbg !755
  %22 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.9, i64 0, i64 0)), !dbg !756
  %23 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.10, i64 0, i64 0)), !dbg !757
  %24 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0)), !dbg !758
  %25 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.12, i64 0, i64 0)), !dbg !759
  %26 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.13, i64 0, i64 0)), !dbg !760
  %27 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.14, i64 0, i64 0)), !dbg !761
  %28 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.15, i64 0, i64 0)), !dbg !762
  %29 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0)), !dbg !763
  %30 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.16, i64 0, i64 0)), !dbg !764
  %31 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0)), !dbg !765
  %32 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.17, i64 0, i64 0)), !dbg !766
  call void @exit(i32 0) #9, !dbg !767
  unreachable, !dbg !767

33:                                               ; preds = %2
  %34 = load i8**, i8*** %5, align 8, !dbg !768
  %35 = getelementptr inbounds i8*, i8** %34, i64 1, !dbg !768
  %36 = load i8*, i8** %35, align 8, !dbg !768
  call void @_Z15InitProblemOncePc(i8* %36), !dbg !769
  %37 = load i32, i32* %4, align 4, !dbg !770
  %38 = icmp sgt i32 %37, 2, !dbg !772
  br i1 %38, label %39, label %47, !dbg !773

39:                                               ; preds = %33
  %40 = load i8**, i8*** %5, align 8, !dbg !774
  %41 = getelementptr inbounds i8*, i8** %40, i64 2, !dbg !774
  %42 = load i8*, i8** %41, align 8, !dbg !774
  %43 = call i32 @strcmp(i8* %42, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.18, i64 0, i64 0)) #10, !dbg !777
  %44 = icmp ne i32 %43, 0, !dbg !777
  br i1 %44, label %46, label %45, !dbg !778

45:                                               ; preds = %39
  store i32 0, i32* %6, align 4, !dbg !779
  br label %46, !dbg !780

46:                                               ; preds = %45, %39
  br label %47, !dbg !781

47:                                               ; preds = %46, %33
  call void @_Z10InitPerRunv(), !dbg !782
  call void @llvm.dbg.declare(metadata %struct.timeval* %7, metadata !783, metadata !DIExpression()), !dbg !792
  %48 = call i32 @gettimeofday(%struct.timeval* %7, %struct.timezone* null) #11, !dbg !793
  call void @_Z10ForwardSubv(), !dbg !794
  call void @llvm.dbg.declare(metadata %struct.timeval* %8, metadata !795, metadata !DIExpression()), !dbg !796
  %49 = call i32 @gettimeofday(%struct.timeval* %8, %struct.timezone* null) #11, !dbg !797
  call void @llvm.dbg.declare(metadata i32* %9, metadata !798, metadata !DIExpression()), !dbg !799
  %50 = getelementptr inbounds %struct.timeval, %struct.timeval* %8, i32 0, i32 0, !dbg !800
  %51 = load i64, i64* %50, align 8, !dbg !800
  %52 = mul nsw i64 %51, 1000000, !dbg !801
  %53 = getelementptr inbounds %struct.timeval, %struct.timeval* %8, i32 0, i32 1, !dbg !802
  %54 = load i64, i64* %53, align 8, !dbg !802
  %55 = add nsw i64 %52, %54, !dbg !803
  %56 = getelementptr inbounds %struct.timeval, %struct.timeval* %7, i32 0, i32 0, !dbg !804
  %57 = load i64, i64* %56, align 8, !dbg !804
  %58 = mul nsw i64 %57, 1000000, !dbg !805
  %59 = getelementptr inbounds %struct.timeval, %struct.timeval* %7, i32 0, i32 1, !dbg !806
  %60 = load i64, i64* %59, align 8, !dbg !806
  %61 = add nsw i64 %58, %60, !dbg !807
  %62 = sub nsw i64 %55, %61, !dbg !808
  %63 = trunc i64 %62 to i32, !dbg !809
  store i32 %63, i32* %9, align 4, !dbg !799
  %64 = load i32, i32* %6, align 4, !dbg !810
  %65 = icmp ne i32 %64, 0, !dbg !810
  br i1 %65, label %66, label %78, !dbg !812

66:                                               ; preds = %47
  %67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.19, i64 0, i64 0)), !dbg !813
  %68 = load float*, float** @m, align 8, !dbg !815
  %69 = load i32, i32* @Size, align 4, !dbg !816
  %70 = load i32, i32* @Size, align 4, !dbg !817
  call void @_Z8PrintMatPfii(float* %68, i32 %69, i32 %70), !dbg !818
  %71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.20, i64 0, i64 0)), !dbg !819
  %72 = load float*, float** @a, align 8, !dbg !820
  %73 = load i32, i32* @Size, align 4, !dbg !821
  %74 = load i32, i32* @Size, align 4, !dbg !822
  call void @_Z8PrintMatPfii(float* %72, i32 %73, i32 %74), !dbg !823
  %75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.21, i64 0, i64 0)), !dbg !824
  %76 = load float*, float** @b, align 8, !dbg !825
  %77 = load i32, i32* @Size, align 4, !dbg !826
  call void @_Z8PrintAryPfi(float* %76, i32 %77), !dbg !827
  br label %78, !dbg !828

78:                                               ; preds = %66, %47
  call void @_Z7BackSubv(), !dbg !829
  %79 = load i32, i32* %6, align 4, !dbg !830
  %80 = icmp ne i32 %79, 0, !dbg !830
  br i1 %80, label %81, label %85, !dbg !832

81:                                               ; preds = %78
  %82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.22, i64 0, i64 0)), !dbg !833
  %83 = load float*, float** @finalVec, align 8, !dbg !835
  %84 = load i32, i32* @Size, align 4, !dbg !836
  call void @_Z8PrintAryPfi(float* %83, i32 %84), !dbg !837
  br label %85, !dbg !838

85:                                               ; preds = %81, %78
  %86 = load i32, i32* %9, align 4, !dbg !839
  %87 = uitofp i32 %86 to double, !dbg !839
  %88 = fmul contract double %87, 0x3EB0C6F7A0B5ED8D, !dbg !840
  %89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.23, i64 0, i64 0), double %88), !dbg !841
  %90 = load i32, i32* @totalKernelTime, align 4, !dbg !842
  %91 = uitofp i32 %90 to double, !dbg !842
  %92 = fmul contract double %91, 0x3EB0C6F7A0B5ED8D, !dbg !843
  %93 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.24, i64 0, i64 0), double %92), !dbg !844
  %94 = load float*, float** @m, align 8, !dbg !845
  %95 = bitcast float* %94 to i8*, !dbg !845
  call void @free(i8* %95) #11, !dbg !846
  %96 = load float*, float** @a, align 8, !dbg !847
  %97 = bitcast float* %96 to i8*, !dbg !847
  call void @free(i8* %97) #11, !dbg !848
  %98 = load float*, float** @b, align 8, !dbg !849
  %99 = bitcast float* %98 to i8*, !dbg !849
  call void @free(i8* %99) #11, !dbg !850
  %100 = load i32, i32* %3, align 4, !dbg !851
  ret i32 %100, !dbg !851
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #3

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z15InitProblemOncePc(i8* %0) #4 !dbg !852 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  call void @llvm.dbg.declare(metadata i8** %2, metadata !855, metadata !DIExpression()), !dbg !856
  %3 = load i8*, i8** %2, align 8, !dbg !857
  %4 = call %struct._IO_FILE* @fopen(i8* %3, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0)), !dbg !858
  store %struct._IO_FILE* %4, %struct._IO_FILE** @fp, align 8, !dbg !859
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8, !dbg !860
  %6 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.46, i64 0, i64 0), i32* @Size), !dbg !861
  %7 = load i32, i32* @Size, align 4, !dbg !862
  %8 = load i32, i32* @Size, align 4, !dbg !863
  %9 = mul nsw i32 %7, %8, !dbg !864
  %10 = sext i32 %9 to i64, !dbg !862
  %11 = mul i64 %10, 4, !dbg !865
  %12 = call noalias i8* @malloc(i64 %11) #11, !dbg !866
  %13 = bitcast i8* %12 to float*, !dbg !867
  store float* %13, float** @a, align 8, !dbg !868
  %14 = load float*, float** @a, align 8, !dbg !869
  %15 = load i32, i32* @Size, align 4, !dbg !870
  %16 = load i32, i32* @Size, align 4, !dbg !871
  call void @_Z7InitMatPfii(float* %14, i32 %15, i32 %16), !dbg !872
  %17 = load i32, i32* @Size, align 4, !dbg !873
  %18 = sext i32 %17 to i64, !dbg !873
  %19 = mul i64 %18, 4, !dbg !874
  %20 = call noalias i8* @malloc(i64 %19) #11, !dbg !875
  %21 = bitcast i8* %20 to float*, !dbg !876
  store float* %21, float** @b, align 8, !dbg !877
  %22 = load float*, float** @b, align 8, !dbg !878
  %23 = load i32, i32* @Size, align 4, !dbg !879
  call void @_Z7InitAryPfi(float* %22, i32 %23), !dbg !880
  %24 = load i32, i32* @Size, align 4, !dbg !881
  %25 = load i32, i32* @Size, align 4, !dbg !882
  %26 = mul nsw i32 %24, %25, !dbg !883
  %27 = sext i32 %26 to i64, !dbg !881
  %28 = mul i64 %27, 4, !dbg !884
  %29 = call noalias i8* @malloc(i64 %28) #11, !dbg !885
  %30 = bitcast i8* %29 to float*, !dbg !886
  store float* %30, float** @m, align 8, !dbg !887
  ret void, !dbg !888
}

; Function Attrs: nounwind readonly
declare dso_local i32 @strcmp(i8*, i8*) #5

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z10InitPerRunv() #6 !dbg !889 {
  %1 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %1, metadata !890, metadata !DIExpression()), !dbg !891
  store i32 0, i32* %1, align 4, !dbg !892
  br label %2, !dbg !894

2:                                                ; preds = %13, %0
  %3 = load i32, i32* %1, align 4, !dbg !895
  %4 = load i32, i32* @Size, align 4, !dbg !897
  %5 = load i32, i32* @Size, align 4, !dbg !898
  %6 = mul nsw i32 %4, %5, !dbg !899
  %7 = icmp slt i32 %3, %6, !dbg !900
  br i1 %7, label %8, label %16, !dbg !901

8:                                                ; preds = %2
  %9 = load float*, float** @m, align 8, !dbg !902
  %10 = load i32, i32* %1, align 4, !dbg !903
  %11 = sext i32 %10 to i64, !dbg !904
  %12 = getelementptr inbounds float, float* %9, i64 %11, !dbg !904
  store float 0.000000e+00, float* %12, align 4, !dbg !905
  br label %13, !dbg !906

13:                                               ; preds = %8
  %14 = load i32, i32* %1, align 4, !dbg !907
  %15 = add nsw i32 %14, 1, !dbg !907
  store i32 %15, i32* %1, align 4, !dbg !907
  br label %2, !dbg !908, !llvm.loop !909

16:                                               ; preds = %2
  ret void, !dbg !911
}

; Function Attrs: nounwind
declare dso_local i32 @gettimeofday(%struct.timeval*, %struct.timezone*) #7

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z10ForwardSubv() #4 !dbg !912 {
  %1 = alloca i32, align 4
  %2 = alloca float*, align 8
  %3 = alloca float*, align 8
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca %struct.dim3, align 4
  %8 = alloca %struct.dim3, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca %struct.dim3, align 4
  %12 = alloca %struct.dim3, align 4
  %13 = alloca %struct.timeval, align 8
  %14 = alloca %struct.dim3, align 4
  %15 = alloca %struct.dim3, align 4
  %16 = alloca { i64, i32 }, align 4
  %17 = alloca { i64, i32 }, align 4
  %18 = alloca %struct.dim3, align 4
  %19 = alloca %struct.dim3, align 4
  %20 = alloca { i64, i32 }, align 4
  %21 = alloca { i64, i32 }, align 4
  %22 = alloca %struct.timeval, align 8
  call void @llvm.dbg.declare(metadata i32* %1, metadata !913, metadata !DIExpression()), !dbg !914
  call void @llvm.dbg.declare(metadata float** %2, metadata !915, metadata !DIExpression()), !dbg !916
  call void @llvm.dbg.declare(metadata float** %3, metadata !917, metadata !DIExpression()), !dbg !918
  call void @llvm.dbg.declare(metadata float** %4, metadata !919, metadata !DIExpression()), !dbg !920
  %23 = bitcast float** %2 to i8**, !dbg !921
  %24 = load i32, i32* @Size, align 4, !dbg !922
  %25 = load i32, i32* @Size, align 4, !dbg !923
  %26 = mul nsw i32 %24, %25, !dbg !924
  %27 = sext i32 %26 to i64, !dbg !922
  %28 = mul i64 %27, 4, !dbg !925
  %29 = call i32 @cudaMalloc(i8** %23, i64 %28), !dbg !926
  %30 = bitcast float** %3 to i8**, !dbg !927
  %31 = load i32, i32* @Size, align 4, !dbg !928
  %32 = load i32, i32* @Size, align 4, !dbg !929
  %33 = mul nsw i32 %31, %32, !dbg !930
  %34 = sext i32 %33 to i64, !dbg !928
  %35 = mul i64 %34, 4, !dbg !931
  %36 = call i32 @cudaMalloc(i8** %30, i64 %35), !dbg !932
  %37 = bitcast float** %4 to i8**, !dbg !933
  %38 = load i32, i32* @Size, align 4, !dbg !934
  %39 = sext i32 %38 to i64, !dbg !934
  %40 = mul i64 %39, 4, !dbg !935
  %41 = call i32 @cudaMalloc(i8** %37, i64 %40), !dbg !936
  %42 = load float*, float** %2, align 8, !dbg !937
  %43 = bitcast float* %42 to i8*, !dbg !937
  %44 = load float*, float** @m, align 8, !dbg !938
  %45 = bitcast float* %44 to i8*, !dbg !938
  %46 = load i32, i32* @Size, align 4, !dbg !939
  %47 = load i32, i32* @Size, align 4, !dbg !940
  %48 = mul nsw i32 %46, %47, !dbg !941
  %49 = sext i32 %48 to i64, !dbg !939
  %50 = mul i64 %49, 4, !dbg !942
  %51 = call i32 @cudaMemcpy(i8* %43, i8* %45, i64 %50, i32 1), !dbg !943
  %52 = load float*, float** %3, align 8, !dbg !944
  %53 = bitcast float* %52 to i8*, !dbg !944
  %54 = load float*, float** @a, align 8, !dbg !945
  %55 = bitcast float* %54 to i8*, !dbg !945
  %56 = load i32, i32* @Size, align 4, !dbg !946
  %57 = load i32, i32* @Size, align 4, !dbg !947
  %58 = mul nsw i32 %56, %57, !dbg !948
  %59 = sext i32 %58 to i64, !dbg !946
  %60 = mul i64 %59, 4, !dbg !949
  %61 = call i32 @cudaMemcpy(i8* %53, i8* %55, i64 %60, i32 1), !dbg !950
  %62 = load float*, float** %4, align 8, !dbg !951
  %63 = bitcast float* %62 to i8*, !dbg !951
  %64 = load float*, float** @b, align 8, !dbg !952
  %65 = bitcast float* %64 to i8*, !dbg !952
  %66 = load i32, i32* @Size, align 4, !dbg !953
  %67 = sext i32 %66 to i64, !dbg !953
  %68 = mul i64 %67, 4, !dbg !954
  %69 = call i32 @cudaMemcpy(i8* %63, i8* %65, i64 %68, i32 1), !dbg !955
  call void @llvm.dbg.declare(metadata i32* %5, metadata !956, metadata !DIExpression()), !dbg !957
  call void @llvm.dbg.declare(metadata i32* %6, metadata !958, metadata !DIExpression()), !dbg !959
  store i32 512, i32* %5, align 4, !dbg !960
  %70 = load i32, i32* @Size, align 4, !dbg !961
  %71 = load i32, i32* %5, align 4, !dbg !962
  %72 = sdiv i32 %70, %71, !dbg !963
  %73 = load i32, i32* @Size, align 4, !dbg !964
  %74 = load i32, i32* %5, align 4, !dbg !965
  %75 = srem i32 %73, %74, !dbg !966
  %76 = icmp ne i32 %75, 0, !dbg !967
  %77 = xor i1 %76, true, !dbg !968
  %78 = zext i1 %77 to i64, !dbg !968
  %79 = select i1 %77, i32 0, i32 1, !dbg !968
  %80 = add nsw i32 %72, %79, !dbg !969
  store i32 %80, i32* %6, align 4, !dbg !970
  call void @llvm.dbg.declare(metadata %struct.dim3* %7, metadata !971, metadata !DIExpression()), !dbg !995
  %81 = load i32, i32* %5, align 4, !dbg !996
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %7, i32 %81, i32 1, i32 1), !dbg !995
  call void @llvm.dbg.declare(metadata %struct.dim3* %8, metadata !997, metadata !DIExpression()), !dbg !998
  %82 = load i32, i32* %6, align 4, !dbg !999
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %8, i32 %82, i32 1, i32 1), !dbg !998
  call void @llvm.dbg.declare(metadata i32* %9, metadata !1000, metadata !DIExpression()), !dbg !1001
  call void @llvm.dbg.declare(metadata i32* %10, metadata !1002, metadata !DIExpression()), !dbg !1003
  store i32 4, i32* %9, align 4, !dbg !1004
  %83 = load i32, i32* @Size, align 4, !dbg !1005
  %84 = load i32, i32* %9, align 4, !dbg !1006
  %85 = sdiv i32 %83, %84, !dbg !1007
  %86 = load i32, i32* @Size, align 4, !dbg !1008
  %87 = load i32, i32* %9, align 4, !dbg !1009
  %88 = srem i32 %86, %87, !dbg !1010
  %89 = icmp ne i32 %88, 0, !dbg !1008
  %90 = zext i1 %89 to i64, !dbg !1008
  %91 = select i1 %89, i32 0, i32 1, !dbg !1008
  %92 = icmp ne i32 %91, 0, !dbg !1011
  %93 = xor i1 %92, true, !dbg !1012
  %94 = zext i1 %93 to i32, !dbg !1013
  %95 = add nsw i32 %85, %94, !dbg !1014
  store i32 %95, i32* %10, align 4, !dbg !1015
  call void @llvm.dbg.declare(metadata %struct.dim3* %11, metadata !1016, metadata !DIExpression()), !dbg !1017
  %96 = load i32, i32* %9, align 4, !dbg !1018
  %97 = load i32, i32* %9, align 4, !dbg !1019
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %11, i32 %96, i32 %97, i32 1), !dbg !1017
  call void @llvm.dbg.declare(metadata %struct.dim3* %12, metadata !1020, metadata !DIExpression()), !dbg !1021
  %98 = load i32, i32* %10, align 4, !dbg !1022
  %99 = load i32, i32* %10, align 4, !dbg !1023
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %12, i32 %98, i32 %99, i32 1), !dbg !1021
  call void @llvm.dbg.declare(metadata %struct.timeval* %13, metadata !1024, metadata !DIExpression()), !dbg !1025
  %100 = call i32 @gettimeofday(%struct.timeval* %13, %struct.timezone* null) #11, !dbg !1026
  store i32 0, i32* %1, align 4, !dbg !1027
  br label %101, !dbg !1029

101:                                              ; preds = %161, %0
  %102 = load i32, i32* %1, align 4, !dbg !1030
  %103 = load i32, i32* @Size, align 4, !dbg !1032
  %104 = sub nsw i32 %103, 1, !dbg !1033
  %105 = icmp slt i32 %102, %104, !dbg !1034
  br i1 %105, label %106, label %164, !dbg !1035

106:                                              ; preds = %101
  %107 = bitcast %struct.dim3* %14 to i8*, !dbg !1036
  %108 = bitcast %struct.dim3* %8 to i8*, !dbg !1036
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %107, i8* align 4 %108, i64 12, i1 false), !dbg !1036
  %109 = bitcast %struct.dim3* %15 to i8*, !dbg !1038
  %110 = bitcast %struct.dim3* %7 to i8*, !dbg !1038
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %109, i8* align 4 %110, i64 12, i1 false), !dbg !1038
  %111 = bitcast { i64, i32 }* %16 to i8*, !dbg !1039
  %112 = bitcast %struct.dim3* %14 to i8*, !dbg !1039
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %111, i8* align 4 %112, i64 12, i1 false), !dbg !1039
  %113 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 0, !dbg !1039
  %114 = load i64, i64* %113, align 4, !dbg !1039
  %115 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 1, !dbg !1039
  %116 = load i32, i32* %115, align 4, !dbg !1039
  %117 = bitcast { i64, i32 }* %17 to i8*, !dbg !1039
  %118 = bitcast %struct.dim3* %15 to i8*, !dbg !1039
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %117, i8* align 4 %118, i64 12, i1 false), !dbg !1039
  %119 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %17, i32 0, i32 0, !dbg !1039
  %120 = load i64, i64* %119, align 4, !dbg !1039
  %121 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %17, i32 0, i32 1, !dbg !1039
  %122 = load i32, i32* %121, align 4, !dbg !1039
  %123 = call i32 @__cudaPushCallConfiguration(i64 %114, i32 %116, i64 %120, i32 %122, i64 0, i8* null), !dbg !1039
  %124 = icmp ne i32 %123, 0, !dbg !1039
  br i1 %124, label %130, label %125, !dbg !1040

125:                                              ; preds = %106
  %126 = load float*, float** %2, align 8, !dbg !1041
  %127 = load float*, float** %3, align 8, !dbg !1042
  %128 = load i32, i32* @Size, align 4, !dbg !1043
  %129 = load i32, i32* %1, align 4, !dbg !1044
  call void @_Z4Fan1PfS_ii(float* %126, float* %127, i32 %128, i32 %129), !dbg !1040
  br label %130, !dbg !1040

130:                                              ; preds = %125, %106
  %131 = call i32 @cudaThreadSynchronize(), !dbg !1045
  %132 = bitcast %struct.dim3* %18 to i8*, !dbg !1046
  %133 = bitcast %struct.dim3* %12 to i8*, !dbg !1046
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %132, i8* align 4 %133, i64 12, i1 false), !dbg !1046
  %134 = bitcast %struct.dim3* %19 to i8*, !dbg !1047
  %135 = bitcast %struct.dim3* %11 to i8*, !dbg !1047
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %134, i8* align 4 %135, i64 12, i1 false), !dbg !1047
  %136 = bitcast { i64, i32 }* %20 to i8*, !dbg !1048
  %137 = bitcast %struct.dim3* %18 to i8*, !dbg !1048
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %136, i8* align 4 %137, i64 12, i1 false), !dbg !1048
  %138 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 0, !dbg !1048
  %139 = load i64, i64* %138, align 4, !dbg !1048
  %140 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 1, !dbg !1048
  %141 = load i32, i32* %140, align 4, !dbg !1048
  %142 = bitcast { i64, i32 }* %21 to i8*, !dbg !1048
  %143 = bitcast %struct.dim3* %19 to i8*, !dbg !1048
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %142, i8* align 4 %143, i64 12, i1 false), !dbg !1048
  %144 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %21, i32 0, i32 0, !dbg !1048
  %145 = load i64, i64* %144, align 4, !dbg !1048
  %146 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %21, i32 0, i32 1, !dbg !1048
  %147 = load i32, i32* %146, align 4, !dbg !1048
  %148 = call i32 @__cudaPushCallConfiguration(i64 %139, i32 %141, i64 %145, i32 %147, i64 0, i8* null), !dbg !1048
  %149 = icmp ne i32 %148, 0, !dbg !1048
  br i1 %149, label %159, label %150, !dbg !1049

150:                                              ; preds = %130
  %151 = load float*, float** %2, align 8, !dbg !1050
  %152 = load float*, float** %3, align 8, !dbg !1051
  %153 = load float*, float** %4, align 8, !dbg !1052
  %154 = load i32, i32* @Size, align 4, !dbg !1053
  %155 = load i32, i32* @Size, align 4, !dbg !1054
  %156 = load i32, i32* %1, align 4, !dbg !1055
  %157 = sub nsw i32 %155, %156, !dbg !1056
  %158 = load i32, i32* %1, align 4, !dbg !1057
  call void @_Z4Fan2PfS_S_iii(float* %151, float* %152, float* %153, i32 %154, i32 %157, i32 %158), !dbg !1049
  br label %159, !dbg !1049

159:                                              ; preds = %150, %130
  %160 = call i32 @cudaThreadSynchronize(), !dbg !1058
  call void @_Z14checkCUDAErrorPKc(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.47, i64 0, i64 0)), !dbg !1059
  br label %161, !dbg !1060

161:                                              ; preds = %159
  %162 = load i32, i32* %1, align 4, !dbg !1061
  %163 = add nsw i32 %162, 1, !dbg !1061
  store i32 %163, i32* %1, align 4, !dbg !1061
  br label %101, !dbg !1062, !llvm.loop !1063

164:                                              ; preds = %101
  call void @llvm.dbg.declare(metadata %struct.timeval* %22, metadata !1065, metadata !DIExpression()), !dbg !1066
  %165 = call i32 @gettimeofday(%struct.timeval* %22, %struct.timezone* null) #11, !dbg !1067
  %166 = getelementptr inbounds %struct.timeval, %struct.timeval* %22, i32 0, i32 0, !dbg !1068
  %167 = load i64, i64* %166, align 8, !dbg !1068
  %168 = mul nsw i64 %167, 1000000, !dbg !1069
  %169 = getelementptr inbounds %struct.timeval, %struct.timeval* %22, i32 0, i32 1, !dbg !1070
  %170 = load i64, i64* %169, align 8, !dbg !1070
  %171 = add nsw i64 %168, %170, !dbg !1071
  %172 = getelementptr inbounds %struct.timeval, %struct.timeval* %13, i32 0, i32 0, !dbg !1072
  %173 = load i64, i64* %172, align 8, !dbg !1072
  %174 = mul nsw i64 %173, 1000000, !dbg !1073
  %175 = getelementptr inbounds %struct.timeval, %struct.timeval* %13, i32 0, i32 1, !dbg !1074
  %176 = load i64, i64* %175, align 8, !dbg !1074
  %177 = add nsw i64 %174, %176, !dbg !1075
  %178 = sub nsw i64 %171, %177, !dbg !1076
  %179 = trunc i64 %178 to i32, !dbg !1077
  store i32 %179, i32* @totalKernelTime, align 4, !dbg !1078
  %180 = load float*, float** @m, align 8, !dbg !1079
  %181 = bitcast float* %180 to i8*, !dbg !1079
  %182 = load float*, float** %2, align 8, !dbg !1080
  %183 = bitcast float* %182 to i8*, !dbg !1080
  %184 = load i32, i32* @Size, align 4, !dbg !1081
  %185 = load i32, i32* @Size, align 4, !dbg !1082
  %186 = mul nsw i32 %184, %185, !dbg !1083
  %187 = sext i32 %186 to i64, !dbg !1081
  %188 = mul i64 %187, 4, !dbg !1084
  %189 = call i32 @cudaMemcpy(i8* %181, i8* %183, i64 %188, i32 2), !dbg !1085
  %190 = load float*, float** @a, align 8, !dbg !1086
  %191 = bitcast float* %190 to i8*, !dbg !1086
  %192 = load float*, float** %3, align 8, !dbg !1087
  %193 = bitcast float* %192 to i8*, !dbg !1087
  %194 = load i32, i32* @Size, align 4, !dbg !1088
  %195 = load i32, i32* @Size, align 4, !dbg !1089
  %196 = mul nsw i32 %194, %195, !dbg !1090
  %197 = sext i32 %196 to i64, !dbg !1088
  %198 = mul i64 %197, 4, !dbg !1091
  %199 = call i32 @cudaMemcpy(i8* %191, i8* %193, i64 %198, i32 2), !dbg !1092
  %200 = load float*, float** @b, align 8, !dbg !1093
  %201 = bitcast float* %200 to i8*, !dbg !1093
  %202 = load float*, float** %4, align 8, !dbg !1094
  %203 = bitcast float* %202 to i8*, !dbg !1094
  %204 = load i32, i32* @Size, align 4, !dbg !1095
  %205 = sext i32 %204 to i64, !dbg !1095
  %206 = mul i64 %205, 4, !dbg !1096
  %207 = call i32 @cudaMemcpy(i8* %201, i8* %203, i64 %206, i32 2), !dbg !1097
  %208 = load float*, float** %2, align 8, !dbg !1098
  %209 = bitcast float* %208 to i8*, !dbg !1098
  %210 = call i32 @cudaFree(i8* %209), !dbg !1099
  %211 = load float*, float** %3, align 8, !dbg !1100
  %212 = bitcast float* %211 to i8*, !dbg !1100
  %213 = call i32 @cudaFree(i8* %212), !dbg !1101
  %214 = load float*, float** %4, align 8, !dbg !1102
  %215 = bitcast float* %214 to i8*, !dbg !1102
  %216 = call i32 @cudaFree(i8* %215), !dbg !1103
  ret void, !dbg !1104
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z8PrintMatPfii(float* %0, i32 %1, i32 %2) #4 !dbg !1105 {
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store float* %0, float** %4, align 8
  call void @llvm.dbg.declare(metadata float** %4, metadata !1108, metadata !DIExpression()), !dbg !1109
  store i32 %1, i32* %5, align 4
  call void @llvm.dbg.declare(metadata i32* %5, metadata !1110, metadata !DIExpression()), !dbg !1111
  store i32 %2, i32* %6, align 4
  call void @llvm.dbg.declare(metadata i32* %6, metadata !1112, metadata !DIExpression()), !dbg !1113
  call void @llvm.dbg.declare(metadata i32* %7, metadata !1114, metadata !DIExpression()), !dbg !1115
  call void @llvm.dbg.declare(metadata i32* %8, metadata !1116, metadata !DIExpression()), !dbg !1117
  store i32 0, i32* %7, align 4, !dbg !1118
  br label %9, !dbg !1120

9:                                                ; preds = %36, %3
  %10 = load i32, i32* %7, align 4, !dbg !1121
  %11 = load i32, i32* %5, align 4, !dbg !1123
  %12 = icmp slt i32 %10, %11, !dbg !1124
  br i1 %12, label %13, label %39, !dbg !1125

13:                                               ; preds = %9
  store i32 0, i32* %8, align 4, !dbg !1126
  br label %14, !dbg !1129

14:                                               ; preds = %31, %13
  %15 = load i32, i32* %8, align 4, !dbg !1130
  %16 = load i32, i32* %6, align 4, !dbg !1132
  %17 = icmp slt i32 %15, %16, !dbg !1133
  br i1 %17, label %18, label %34, !dbg !1134

18:                                               ; preds = %14
  %19 = load float*, float** %4, align 8, !dbg !1135
  %20 = load i32, i32* @Size, align 4, !dbg !1137
  %21 = load i32, i32* %7, align 4, !dbg !1138
  %22 = mul nsw i32 %20, %21, !dbg !1139
  %23 = sext i32 %22 to i64, !dbg !1140
  %24 = getelementptr inbounds float, float* %19, i64 %23, !dbg !1140
  %25 = load i32, i32* %8, align 4, !dbg !1141
  %26 = sext i32 %25 to i64, !dbg !1142
  %27 = getelementptr inbounds float, float* %24, i64 %26, !dbg !1142
  %28 = load float, float* %27, align 4, !dbg !1143
  %29 = fpext float %28 to double, !dbg !1143
  %30 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.49, i64 0, i64 0), double %29), !dbg !1144
  br label %31, !dbg !1145

31:                                               ; preds = %18
  %32 = load i32, i32* %8, align 4, !dbg !1146
  %33 = add nsw i32 %32, 1, !dbg !1146
  store i32 %33, i32* %8, align 4, !dbg !1146
  br label %14, !dbg !1147, !llvm.loop !1148

34:                                               ; preds = %14
  %35 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0)), !dbg !1150
  br label %36, !dbg !1151

36:                                               ; preds = %34
  %37 = load i32, i32* %7, align 4, !dbg !1152
  %38 = add nsw i32 %37, 1, !dbg !1152
  store i32 %38, i32* %7, align 4, !dbg !1152
  br label %9, !dbg !1153, !llvm.loop !1154

39:                                               ; preds = %9
  %40 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0)), !dbg !1156
  ret void, !dbg !1157
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z8PrintAryPfi(float* %0, i32 %1) #4 !dbg !1158 {
  %3 = alloca float*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store float* %0, float** %3, align 8
  call void @llvm.dbg.declare(metadata float** %3, metadata !1161, metadata !DIExpression()), !dbg !1162
  store i32 %1, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !1163, metadata !DIExpression()), !dbg !1164
  call void @llvm.dbg.declare(metadata i32* %5, metadata !1165, metadata !DIExpression()), !dbg !1166
  store i32 0, i32* %5, align 4, !dbg !1167
  br label %6, !dbg !1169

6:                                                ; preds = %18, %2
  %7 = load i32, i32* %5, align 4, !dbg !1170
  %8 = load i32, i32* %4, align 4, !dbg !1172
  %9 = icmp slt i32 %7, %8, !dbg !1173
  br i1 %9, label %10, label %21, !dbg !1174

10:                                               ; preds = %6
  %11 = load float*, float** %3, align 8, !dbg !1175
  %12 = load i32, i32* %5, align 4, !dbg !1177
  %13 = sext i32 %12 to i64, !dbg !1175
  %14 = getelementptr inbounds float, float* %11, i64 %13, !dbg !1175
  %15 = load float, float* %14, align 4, !dbg !1175
  %16 = fpext float %15 to double, !dbg !1175
  %17 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.50, i64 0, i64 0), double %16), !dbg !1178
  br label %18, !dbg !1179

18:                                               ; preds = %10
  %19 = load i32, i32* %5, align 4, !dbg !1180
  %20 = add nsw i32 %19, 1, !dbg !1180
  store i32 %20, i32* %5, align 4, !dbg !1180
  br label %6, !dbg !1181, !llvm.loop !1182

21:                                               ; preds = %6
  %22 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.51, i64 0, i64 0)), !dbg !1184
  ret void, !dbg !1185
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @_Z7BackSubv() #6 !dbg !1186 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = load i32, i32* @Size, align 4, !dbg !1187
  %4 = sext i32 %3 to i64, !dbg !1187
  %5 = mul i64 %4, 4, !dbg !1188
  %6 = call noalias i8* @malloc(i64 %5) #11, !dbg !1189
  %7 = bitcast i8* %6 to float*, !dbg !1190
  store float* %7, float** @finalVec, align 8, !dbg !1191
  call void @llvm.dbg.declare(metadata i32* %1, metadata !1192, metadata !DIExpression()), !dbg !1193
  call void @llvm.dbg.declare(metadata i32* %2, metadata !1194, metadata !DIExpression()), !dbg !1195
  store i32 0, i32* %1, align 4, !dbg !1196
  br label %8, !dbg !1198

8:                                                ; preds = %103, %0
  %9 = load i32, i32* %1, align 4, !dbg !1199
  %10 = load i32, i32* @Size, align 4, !dbg !1201
  %11 = icmp slt i32 %9, %10, !dbg !1202
  br i1 %11, label %12, label %106, !dbg !1203

12:                                               ; preds = %8
  %13 = load float*, float** @b, align 8, !dbg !1204
  %14 = load i32, i32* @Size, align 4, !dbg !1206
  %15 = load i32, i32* %1, align 4, !dbg !1207
  %16 = sub nsw i32 %14, %15, !dbg !1208
  %17 = sub nsw i32 %16, 1, !dbg !1209
  %18 = sext i32 %17 to i64, !dbg !1204
  %19 = getelementptr inbounds float, float* %13, i64 %18, !dbg !1204
  %20 = load float, float* %19, align 4, !dbg !1204
  %21 = load float*, float** @finalVec, align 8, !dbg !1210
  %22 = load i32, i32* @Size, align 4, !dbg !1211
  %23 = load i32, i32* %1, align 4, !dbg !1212
  %24 = sub nsw i32 %22, %23, !dbg !1213
  %25 = sub nsw i32 %24, 1, !dbg !1214
  %26 = sext i32 %25 to i64, !dbg !1210
  %27 = getelementptr inbounds float, float* %21, i64 %26, !dbg !1210
  store float %20, float* %27, align 4, !dbg !1215
  store i32 0, i32* %2, align 4, !dbg !1216
  br label %28, !dbg !1218

28:                                               ; preds = %67, %12
  %29 = load i32, i32* %2, align 4, !dbg !1219
  %30 = load i32, i32* %1, align 4, !dbg !1221
  %31 = icmp slt i32 %29, %30, !dbg !1222
  br i1 %31, label %32, label %70, !dbg !1223

32:                                               ; preds = %28
  %33 = load float*, float** @a, align 8, !dbg !1224
  %34 = load i32, i32* @Size, align 4, !dbg !1226
  %35 = load i32, i32* @Size, align 4, !dbg !1227
  %36 = load i32, i32* %1, align 4, !dbg !1228
  %37 = sub nsw i32 %35, %36, !dbg !1229
  %38 = sub nsw i32 %37, 1, !dbg !1230
  %39 = mul nsw i32 %34, %38, !dbg !1231
  %40 = sext i32 %39 to i64, !dbg !1232
  %41 = getelementptr inbounds float, float* %33, i64 %40, !dbg !1232
  %42 = load i32, i32* @Size, align 4, !dbg !1233
  %43 = load i32, i32* %2, align 4, !dbg !1234
  %44 = sub nsw i32 %42, %43, !dbg !1235
  %45 = sub nsw i32 %44, 1, !dbg !1236
  %46 = sext i32 %45 to i64, !dbg !1237
  %47 = getelementptr inbounds float, float* %41, i64 %46, !dbg !1237
  %48 = load float, float* %47, align 4, !dbg !1238
  %49 = load float*, float** @finalVec, align 8, !dbg !1239
  %50 = load i32, i32* @Size, align 4, !dbg !1240
  %51 = load i32, i32* %2, align 4, !dbg !1241
  %52 = sub nsw i32 %50, %51, !dbg !1242
  %53 = sub nsw i32 %52, 1, !dbg !1243
  %54 = sext i32 %53 to i64, !dbg !1239
  %55 = getelementptr inbounds float, float* %49, i64 %54, !dbg !1239
  %56 = load float, float* %55, align 4, !dbg !1239
  %57 = fmul contract float %48, %56, !dbg !1244
  %58 = load float*, float** @finalVec, align 8, !dbg !1245
  %59 = load i32, i32* @Size, align 4, !dbg !1246
  %60 = load i32, i32* %1, align 4, !dbg !1247
  %61 = sub nsw i32 %59, %60, !dbg !1248
  %62 = sub nsw i32 %61, 1, !dbg !1249
  %63 = sext i32 %62 to i64, !dbg !1245
  %64 = getelementptr inbounds float, float* %58, i64 %63, !dbg !1245
  %65 = load float, float* %64, align 4, !dbg !1250
  %66 = fsub contract float %65, %57, !dbg !1250
  store float %66, float* %64, align 4, !dbg !1250
  br label %67, !dbg !1251

67:                                               ; preds = %32
  %68 = load i32, i32* %2, align 4, !dbg !1252
  %69 = add nsw i32 %68, 1, !dbg !1252
  store i32 %69, i32* %2, align 4, !dbg !1252
  br label %28, !dbg !1253, !llvm.loop !1254

70:                                               ; preds = %28
  %71 = load float*, float** @finalVec, align 8, !dbg !1256
  %72 = load i32, i32* @Size, align 4, !dbg !1257
  %73 = load i32, i32* %1, align 4, !dbg !1258
  %74 = sub nsw i32 %72, %73, !dbg !1259
  %75 = sub nsw i32 %74, 1, !dbg !1260
  %76 = sext i32 %75 to i64, !dbg !1256
  %77 = getelementptr inbounds float, float* %71, i64 %76, !dbg !1256
  %78 = load float, float* %77, align 4, !dbg !1256
  %79 = load float*, float** @a, align 8, !dbg !1261
  %80 = load i32, i32* @Size, align 4, !dbg !1262
  %81 = load i32, i32* @Size, align 4, !dbg !1263
  %82 = load i32, i32* %1, align 4, !dbg !1264
  %83 = sub nsw i32 %81, %82, !dbg !1265
  %84 = sub nsw i32 %83, 1, !dbg !1266
  %85 = mul nsw i32 %80, %84, !dbg !1267
  %86 = sext i32 %85 to i64, !dbg !1268
  %87 = getelementptr inbounds float, float* %79, i64 %86, !dbg !1268
  %88 = load i32, i32* @Size, align 4, !dbg !1269
  %89 = load i32, i32* %1, align 4, !dbg !1270
  %90 = sub nsw i32 %88, %89, !dbg !1271
  %91 = sub nsw i32 %90, 1, !dbg !1272
  %92 = sext i32 %91 to i64, !dbg !1273
  %93 = getelementptr inbounds float, float* %87, i64 %92, !dbg !1273
  %94 = load float, float* %93, align 4, !dbg !1274
  %95 = fdiv float %78, %94, !dbg !1275
  %96 = load float*, float** @finalVec, align 8, !dbg !1276
  %97 = load i32, i32* @Size, align 4, !dbg !1277
  %98 = load i32, i32* %1, align 4, !dbg !1278
  %99 = sub nsw i32 %97, %98, !dbg !1279
  %100 = sub nsw i32 %99, 1, !dbg !1280
  %101 = sext i32 %100 to i64, !dbg !1276
  %102 = getelementptr inbounds float, float* %96, i64 %101, !dbg !1276
  store float %95, float* %102, align 4, !dbg !1281
  br label %103, !dbg !1282

103:                                              ; preds = %70
  %104 = load i32, i32* %1, align 4, !dbg !1283
  %105 = add nsw i32 %104, 1, !dbg !1283
  store i32 %105, i32* %1, align 4, !dbg !1283
  br label %8, !dbg !1284, !llvm.loop !1285

106:                                              ; preds = %8
  ret void, !dbg !1287
}

; Function Attrs: nounwind
declare dso_local void @free(i8*) #7

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z21PrintDevicePropertiesv() #4 !dbg !1288 {
  %1 = alloca %struct.cudaDeviceProp, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.cudaDeviceProp* %1, metadata !1289, metadata !DIExpression()), !dbg !1388
  call void @llvm.dbg.declare(metadata i32* %2, metadata !1389, metadata !DIExpression()), !dbg !1390
  store i32 0, i32* %2, align 4, !dbg !1390
  %4 = call i32 @cudaGetDeviceCount(i32* %2), !dbg !1391
  %5 = load i32, i32* %2, align 4, !dbg !1392
  %6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i32 %5), !dbg !1393
  call void @llvm.dbg.declare(metadata i32* %3, metadata !1394, metadata !DIExpression()), !dbg !1396
  store i32 0, i32* %3, align 4, !dbg !1396
  br label %7, !dbg !1397

7:                                                ; preds = %89, %0
  %8 = load i32, i32* %3, align 4, !dbg !1398
  %9 = load i32, i32* %2, align 4, !dbg !1400
  %10 = icmp slt i32 %8, %9, !dbg !1401
  br i1 %10, label %11, label %92, !dbg !1402

11:                                               ; preds = %7
  %12 = bitcast %struct.cudaDeviceProp* %1 to i8*, !dbg !1403
  call void @llvm.memset.p0i8.i64(i8* align 8 %12, i8 0, i64 712, i1 false), !dbg !1403
  %13 = load i32, i32* %3, align 4, !dbg !1405
  %14 = call i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp* %1, i32 %13), !dbg !1407
  %15 = icmp eq i32 0, %14, !dbg !1408
  br i1 %15, label %16, label %84, !dbg !1409

16:                                               ; preds = %11
  %17 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 0, !dbg !1410
  %18 = getelementptr inbounds [256 x i8], [256 x i8]* %17, i64 0, i64 0, !dbg !1412
  %19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.26, i64 0, i64 0), i8* %18), !dbg !1413
  %20 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.27, i64 0, i64 0)), !dbg !1414
  %21 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 4, !dbg !1415
  %22 = load i64, i64* %21, align 8, !dbg !1415
  %23 = udiv i64 %22, 1024, !dbg !1416
  %24 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.28, i64 0, i64 0), i64 %23), !dbg !1417
  %25 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 5, !dbg !1418
  %26 = load i64, i64* %25, align 8, !dbg !1418
  %27 = udiv i64 %26, 1024, !dbg !1419
  %28 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.29, i64 0, i64 0), i64 %27), !dbg !1420
  %29 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 6, !dbg !1421
  %30 = load i32, i32* %29, align 8, !dbg !1421
  %31 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.30, i64 0, i64 0), i32 %30), !dbg !1422
  %32 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 7, !dbg !1423
  %33 = load i32, i32* %32, align 4, !dbg !1423
  %34 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.31, i64 0, i64 0), i32 %33), !dbg !1424
  %35 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 8, !dbg !1425
  %36 = load i64, i64* %35, align 8, !dbg !1425
  %37 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.32, i64 0, i64 0), i64 %36), !dbg !1426
  %38 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 9, !dbg !1427
  %39 = load i32, i32* %38, align 8, !dbg !1427
  %40 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.33, i64 0, i64 0), i32 %39), !dbg !1428
  %41 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 10, !dbg !1429
  %42 = getelementptr inbounds [3 x i32], [3 x i32]* %41, i64 0, i64 0, !dbg !1430
  %43 = load i32, i32* %42, align 4, !dbg !1430
  %44 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 10, !dbg !1431
  %45 = getelementptr inbounds [3 x i32], [3 x i32]* %44, i64 0, i64 1, !dbg !1432
  %46 = load i32, i32* %45, align 4, !dbg !1432
  %47 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 10, !dbg !1433
  %48 = getelementptr inbounds [3 x i32], [3 x i32]* %47, i64 0, i64 2, !dbg !1434
  %49 = load i32, i32* %48, align 4, !dbg !1434
  %50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([47 x i8], [47 x i8]* @.str.34, i64 0, i64 0), i32 %43, i32 %46, i32 %49), !dbg !1435
  %51 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 11, !dbg !1436
  %52 = getelementptr inbounds [3 x i32], [3 x i32]* %51, i64 0, i64 0, !dbg !1437
  %53 = load i32, i32* %52, align 8, !dbg !1437
  %54 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 11, !dbg !1438
  %55 = getelementptr inbounds [3 x i32], [3 x i32]* %54, i64 0, i64 1, !dbg !1439
  %56 = load i32, i32* %55, align 4, !dbg !1439
  %57 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 11, !dbg !1440
  %58 = getelementptr inbounds [3 x i32], [3 x i32]* %57, i64 0, i64 2, !dbg !1441
  %59 = load i32, i32* %58, align 8, !dbg !1441
  %60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.35, i64 0, i64 0), i32 %53, i32 %56, i32 %59), !dbg !1442
  %61 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 13, !dbg !1443
  %62 = load i64, i64* %61, align 8, !dbg !1443
  %63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.36, i64 0, i64 0), i64 %62), !dbg !1444
  %64 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 14, !dbg !1445
  %65 = load i32, i32* %64, align 8, !dbg !1445
  %66 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 15, !dbg !1446
  %67 = load i32, i32* %66, align 4, !dbg !1446
  %68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.37, i64 0, i64 0), i32 %65, i32 %67), !dbg !1447
  %69 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 12, !dbg !1448
  %70 = load i32, i32* %69, align 4, !dbg !1448
  %71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.38, i64 0, i64 0), i32 %70), !dbg !1449
  %72 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 16, !dbg !1450
  %73 = load i64, i64* %72, align 8, !dbg !1450
  %74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.39, i64 0, i64 0), i64 %73), !dbg !1451
  %75 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 18, !dbg !1452
  %76 = load i32, i32* %75, align 8, !dbg !1452
  %77 = icmp ne i32 %76, 0, !dbg !1453
  %78 = zext i1 %77 to i64, !dbg !1453
  %79 = select i1 %77, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.41, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.42, i64 0, i64 0), !dbg !1453
  %80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.40, i64 0, i64 0), i8* %79), !dbg !1454
  %81 = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %1, i32 0, i32 19, !dbg !1455
  %82 = load i32, i32* %81, align 4, !dbg !1455
  %83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.43, i64 0, i64 0), i32 %82), !dbg !1456
  br label %88, !dbg !1457

84:                                               ; preds = %11
  %85 = call i32 @cudaGetLastError(), !dbg !1458
  %86 = call i8* @cudaGetErrorString(i32 %85), !dbg !1459
  %87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.44, i64 0, i64 0), i8* %86), !dbg !1460
  br label %88

88:                                               ; preds = %84, %16
  br label %89, !dbg !1461

89:                                               ; preds = %88
  %90 = load i32, i32* %3, align 4, !dbg !1462
  %91 = add nsw i32 %90, 1, !dbg !1462
  store i32 %91, i32* %3, align 4, !dbg !1462
  br label %7, !dbg !1463, !llvm.loop !1464

92:                                               ; preds = %7
  ret void, !dbg !1466
}

declare dso_local i32 @cudaGetDeviceCount(i32*) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #8

declare dso_local i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp*, i32) #2

declare dso_local i8* @cudaGetErrorString(i32) #2

declare dso_local i32 @cudaGetLastError() #2

declare dso_local %struct._IO_FILE* @fopen(i8*, i8*) #2

declare dso_local i32 @fscanf(%struct._IO_FILE*, i8*, ...) #2

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #7

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z7InitMatPfii(float* %0, i32 %1, i32 %2) #4 !dbg !1467 {
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store float* %0, float** %4, align 8
  call void @llvm.dbg.declare(metadata float** %4, metadata !1468, metadata !DIExpression()), !dbg !1469
  store i32 %1, i32* %5, align 4
  call void @llvm.dbg.declare(metadata i32* %5, metadata !1470, metadata !DIExpression()), !dbg !1471
  store i32 %2, i32* %6, align 4
  call void @llvm.dbg.declare(metadata i32* %6, metadata !1472, metadata !DIExpression()), !dbg !1473
  call void @llvm.dbg.declare(metadata i32* %7, metadata !1474, metadata !DIExpression()), !dbg !1475
  call void @llvm.dbg.declare(metadata i32* %8, metadata !1476, metadata !DIExpression()), !dbg !1477
  store i32 0, i32* %7, align 4, !dbg !1478
  br label %9, !dbg !1480

9:                                                ; preds = %34, %3
  %10 = load i32, i32* %7, align 4, !dbg !1481
  %11 = load i32, i32* %5, align 4, !dbg !1483
  %12 = icmp slt i32 %10, %11, !dbg !1484
  br i1 %12, label %13, label %37, !dbg !1485

13:                                               ; preds = %9
  store i32 0, i32* %8, align 4, !dbg !1486
  br label %14, !dbg !1489

14:                                               ; preds = %30, %13
  %15 = load i32, i32* %8, align 4, !dbg !1490
  %16 = load i32, i32* %6, align 4, !dbg !1492
  %17 = icmp slt i32 %15, %16, !dbg !1493
  br i1 %17, label %18, label %33, !dbg !1494

18:                                               ; preds = %14
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8, !dbg !1495
  %20 = load float*, float** %4, align 8, !dbg !1497
  %21 = load i32, i32* @Size, align 4, !dbg !1498
  %22 = load i32, i32* %7, align 4, !dbg !1499
  %23 = mul nsw i32 %21, %22, !dbg !1500
  %24 = sext i32 %23 to i64, !dbg !1501
  %25 = getelementptr inbounds float, float* %20, i64 %24, !dbg !1501
  %26 = load i32, i32* %8, align 4, !dbg !1502
  %27 = sext i32 %26 to i64, !dbg !1503
  %28 = getelementptr inbounds float, float* %25, i64 %27, !dbg !1503
  %29 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %19, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.48, i64 0, i64 0), float* %28), !dbg !1504
  br label %30, !dbg !1505

30:                                               ; preds = %18
  %31 = load i32, i32* %8, align 4, !dbg !1506
  %32 = add nsw i32 %31, 1, !dbg !1506
  store i32 %32, i32* %8, align 4, !dbg !1506
  br label %14, !dbg !1507, !llvm.loop !1508

33:                                               ; preds = %14
  br label %34, !dbg !1510

34:                                               ; preds = %33
  %35 = load i32, i32* %7, align 4, !dbg !1511
  %36 = add nsw i32 %35, 1, !dbg !1511
  store i32 %36, i32* %7, align 4, !dbg !1511
  br label %9, !dbg !1512, !llvm.loop !1513

37:                                               ; preds = %9
  ret void, !dbg !1515
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z7InitAryPfi(float* %0, i32 %1) #4 !dbg !1516 {
  %3 = alloca float*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store float* %0, float** %3, align 8
  call void @llvm.dbg.declare(metadata float** %3, metadata !1517, metadata !DIExpression()), !dbg !1518
  store i32 %1, i32* %4, align 4
  call void @llvm.dbg.declare(metadata i32* %4, metadata !1519, metadata !DIExpression()), !dbg !1520
  call void @llvm.dbg.declare(metadata i32* %5, metadata !1521, metadata !DIExpression()), !dbg !1522
  store i32 0, i32* %5, align 4, !dbg !1523
  br label %6, !dbg !1525

6:                                                ; preds = %17, %2
  %7 = load i32, i32* %5, align 4, !dbg !1526
  %8 = load i32, i32* %4, align 4, !dbg !1528
  %9 = icmp slt i32 %7, %8, !dbg !1529
  br i1 %9, label %10, label %20, !dbg !1530

10:                                               ; preds = %6
  %11 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8, !dbg !1531
  %12 = load float*, float** %3, align 8, !dbg !1533
  %13 = load i32, i32* %5, align 4, !dbg !1534
  %14 = sext i32 %13 to i64, !dbg !1533
  %15 = getelementptr inbounds float, float* %12, i64 %14, !dbg !1533
  %16 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %11, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.48, i64 0, i64 0), float* %15), !dbg !1535
  br label %17, !dbg !1536

17:                                               ; preds = %10
  %18 = load i32, i32* %5, align 4, !dbg !1537
  %19 = add nsw i32 %18, 1, !dbg !1537
  store i32 %19, i32* %5, align 4, !dbg !1537
  br label %6, !dbg !1538, !llvm.loop !1539

20:                                               ; preds = %6
  ret void, !dbg !1541
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z4Fan1PfS_ii(float* %0, float* %1, i32 %2, i32 %3) #4 !dbg !1542 {
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca i8*, align 8
  %13 = alloca { i64, i32 }, align 8
  %14 = alloca { i64, i32 }, align 8
  store float* %0, float** %5, align 8
  call void @llvm.dbg.declare(metadata float** %5, metadata !1545, metadata !DIExpression()), !dbg !1546
  store float* %1, float** %6, align 8
  call void @llvm.dbg.declare(metadata float** %6, metadata !1547, metadata !DIExpression()), !dbg !1548
  store i32 %2, i32* %7, align 4
  call void @llvm.dbg.declare(metadata i32* %7, metadata !1549, metadata !DIExpression()), !dbg !1550
  store i32 %3, i32* %8, align 4
  call void @llvm.dbg.declare(metadata i32* %8, metadata !1551, metadata !DIExpression()), !dbg !1552
  %15 = alloca i8*, i64 4, align 16, !dbg !1553
  %16 = bitcast float** %5 to i8*, !dbg !1553
  %17 = getelementptr i8*, i8** %15, i32 0, !dbg !1553
  store i8* %16, i8** %17, !dbg !1553
  %18 = bitcast float** %6 to i8*, !dbg !1553
  %19 = getelementptr i8*, i8** %15, i32 1, !dbg !1553
  store i8* %18, i8** %19, !dbg !1553
  %20 = bitcast i32* %7 to i8*, !dbg !1553
  %21 = getelementptr i8*, i8** %15, i32 2, !dbg !1553
  store i8* %20, i8** %21, !dbg !1553
  %22 = bitcast i32* %8 to i8*, !dbg !1553
  %23 = getelementptr i8*, i8** %15, i32 3, !dbg !1553
  store i8* %22, i8** %23, !dbg !1553
  %24 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %9, %struct.dim3* %10, i64* %11, i8** %12), !dbg !1553
  %25 = load i64, i64* %11, align 8, !dbg !1553
  %26 = load i8*, i8** %12, align 8, !dbg !1553
  %27 = bitcast { i64, i32 }* %13 to i8*, !dbg !1553
  %28 = bitcast %struct.dim3* %9 to i8*, !dbg !1553
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %27, i8* align 8 %28, i64 12, i1 false), !dbg !1553
  %29 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %13, i32 0, i32 0, !dbg !1553
  %30 = load i64, i64* %29, align 8, !dbg !1553
  %31 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %13, i32 0, i32 1, !dbg !1553
  %32 = load i32, i32* %31, align 8, !dbg !1553
  %33 = bitcast { i64, i32 }* %14 to i8*, !dbg !1553
  %34 = bitcast %struct.dim3* %10 to i8*, !dbg !1553
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %33, i8* align 8 %34, i64 12, i1 false), !dbg !1553
  %35 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %14, i32 0, i32 0, !dbg !1553
  %36 = load i64, i64* %35, align 8, !dbg !1553
  %37 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %14, i32 0, i32 1, !dbg !1553
  %38 = load i32, i32* %37, align 8, !dbg !1553
  %39 = bitcast i8* %26 to %struct.CUstream_st*, !dbg !1553
  %40 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, i32, i32)* @_Z4Fan1PfS_ii to i8*), i64 %30, i32 %32, i64 %36, i32 %38, i8** %15, i64 %25, %struct.CUstream_st* %39), !dbg !1553
  br label %41, !dbg !1553

41:                                               ; preds = %4
  ret void, !dbg !1554
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**)

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #8

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z4Fan2PfS_S_iii(float* %0, float* %1, float* %2, i32 %3, i32 %4, i32 %5) #4 !dbg !1555 {
  %7 = alloca float*, align 8
  %8 = alloca float*, align 8
  %9 = alloca float*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %struct.dim3, align 8
  %14 = alloca %struct.dim3, align 8
  %15 = alloca i64, align 8
  %16 = alloca i8*, align 8
  %17 = alloca { i64, i32 }, align 8
  %18 = alloca { i64, i32 }, align 8
  store float* %0, float** %7, align 8
  call void @llvm.dbg.declare(metadata float** %7, metadata !1558, metadata !DIExpression()), !dbg !1559
  store float* %1, float** %8, align 8
  call void @llvm.dbg.declare(metadata float** %8, metadata !1560, metadata !DIExpression()), !dbg !1561
  store float* %2, float** %9, align 8
  call void @llvm.dbg.declare(metadata float** %9, metadata !1562, metadata !DIExpression()), !dbg !1563
  store i32 %3, i32* %10, align 4
  call void @llvm.dbg.declare(metadata i32* %10, metadata !1564, metadata !DIExpression()), !dbg !1565
  store i32 %4, i32* %11, align 4
  call void @llvm.dbg.declare(metadata i32* %11, metadata !1566, metadata !DIExpression()), !dbg !1567
  store i32 %5, i32* %12, align 4
  call void @llvm.dbg.declare(metadata i32* %12, metadata !1568, metadata !DIExpression()), !dbg !1569
  %19 = alloca i8*, i64 6, align 16, !dbg !1570
  %20 = bitcast float** %7 to i8*, !dbg !1570
  %21 = getelementptr i8*, i8** %19, i32 0, !dbg !1570
  store i8* %20, i8** %21, !dbg !1570
  %22 = bitcast float** %8 to i8*, !dbg !1570
  %23 = getelementptr i8*, i8** %19, i32 1, !dbg !1570
  store i8* %22, i8** %23, !dbg !1570
  %24 = bitcast float** %9 to i8*, !dbg !1570
  %25 = getelementptr i8*, i8** %19, i32 2, !dbg !1570
  store i8* %24, i8** %25, !dbg !1570
  %26 = bitcast i32* %10 to i8*, !dbg !1570
  %27 = getelementptr i8*, i8** %19, i32 3, !dbg !1570
  store i8* %26, i8** %27, !dbg !1570
  %28 = bitcast i32* %11 to i8*, !dbg !1570
  %29 = getelementptr i8*, i8** %19, i32 4, !dbg !1570
  store i8* %28, i8** %29, !dbg !1570
  %30 = bitcast i32* %12 to i8*, !dbg !1570
  %31 = getelementptr i8*, i8** %19, i32 5, !dbg !1570
  store i8* %30, i8** %31, !dbg !1570
  %32 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %13, %struct.dim3* %14, i64* %15, i8** %16), !dbg !1570
  %33 = load i64, i64* %15, align 8, !dbg !1570
  %34 = load i8*, i8** %16, align 8, !dbg !1570
  %35 = bitcast { i64, i32 }* %17 to i8*, !dbg !1570
  %36 = bitcast %struct.dim3* %13 to i8*, !dbg !1570
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %35, i8* align 8 %36, i64 12, i1 false), !dbg !1570
  %37 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %17, i32 0, i32 0, !dbg !1570
  %38 = load i64, i64* %37, align 8, !dbg !1570
  %39 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %17, i32 0, i32 1, !dbg !1570
  %40 = load i32, i32* %39, align 8, !dbg !1570
  %41 = bitcast { i64, i32 }* %18 to i8*, !dbg !1570
  %42 = bitcast %struct.dim3* %14 to i8*, !dbg !1570
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %41, i8* align 8 %42, i64 12, i1 false), !dbg !1570
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %18, i32 0, i32 0, !dbg !1570
  %44 = load i64, i64* %43, align 8, !dbg !1570
  %45 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %18, i32 0, i32 1, !dbg !1570
  %46 = load i32, i32* %45, align 8, !dbg !1570
  %47 = bitcast i8* %34 to %struct.CUstream_st*, !dbg !1570
  %48 = call i32 @cudaLaunchKernel(i8* bitcast (void (float*, float*, float*, i32, i32, i32)* @_Z4Fan2PfS_S_iii to i8*), i64 %38, i32 %40, i64 %44, i32 %46, i8** %19, i64 %33, %struct.CUstream_st* %47), !dbg !1570
  br label %49, !dbg !1570

49:                                               ; preds = %6
  ret void, !dbg !1571
}

declare dso_local i32 @cudaMalloc(i8**, i64) #2

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #2

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %0, i32 %1, i32 %2, i32 %3) unnamed_addr #6 comdat align 2 !dbg !1572 {
  %5 = alloca %struct.dim3*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store %struct.dim3* %0, %struct.dim3** %5, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %5, metadata !1573, metadata !DIExpression()), !dbg !1575
  store i32 %1, i32* %6, align 4
  call void @llvm.dbg.declare(metadata i32* %6, metadata !1576, metadata !DIExpression()), !dbg !1577
  store i32 %2, i32* %7, align 4
  call void @llvm.dbg.declare(metadata i32* %7, metadata !1578, metadata !DIExpression()), !dbg !1579
  store i32 %3, i32* %8, align 4
  call void @llvm.dbg.declare(metadata i32* %8, metadata !1580, metadata !DIExpression()), !dbg !1581
  %9 = load %struct.dim3*, %struct.dim3** %5, align 8
  %10 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 0, !dbg !1582
  %11 = load i32, i32* %6, align 4, !dbg !1583
  store i32 %11, i32* %10, align 4, !dbg !1582
  %12 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 1, !dbg !1584
  %13 = load i32, i32* %7, align 4, !dbg !1585
  store i32 %13, i32* %12, align 4, !dbg !1584
  %14 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 2, !dbg !1586
  %15 = load i32, i32* %8, align 4, !dbg !1587
  store i32 %15, i32* %14, align 4, !dbg !1586
  ret void, !dbg !1588
}

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) #2

declare dso_local i32 @cudaThreadSynchronize() #2

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z14checkCUDAErrorPKc(i8* %0) #4 !dbg !1589 {
  %2 = alloca i8*, align 8
  %3 = alloca i32, align 4
  store i8* %0, i8** %2, align 8
  call void @llvm.dbg.declare(metadata i8** %2, metadata !1592, metadata !DIExpression()), !dbg !1593
  call void @llvm.dbg.declare(metadata i32* %3, metadata !1594, metadata !DIExpression()), !dbg !1596
  %4 = call i32 @cudaGetLastError(), !dbg !1597
  store i32 %4, i32* %3, align 4, !dbg !1596
  %5 = load i32, i32* %3, align 4, !dbg !1598
  %6 = icmp ne i32 0, %5, !dbg !1600
  br i1 %6, label %7, label %13, !dbg !1601

7:                                                ; preds = %1
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !1602
  %9 = load i8*, i8** %2, align 8, !dbg !1604
  %10 = load i32, i32* %3, align 4, !dbg !1605
  %11 = call i8* @cudaGetErrorString(i32 %10), !dbg !1606
  %12 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %8, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.52, i64 0, i64 0), i8* %9, i8* %11), !dbg !1607
  call void @exit(i32 1) #9, !dbg !1608
  unreachable, !dbg !1608

13:                                               ; preds = %1
  ret void, !dbg !1609
}

declare dso_local i32 @cudaFree(i8*) #2

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #2

attributes #0 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { argmemonly nounwind willreturn }
attributes #9 = { noreturn nounwind }
attributes #10 = { nounwind readonly }
attributes #11 = { nounwind }

!llvm.module.flags = !{!727, !728, !729, !730}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!731}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Size", scope: !2, file: !3, line: 23, type: !154, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !124, globals: !129, imports: !147, nameTableKind: None)
!3 = !DIFile(filename: "gaussian.cu", directory: "/home/gkarlos/Projects/kerma-static-analysis/input/rodinia/gaussian")
!4 = !{!5, !117}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaError", file: !6, line: 189, baseType: !7, size: 32, elements: !8, identifier: "_ZTS9cudaError")
!6 = !DIFile(filename: "/usr/local/cuda/include/driver_types.h", directory: "")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116}
!9 = !DIEnumerator(name: "cudaSuccess", value: 0, isUnsigned: true)
!10 = !DIEnumerator(name: "cudaErrorInvalidValue", value: 1, isUnsigned: true)
!11 = !DIEnumerator(name: "cudaErrorMemoryAllocation", value: 2, isUnsigned: true)
!12 = !DIEnumerator(name: "cudaErrorInitializationError", value: 3, isUnsigned: true)
!13 = !DIEnumerator(name: "cudaErrorCudartUnloading", value: 4, isUnsigned: true)
!14 = !DIEnumerator(name: "cudaErrorProfilerDisabled", value: 5, isUnsigned: true)
!15 = !DIEnumerator(name: "cudaErrorProfilerNotInitialized", value: 6, isUnsigned: true)
!16 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStarted", value: 7, isUnsigned: true)
!17 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStopped", value: 8, isUnsigned: true)
!18 = !DIEnumerator(name: "cudaErrorInvalidConfiguration", value: 9, isUnsigned: true)
!19 = !DIEnumerator(name: "cudaErrorInvalidPitchValue", value: 12, isUnsigned: true)
!20 = !DIEnumerator(name: "cudaErrorInvalidSymbol", value: 13, isUnsigned: true)
!21 = !DIEnumerator(name: "cudaErrorInvalidHostPointer", value: 16, isUnsigned: true)
!22 = !DIEnumerator(name: "cudaErrorInvalidDevicePointer", value: 17, isUnsigned: true)
!23 = !DIEnumerator(name: "cudaErrorInvalidTexture", value: 18, isUnsigned: true)
!24 = !DIEnumerator(name: "cudaErrorInvalidTextureBinding", value: 19, isUnsigned: true)
!25 = !DIEnumerator(name: "cudaErrorInvalidChannelDescriptor", value: 20, isUnsigned: true)
!26 = !DIEnumerator(name: "cudaErrorInvalidMemcpyDirection", value: 21, isUnsigned: true)
!27 = !DIEnumerator(name: "cudaErrorAddressOfConstant", value: 22, isUnsigned: true)
!28 = !DIEnumerator(name: "cudaErrorTextureFetchFailed", value: 23, isUnsigned: true)
!29 = !DIEnumerator(name: "cudaErrorTextureNotBound", value: 24, isUnsigned: true)
!30 = !DIEnumerator(name: "cudaErrorSynchronizationError", value: 25, isUnsigned: true)
!31 = !DIEnumerator(name: "cudaErrorInvalidFilterSetting", value: 26, isUnsigned: true)
!32 = !DIEnumerator(name: "cudaErrorInvalidNormSetting", value: 27, isUnsigned: true)
!33 = !DIEnumerator(name: "cudaErrorMixedDeviceExecution", value: 28, isUnsigned: true)
!34 = !DIEnumerator(name: "cudaErrorNotYetImplemented", value: 31, isUnsigned: true)
!35 = !DIEnumerator(name: "cudaErrorMemoryValueTooLarge", value: 32, isUnsigned: true)
!36 = !DIEnumerator(name: "cudaErrorInsufficientDriver", value: 35, isUnsigned: true)
!37 = !DIEnumerator(name: "cudaErrorInvalidSurface", value: 37, isUnsigned: true)
!38 = !DIEnumerator(name: "cudaErrorDuplicateVariableName", value: 43, isUnsigned: true)
!39 = !DIEnumerator(name: "cudaErrorDuplicateTextureName", value: 44, isUnsigned: true)
!40 = !DIEnumerator(name: "cudaErrorDuplicateSurfaceName", value: 45, isUnsigned: true)
!41 = !DIEnumerator(name: "cudaErrorDevicesUnavailable", value: 46, isUnsigned: true)
!42 = !DIEnumerator(name: "cudaErrorIncompatibleDriverContext", value: 49, isUnsigned: true)
!43 = !DIEnumerator(name: "cudaErrorMissingConfiguration", value: 52, isUnsigned: true)
!44 = !DIEnumerator(name: "cudaErrorPriorLaunchFailure", value: 53, isUnsigned: true)
!45 = !DIEnumerator(name: "cudaErrorLaunchMaxDepthExceeded", value: 65, isUnsigned: true)
!46 = !DIEnumerator(name: "cudaErrorLaunchFileScopedTex", value: 66, isUnsigned: true)
!47 = !DIEnumerator(name: "cudaErrorLaunchFileScopedSurf", value: 67, isUnsigned: true)
!48 = !DIEnumerator(name: "cudaErrorSyncDepthExceeded", value: 68, isUnsigned: true)
!49 = !DIEnumerator(name: "cudaErrorLaunchPendingCountExceeded", value: 69, isUnsigned: true)
!50 = !DIEnumerator(name: "cudaErrorInvalidDeviceFunction", value: 98, isUnsigned: true)
!51 = !DIEnumerator(name: "cudaErrorNoDevice", value: 100, isUnsigned: true)
!52 = !DIEnumerator(name: "cudaErrorInvalidDevice", value: 101, isUnsigned: true)
!53 = !DIEnumerator(name: "cudaErrorStartupFailure", value: 127, isUnsigned: true)
!54 = !DIEnumerator(name: "cudaErrorInvalidKernelImage", value: 200, isUnsigned: true)
!55 = !DIEnumerator(name: "cudaErrorDeviceUninitilialized", value: 201, isUnsigned: true)
!56 = !DIEnumerator(name: "cudaErrorMapBufferObjectFailed", value: 205, isUnsigned: true)
!57 = !DIEnumerator(name: "cudaErrorUnmapBufferObjectFailed", value: 206, isUnsigned: true)
!58 = !DIEnumerator(name: "cudaErrorArrayIsMapped", value: 207, isUnsigned: true)
!59 = !DIEnumerator(name: "cudaErrorAlreadyMapped", value: 208, isUnsigned: true)
!60 = !DIEnumerator(name: "cudaErrorNoKernelImageForDevice", value: 209, isUnsigned: true)
!61 = !DIEnumerator(name: "cudaErrorAlreadyAcquired", value: 210, isUnsigned: true)
!62 = !DIEnumerator(name: "cudaErrorNotMapped", value: 211, isUnsigned: true)
!63 = !DIEnumerator(name: "cudaErrorNotMappedAsArray", value: 212, isUnsigned: true)
!64 = !DIEnumerator(name: "cudaErrorNotMappedAsPointer", value: 213, isUnsigned: true)
!65 = !DIEnumerator(name: "cudaErrorECCUncorrectable", value: 214, isUnsigned: true)
!66 = !DIEnumerator(name: "cudaErrorUnsupportedLimit", value: 215, isUnsigned: true)
!67 = !DIEnumerator(name: "cudaErrorDeviceAlreadyInUse", value: 216, isUnsigned: true)
!68 = !DIEnumerator(name: "cudaErrorPeerAccessUnsupported", value: 217, isUnsigned: true)
!69 = !DIEnumerator(name: "cudaErrorInvalidPtx", value: 218, isUnsigned: true)
!70 = !DIEnumerator(name: "cudaErrorInvalidGraphicsContext", value: 219, isUnsigned: true)
!71 = !DIEnumerator(name: "cudaErrorNvlinkUncorrectable", value: 220, isUnsigned: true)
!72 = !DIEnumerator(name: "cudaErrorJitCompilerNotFound", value: 221, isUnsigned: true)
!73 = !DIEnumerator(name: "cudaErrorInvalidSource", value: 300, isUnsigned: true)
!74 = !DIEnumerator(name: "cudaErrorFileNotFound", value: 301, isUnsigned: true)
!75 = !DIEnumerator(name: "cudaErrorSharedObjectSymbolNotFound", value: 302, isUnsigned: true)
!76 = !DIEnumerator(name: "cudaErrorSharedObjectInitFailed", value: 303, isUnsigned: true)
!77 = !DIEnumerator(name: "cudaErrorOperatingSystem", value: 304, isUnsigned: true)
!78 = !DIEnumerator(name: "cudaErrorInvalidResourceHandle", value: 400, isUnsigned: true)
!79 = !DIEnumerator(name: "cudaErrorIllegalState", value: 401, isUnsigned: true)
!80 = !DIEnumerator(name: "cudaErrorSymbolNotFound", value: 500, isUnsigned: true)
!81 = !DIEnumerator(name: "cudaErrorNotReady", value: 600, isUnsigned: true)
!82 = !DIEnumerator(name: "cudaErrorIllegalAddress", value: 700, isUnsigned: true)
!83 = !DIEnumerator(name: "cudaErrorLaunchOutOfResources", value: 701, isUnsigned: true)
!84 = !DIEnumerator(name: "cudaErrorLaunchTimeout", value: 702, isUnsigned: true)
!85 = !DIEnumerator(name: "cudaErrorLaunchIncompatibleTexturing", value: 703, isUnsigned: true)
!86 = !DIEnumerator(name: "cudaErrorPeerAccessAlreadyEnabled", value: 704, isUnsigned: true)
!87 = !DIEnumerator(name: "cudaErrorPeerAccessNotEnabled", value: 705, isUnsigned: true)
!88 = !DIEnumerator(name: "cudaErrorSetOnActiveProcess", value: 708, isUnsigned: true)
!89 = !DIEnumerator(name: "cudaErrorContextIsDestroyed", value: 709, isUnsigned: true)
!90 = !DIEnumerator(name: "cudaErrorAssert", value: 710, isUnsigned: true)
!91 = !DIEnumerator(name: "cudaErrorTooManyPeers", value: 711, isUnsigned: true)
!92 = !DIEnumerator(name: "cudaErrorHostMemoryAlreadyRegistered", value: 712, isUnsigned: true)
!93 = !DIEnumerator(name: "cudaErrorHostMemoryNotRegistered", value: 713, isUnsigned: true)
!94 = !DIEnumerator(name: "cudaErrorHardwareStackError", value: 714, isUnsigned: true)
!95 = !DIEnumerator(name: "cudaErrorIllegalInstruction", value: 715, isUnsigned: true)
!96 = !DIEnumerator(name: "cudaErrorMisalignedAddress", value: 716, isUnsigned: true)
!97 = !DIEnumerator(name: "cudaErrorInvalidAddressSpace", value: 717, isUnsigned: true)
!98 = !DIEnumerator(name: "cudaErrorInvalidPc", value: 718, isUnsigned: true)
!99 = !DIEnumerator(name: "cudaErrorLaunchFailure", value: 719, isUnsigned: true)
!100 = !DIEnumerator(name: "cudaErrorCooperativeLaunchTooLarge", value: 720, isUnsigned: true)
!101 = !DIEnumerator(name: "cudaErrorNotPermitted", value: 800, isUnsigned: true)
!102 = !DIEnumerator(name: "cudaErrorNotSupported", value: 801, isUnsigned: true)
!103 = !DIEnumerator(name: "cudaErrorSystemNotReady", value: 802, isUnsigned: true)
!104 = !DIEnumerator(name: "cudaErrorSystemDriverMismatch", value: 803, isUnsigned: true)
!105 = !DIEnumerator(name: "cudaErrorCompatNotSupportedOnDevice", value: 804, isUnsigned: true)
!106 = !DIEnumerator(name: "cudaErrorStreamCaptureUnsupported", value: 900, isUnsigned: true)
!107 = !DIEnumerator(name: "cudaErrorStreamCaptureInvalidated", value: 901, isUnsigned: true)
!108 = !DIEnumerator(name: "cudaErrorStreamCaptureMerge", value: 902, isUnsigned: true)
!109 = !DIEnumerator(name: "cudaErrorStreamCaptureUnmatched", value: 903, isUnsigned: true)
!110 = !DIEnumerator(name: "cudaErrorStreamCaptureUnjoined", value: 904, isUnsigned: true)
!111 = !DIEnumerator(name: "cudaErrorStreamCaptureIsolation", value: 905, isUnsigned: true)
!112 = !DIEnumerator(name: "cudaErrorStreamCaptureImplicit", value: 906, isUnsigned: true)
!113 = !DIEnumerator(name: "cudaErrorCapturedEvent", value: 907, isUnsigned: true)
!114 = !DIEnumerator(name: "cudaErrorStreamCaptureWrongThread", value: 908, isUnsigned: true)
!115 = !DIEnumerator(name: "cudaErrorUnknown", value: 999, isUnsigned: true)
!116 = !DIEnumerator(name: "cudaErrorApiFailureBase", value: 10000, isUnsigned: true)
!117 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !6, line: 1020, baseType: !7, size: 32, elements: !118, identifier: "_ZTS14cudaMemcpyKind")
!118 = !{!119, !120, !121, !122, !123}
!119 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0, isUnsigned: true)
!120 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1, isUnsigned: true)
!121 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2, isUnsigned: true)
!122 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3, isUnsigned: true)
!123 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4, isUnsigned: true)
!124 = !{!125, !127}
!125 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !126, size: 64)
!126 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!127 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !128, size: 64)
!128 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!129 = !{!0, !130, !132, !134, !136, !138, !145}
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 24, type: !125, isLocal: false, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 24, type: !125, isLocal: false, isDefinition: true)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = distinct !DIGlobalVariable(name: "finalVec", scope: !2, file: !3, line: 24, type: !125, isLocal: false, isDefinition: true)
!136 = !DIGlobalVariableExpression(var: !137, expr: !DIExpression())
!137 = distinct !DIGlobalVariable(name: "m", scope: !2, file: !3, line: 25, type: !125, isLocal: false, isDefinition: true)
!138 = !DIGlobalVariableExpression(var: !139, expr: !DIExpression())
!139 = distinct !DIGlobalVariable(name: "fp", scope: !2, file: !3, line: 27, type: !140, isLocal: false, isDefinition: true)
!140 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !141, size: 64)
!141 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !142, line: 7, baseType: !143)
!142 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!143 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !144, line: 245, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!144 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/libio.h", directory: "")
!145 = !DIGlobalVariableExpression(var: !146, expr: !DIExpression())
!146 = distinct !DIGlobalVariable(name: "totalKernelTime", scope: !2, file: !3, line: 42, type: !7, isLocal: false, isDefinition: true)
!147 = !{!148, !155, !159, !161, !163, !165, !167, !171, !173, !175, !177, !179, !181, !183, !185, !187, !189, !191, !193, !195, !197, !199, !203, !205, !207, !209, !213, !218, !220, !222, !227, !231, !233, !235, !237, !239, !241, !243, !245, !247, !252, !256, !258, !263, !267, !269, !271, !273, !275, !277, !281, !283, !285, !289, !297, !301, !303, !305, !307, !309, !313, !315, !317, !321, !323, !325, !327, !329, !331, !333, !335, !337, !339, !343, !349, !351, !353, !357, !359, !361, !363, !365, !367, !369, !371, !375, !379, !381, !383, !388, !390, !392, !394, !396, !398, !400, !404, !410, !414, !419, !421, !425, !429, !442, !446, !450, !454, !458, !463, !465, !469, !473, !477, !485, !489, !493, !497, !501, !505, !511, !515, !519, !521, !529, !533, !540, !542, !544, !548, !552, !556, !561, !565, !570, !571, !572, !573, !575, !576, !577, !578, !579, !580, !581, !583, !584, !585, !586, !587, !591, !592, !593, !594, !595, !596, !597, !598, !599, !600, !601, !602, !603, !604, !605, !606, !607, !608, !609, !610, !611, !612, !613, !614, !615, !619, !621, !623, !625, !627, !629, !631, !633, !635, !637, !639, !641, !643, !645, !647, !649, !651, !653, !655, !657, !659, !661, !663, !665, !667, !669, !671, !673, !675, !677, !679, !681, !683, !685, !687, !689, !691, !693, !695, !697, !699, !701, !703, !705, !707, !709, !711, !713, !715, !717, !719, !721, !723, !725}
!148 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !150, file: !151, line: 223)
!149 = !DINamespace(name: "std", scope: null)
!150 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !151, file: !151, line: 53, type: !152, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!151 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/home/gkarlos")
!152 = !DISubroutineType(types: !153)
!153 = !{!154, !154}
!154 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!155 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !156, file: !151, line: 224)
!156 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !151, file: !151, line: 55, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!157 = !DISubroutineType(types: !158)
!158 = !{!126, !126}
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !160, file: !151, line: 225)
!160 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !151, file: !151, line: 57, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!161 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !162, file: !151, line: 226)
!162 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !151, file: !151, line: 59, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!163 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !164, file: !151, line: 227)
!164 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !151, file: !151, line: 61, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!165 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !166, file: !151, line: 228)
!166 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !151, file: !151, line: 65, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!167 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !168, file: !151, line: 229)
!168 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !151, file: !151, line: 63, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!169 = !DISubroutineType(types: !170)
!170 = !{!126, !126, !126}
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !172, file: !151, line: 230)
!172 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !151, file: !151, line: 67, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !174, file: !151, line: 231)
!174 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !151, file: !151, line: 69, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !176, file: !151, line: 232)
!176 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !151, file: !151, line: 71, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !178, file: !151, line: 233)
!178 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !151, file: !151, line: 73, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !180, file: !151, line: 234)
!180 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !151, file: !151, line: 75, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !182, file: !151, line: 235)
!182 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !151, file: !151, line: 77, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !184, file: !151, line: 236)
!184 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !151, file: !151, line: 81, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !186, file: !151, line: 237)
!186 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !151, file: !151, line: 79, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !188, file: !151, line: 238)
!188 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !151, file: !151, line: 85, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !190, file: !151, line: 239)
!190 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !151, file: !151, line: 83, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !192, file: !151, line: 240)
!192 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !151, file: !151, line: 87, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !194, file: !151, line: 241)
!194 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !151, file: !151, line: 89, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !196, file: !151, line: 242)
!196 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !151, file: !151, line: 91, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !198, file: !151, line: 243)
!198 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !151, file: !151, line: 93, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !200, file: !151, line: 244)
!200 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !151, file: !151, line: 95, type: !201, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!201 = !DISubroutineType(types: !202)
!202 = !{!126, !126, !126, !126}
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !204, file: !151, line: 245)
!204 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !151, file: !151, line: 97, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !206, file: !151, line: 246)
!206 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !151, file: !151, line: 99, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !208, file: !151, line: 247)
!208 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !151, file: !151, line: 101, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !210, file: !151, line: 248)
!210 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !151, file: !151, line: 103, type: !211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!211 = !DISubroutineType(types: !212)
!212 = !{!154, !126}
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !214, file: !151, line: 249)
!214 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !151, file: !151, line: 105, type: !215, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!215 = !DISubroutineType(types: !216)
!216 = !{!126, !126, !217}
!217 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !154, size: 64)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !219, file: !151, line: 250)
!219 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !151, file: !151, line: 107, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !221, file: !151, line: 251)
!221 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !151, file: !151, line: 109, type: !211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !223, file: !151, line: 252)
!223 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !151, file: !151, line: 114, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!224 = !DISubroutineType(types: !225)
!225 = !{!226, !126}
!226 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !228, file: !151, line: 253)
!228 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !151, file: !151, line: 118, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!229 = !DISubroutineType(types: !230)
!230 = !{!226, !126, !126}
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !232, file: !151, line: 254)
!232 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !151, file: !151, line: 117, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !234, file: !151, line: 255)
!234 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !151, file: !151, line: 123, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !236, file: !151, line: 256)
!236 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !151, file: !151, line: 127, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !238, file: !151, line: 257)
!238 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !151, file: !151, line: 126, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !240, file: !151, line: 258)
!240 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !151, file: !151, line: 129, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !242, file: !151, line: 259)
!242 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !151, file: !151, line: 134, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !244, file: !151, line: 260)
!244 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !151, file: !151, line: 136, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !246, file: !151, line: 261)
!246 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !151, file: !151, line: 138, type: !229, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !248, file: !151, line: 262)
!248 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !151, file: !151, line: 139, type: !249, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!249 = !DISubroutineType(types: !250)
!250 = !{!251, !251}
!251 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !253, file: !151, line: 263)
!253 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !151, file: !151, line: 141, type: !254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!254 = !DISubroutineType(types: !255)
!255 = !{!126, !126, !154}
!256 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !257, file: !151, line: 264)
!257 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !151, file: !151, line: 143, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!258 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !259, file: !151, line: 265)
!259 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !151, file: !151, line: 144, type: !260, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!260 = !DISubroutineType(types: !261)
!261 = !{!262, !262}
!262 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!263 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !264, file: !151, line: 266)
!264 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !151, file: !151, line: 146, type: !265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!265 = !DISubroutineType(types: !266)
!266 = !{!262, !126}
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !268, file: !151, line: 267)
!268 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !151, file: !151, line: 159, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!269 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !270, file: !151, line: 268)
!270 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !151, file: !151, line: 148, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!271 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !272, file: !151, line: 269)
!272 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !151, file: !151, line: 150, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!273 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !274, file: !151, line: 270)
!274 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !151, file: !151, line: 152, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!275 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !276, file: !151, line: 271)
!276 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !151, file: !151, line: 154, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!277 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !278, file: !151, line: 272)
!278 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !151, file: !151, line: 161, type: !279, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!279 = !DISubroutineType(types: !280)
!280 = !{!251, !126}
!281 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !282, file: !151, line: 273)
!282 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !151, file: !151, line: 163, type: !279, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !284, file: !151, line: 274)
!284 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !151, file: !151, line: 164, type: !265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!285 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !286, file: !151, line: 275)
!286 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !151, file: !151, line: 166, type: !287, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!287 = !DISubroutineType(types: !288)
!288 = !{!126, !126, !125}
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !290, file: !151, line: 276)
!290 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !151, file: !151, line: 167, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!291 = !DISubroutineType(types: !292)
!292 = !{!293, !294}
!293 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!294 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !295, size: 64)
!295 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !296)
!296 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!297 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !298, file: !151, line: 277)
!298 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !151, file: !151, line: 168, type: !299, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!299 = !DISubroutineType(types: !300)
!300 = !{!126, !294}
!301 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !302, file: !151, line: 278)
!302 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !151, file: !151, line: 170, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!303 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !304, file: !151, line: 279)
!304 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !151, file: !151, line: 172, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!305 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !306, file: !151, line: 280)
!306 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !151, file: !151, line: 176, type: !254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!307 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !308, file: !151, line: 281)
!308 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !151, file: !151, line: 178, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!309 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !310, file: !151, line: 282)
!310 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !151, file: !151, line: 180, type: !311, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!311 = !DISubroutineType(types: !312)
!312 = !{!126, !126, !126, !217}
!313 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !314, file: !151, line: 283)
!314 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !151, file: !151, line: 182, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !316, file: !151, line: 284)
!316 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !151, file: !151, line: 184, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!317 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !318, file: !151, line: 285)
!318 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !151, file: !151, line: 186, type: !319, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!319 = !DISubroutineType(types: !320)
!320 = !{!126, !126, !251}
!321 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !322, file: !151, line: 286)
!322 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !151, file: !151, line: 188, type: !254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !324, file: !151, line: 287)
!324 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !151, file: !151, line: 190, type: !224, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!325 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !326, file: !151, line: 288)
!326 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !151, file: !151, line: 192, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!327 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !328, file: !151, line: 289)
!328 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !151, file: !151, line: 194, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!329 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !330, file: !151, line: 290)
!330 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !151, file: !151, line: 196, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!331 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !332, file: !151, line: 291)
!332 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !151, file: !151, line: 198, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!333 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !334, file: !151, line: 292)
!334 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !151, file: !151, line: 200, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!335 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !336, file: !151, line: 293)
!336 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !151, file: !151, line: 202, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!337 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !338, file: !151, line: 294)
!338 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !151, file: !151, line: 204, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!339 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !340, file: !342, line: 52)
!340 = !DISubprogram(name: "abs", scope: !341, file: !341, line: 837, type: !152, flags: DIFlagPrototyped, spFlags: 0)
!341 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!342 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0/bits/std_abs.h", directory: "")
!343 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !344, file: !348, line: 83)
!344 = !DISubprogram(name: "acos", scope: !345, file: !345, line: 53, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!345 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!346 = !DISubroutineType(types: !347)
!347 = !{!293, !293}
!348 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0/cmath", directory: "")
!349 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !350, file: !348, line: 102)
!350 = !DISubprogram(name: "asin", scope: !345, file: !345, line: 55, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!351 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !352, file: !348, line: 121)
!352 = !DISubprogram(name: "atan", scope: !345, file: !345, line: 57, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!353 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !354, file: !348, line: 140)
!354 = !DISubprogram(name: "atan2", scope: !345, file: !345, line: 59, type: !355, flags: DIFlagPrototyped, spFlags: 0)
!355 = !DISubroutineType(types: !356)
!356 = !{!293, !293, !293}
!357 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !358, file: !348, line: 161)
!358 = !DISubprogram(name: "ceil", scope: !345, file: !345, line: 159, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!359 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !360, file: !348, line: 180)
!360 = !DISubprogram(name: "cos", scope: !345, file: !345, line: 62, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!361 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !362, file: !348, line: 199)
!362 = !DISubprogram(name: "cosh", scope: !345, file: !345, line: 71, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!363 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !364, file: !348, line: 218)
!364 = !DISubprogram(name: "exp", scope: !345, file: !345, line: 95, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!365 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !366, file: !348, line: 237)
!366 = !DISubprogram(name: "fabs", scope: !345, file: !345, line: 162, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!367 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !368, file: !348, line: 256)
!368 = !DISubprogram(name: "floor", scope: !345, file: !345, line: 165, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!369 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !370, file: !348, line: 275)
!370 = !DISubprogram(name: "fmod", scope: !345, file: !345, line: 168, type: !355, flags: DIFlagPrototyped, spFlags: 0)
!371 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !372, file: !348, line: 296)
!372 = !DISubprogram(name: "frexp", scope: !345, file: !345, line: 98, type: !373, flags: DIFlagPrototyped, spFlags: 0)
!373 = !DISubroutineType(types: !374)
!374 = !{!293, !293, !217}
!375 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !376, file: !348, line: 315)
!376 = !DISubprogram(name: "ldexp", scope: !345, file: !345, line: 101, type: !377, flags: DIFlagPrototyped, spFlags: 0)
!377 = !DISubroutineType(types: !378)
!378 = !{!293, !293, !154}
!379 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !380, file: !348, line: 334)
!380 = !DISubprogram(name: "log", scope: !345, file: !345, line: 104, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!381 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !382, file: !348, line: 353)
!382 = !DISubprogram(name: "log10", scope: !345, file: !345, line: 107, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!383 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !384, file: !348, line: 372)
!384 = !DISubprogram(name: "modf", scope: !345, file: !345, line: 110, type: !385, flags: DIFlagPrototyped, spFlags: 0)
!385 = !DISubroutineType(types: !386)
!386 = !{!293, !293, !387}
!387 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !293, size: 64)
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !389, file: !348, line: 384)
!389 = !DISubprogram(name: "pow", scope: !345, file: !345, line: 140, type: !355, flags: DIFlagPrototyped, spFlags: 0)
!390 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !391, file: !348, line: 421)
!391 = !DISubprogram(name: "sin", scope: !345, file: !345, line: 64, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !393, file: !348, line: 440)
!393 = !DISubprogram(name: "sinh", scope: !345, file: !345, line: 73, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !395, file: !348, line: 459)
!395 = !DISubprogram(name: "sqrt", scope: !345, file: !345, line: 143, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!396 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !397, file: !348, line: 478)
!397 = !DISubprogram(name: "tan", scope: !345, file: !345, line: 66, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !399, file: !348, line: 497)
!399 = !DISubprogram(name: "tanh", scope: !345, file: !345, line: 75, type: !346, flags: DIFlagPrototyped, spFlags: 0)
!400 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !401, file: !403, line: 127)
!401 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !341, line: 62, baseType: !402)
!402 = !DICompositeType(tag: DW_TAG_structure_type, file: !341, line: 58, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!403 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0/cstdlib", directory: "")
!404 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !405, file: !403, line: 128)
!405 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !341, line: 70, baseType: !406)
!406 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !341, line: 66, size: 128, flags: DIFlagTypePassByValue, elements: !407, identifier: "_ZTS6ldiv_t")
!407 = !{!408, !409}
!408 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !406, file: !341, line: 68, baseType: !251, size: 64)
!409 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !406, file: !341, line: 69, baseType: !251, size: 64, offset: 64)
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !411, file: !403, line: 130)
!411 = !DISubprogram(name: "abort", scope: !341, file: !341, line: 588, type: !412, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!412 = !DISubroutineType(types: !413)
!413 = !{null}
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !415, file: !403, line: 134)
!415 = !DISubprogram(name: "atexit", scope: !341, file: !341, line: 592, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DISubroutineType(types: !417)
!417 = !{!154, !418}
!418 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !412, size: 64)
!419 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !420, file: !403, line: 140)
!420 = !DISubprogram(name: "atof", scope: !341, file: !341, line: 101, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!421 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !422, file: !403, line: 141)
!422 = !DISubprogram(name: "atoi", scope: !341, file: !341, line: 104, type: !423, flags: DIFlagPrototyped, spFlags: 0)
!423 = !DISubroutineType(types: !424)
!424 = !{!154, !294}
!425 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !426, file: !403, line: 142)
!426 = !DISubprogram(name: "atol", scope: !341, file: !341, line: 107, type: !427, flags: DIFlagPrototyped, spFlags: 0)
!427 = !DISubroutineType(types: !428)
!428 = !{!251, !294}
!429 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !430, file: !403, line: 143)
!430 = !DISubprogram(name: "bsearch", scope: !341, file: !341, line: 817, type: !431, flags: DIFlagPrototyped, spFlags: 0)
!431 = !DISubroutineType(types: !432)
!432 = !{!128, !433, !433, !435, !435, !438}
!433 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !434, size: 64)
!434 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!435 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !436, line: 46, baseType: !437)
!436 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/stddef.h", directory: "/home/gkarlos")
!437 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!438 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !341, line: 805, baseType: !439)
!439 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !440, size: 64)
!440 = !DISubroutineType(types: !441)
!441 = !{!154, !433, !433}
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !443, file: !403, line: 144)
!443 = !DISubprogram(name: "calloc", scope: !341, file: !341, line: 541, type: !444, flags: DIFlagPrototyped, spFlags: 0)
!444 = !DISubroutineType(types: !445)
!445 = !{!128, !435, !435}
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !447, file: !403, line: 145)
!447 = !DISubprogram(name: "div", scope: !341, file: !341, line: 849, type: !448, flags: DIFlagPrototyped, spFlags: 0)
!448 = !DISubroutineType(types: !449)
!449 = !{!401, !154, !154}
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !451, file: !403, line: 146)
!451 = !DISubprogram(name: "exit", scope: !341, file: !341, line: 614, type: !452, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!452 = !DISubroutineType(types: !453)
!453 = !{null, !154}
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !455, file: !403, line: 147)
!455 = !DISubprogram(name: "free", scope: !341, file: !341, line: 563, type: !456, flags: DIFlagPrototyped, spFlags: 0)
!456 = !DISubroutineType(types: !457)
!457 = !{null, !128}
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !459, file: !403, line: 148)
!459 = !DISubprogram(name: "getenv", scope: !341, file: !341, line: 631, type: !460, flags: DIFlagPrototyped, spFlags: 0)
!460 = !DISubroutineType(types: !461)
!461 = !{!462, !294}
!462 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !296, size: 64)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !464, file: !403, line: 149)
!464 = !DISubprogram(name: "labs", scope: !341, file: !341, line: 838, type: !249, flags: DIFlagPrototyped, spFlags: 0)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !466, file: !403, line: 150)
!466 = !DISubprogram(name: "ldiv", scope: !341, file: !341, line: 851, type: !467, flags: DIFlagPrototyped, spFlags: 0)
!467 = !DISubroutineType(types: !468)
!468 = !{!405, !251, !251}
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !470, file: !403, line: 151)
!470 = !DISubprogram(name: "malloc", scope: !341, file: !341, line: 539, type: !471, flags: DIFlagPrototyped, spFlags: 0)
!471 = !DISubroutineType(types: !472)
!472 = !{!128, !435}
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !474, file: !403, line: 153)
!474 = !DISubprogram(name: "mblen", scope: !341, file: !341, line: 919, type: !475, flags: DIFlagPrototyped, spFlags: 0)
!475 = !DISubroutineType(types: !476)
!476 = !{!154, !294, !435}
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !478, file: !403, line: 154)
!478 = !DISubprogram(name: "mbstowcs", scope: !341, file: !341, line: 930, type: !479, flags: DIFlagPrototyped, spFlags: 0)
!479 = !DISubroutineType(types: !480)
!480 = !{!435, !481, !484, !435}
!481 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !482)
!482 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !483, size: 64)
!483 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!484 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !294)
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !486, file: !403, line: 155)
!486 = !DISubprogram(name: "mbtowc", scope: !341, file: !341, line: 922, type: !487, flags: DIFlagPrototyped, spFlags: 0)
!487 = !DISubroutineType(types: !488)
!488 = !{!154, !481, !484, !435}
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !490, file: !403, line: 157)
!490 = !DISubprogram(name: "qsort", scope: !341, file: !341, line: 827, type: !491, flags: DIFlagPrototyped, spFlags: 0)
!491 = !DISubroutineType(types: !492)
!492 = !{null, !128, !435, !435, !438}
!493 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !494, file: !403, line: 163)
!494 = !DISubprogram(name: "rand", scope: !341, file: !341, line: 453, type: !495, flags: DIFlagPrototyped, spFlags: 0)
!495 = !DISubroutineType(types: !496)
!496 = !{!154}
!497 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !498, file: !403, line: 164)
!498 = !DISubprogram(name: "realloc", scope: !341, file: !341, line: 549, type: !499, flags: DIFlagPrototyped, spFlags: 0)
!499 = !DISubroutineType(types: !500)
!500 = !{!128, !128, !435}
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !502, file: !403, line: 165)
!502 = !DISubprogram(name: "srand", scope: !341, file: !341, line: 455, type: !503, flags: DIFlagPrototyped, spFlags: 0)
!503 = !DISubroutineType(types: !504)
!504 = !{null, !7}
!505 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !506, file: !403, line: 166)
!506 = !DISubprogram(name: "strtod", scope: !341, file: !341, line: 117, type: !507, flags: DIFlagPrototyped, spFlags: 0)
!507 = !DISubroutineType(types: !508)
!508 = !{!293, !484, !509}
!509 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !510)
!510 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !462, size: 64)
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !512, file: !403, line: 167)
!512 = !DISubprogram(name: "strtol", scope: !341, file: !341, line: 176, type: !513, flags: DIFlagPrototyped, spFlags: 0)
!513 = !DISubroutineType(types: !514)
!514 = !{!251, !484, !509, !154}
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !516, file: !403, line: 168)
!516 = !DISubprogram(name: "strtoul", scope: !341, file: !341, line: 180, type: !517, flags: DIFlagPrototyped, spFlags: 0)
!517 = !DISubroutineType(types: !518)
!518 = !{!437, !484, !509, !154}
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !520, file: !403, line: 169)
!520 = !DISubprogram(name: "system", scope: !341, file: !341, line: 781, type: !423, flags: DIFlagPrototyped, spFlags: 0)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !522, file: !403, line: 171)
!522 = !DISubprogram(name: "wcstombs", scope: !341, file: !341, line: 933, type: !523, flags: DIFlagPrototyped, spFlags: 0)
!523 = !DISubroutineType(types: !524)
!524 = !{!435, !525, !526, !435}
!525 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !462)
!526 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !527)
!527 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !528, size: 64)
!528 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !483)
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !530, file: !403, line: 172)
!530 = !DISubprogram(name: "wctomb", scope: !341, file: !341, line: 926, type: !531, flags: DIFlagPrototyped, spFlags: 0)
!531 = !DISubroutineType(types: !532)
!532 = !{!154, !462, !483}
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !535, file: !403, line: 200)
!534 = !DINamespace(name: "__gnu_cxx", scope: null)
!535 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !341, line: 80, baseType: !536)
!536 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !341, line: 76, size: 128, flags: DIFlagTypePassByValue, elements: !537, identifier: "_ZTS7lldiv_t")
!537 = !{!538, !539}
!538 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !536, file: !341, line: 78, baseType: !262, size: 64)
!539 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !536, file: !341, line: 79, baseType: !262, size: 64, offset: 64)
!540 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !541, file: !403, line: 206)
!541 = !DISubprogram(name: "_Exit", scope: !341, file: !341, line: 626, type: !452, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!542 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !543, file: !403, line: 210)
!543 = !DISubprogram(name: "llabs", scope: !341, file: !341, line: 841, type: !260, flags: DIFlagPrototyped, spFlags: 0)
!544 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !545, file: !403, line: 216)
!545 = !DISubprogram(name: "lldiv", scope: !341, file: !341, line: 855, type: !546, flags: DIFlagPrototyped, spFlags: 0)
!546 = !DISubroutineType(types: !547)
!547 = !{!535, !262, !262}
!548 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !549, file: !403, line: 227)
!549 = !DISubprogram(name: "atoll", scope: !341, file: !341, line: 112, type: !550, flags: DIFlagPrototyped, spFlags: 0)
!550 = !DISubroutineType(types: !551)
!551 = !{!262, !294}
!552 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !553, file: !403, line: 228)
!553 = !DISubprogram(name: "strtoll", scope: !341, file: !341, line: 200, type: !554, flags: DIFlagPrototyped, spFlags: 0)
!554 = !DISubroutineType(types: !555)
!555 = !{!262, !484, !509, !154}
!556 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !557, file: !403, line: 229)
!557 = !DISubprogram(name: "strtoull", scope: !341, file: !341, line: 205, type: !558, flags: DIFlagPrototyped, spFlags: 0)
!558 = !DISubroutineType(types: !559)
!559 = !{!560, !484, !509, !154}
!560 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !562, file: !403, line: 231)
!562 = !DISubprogram(name: "strtof", scope: !341, file: !341, line: 123, type: !563, flags: DIFlagPrototyped, spFlags: 0)
!563 = !DISubroutineType(types: !564)
!564 = !{!126, !484, !509}
!565 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !534, entity: !566, file: !403, line: 232)
!566 = !DISubprogram(name: "strtold", scope: !341, file: !341, line: 126, type: !567, flags: DIFlagPrototyped, spFlags: 0)
!567 = !DISubroutineType(types: !568)
!568 = !{!569, !484, !509}
!569 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !535, file: !403, line: 240)
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !541, file: !403, line: 242)
!572 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !543, file: !403, line: 244)
!573 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !574, file: !403, line: 245)
!574 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !534, file: !403, line: 213, type: !546, flags: DIFlagPrototyped, spFlags: 0)
!575 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !545, file: !403, line: 246)
!576 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !549, file: !403, line: 248)
!577 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !562, file: !403, line: 249)
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !553, file: !403, line: 250)
!579 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !557, file: !403, line: 251)
!580 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !566, file: !403, line: 252)
!581 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !411, file: !582, line: 38)
!582 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/7.4.0/../../../../include/c++/7.4.0/stdlib.h", directory: "")
!583 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !415, file: !582, line: 39)
!584 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !451, file: !582, line: 40)
!585 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !401, file: !582, line: 51)
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !405, file: !582, line: 52)
!587 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !588, file: !582, line: 54)
!588 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !149, file: !342, line: 78, type: !589, flags: DIFlagPrototyped, spFlags: 0)
!589 = !DISubroutineType(types: !590)
!590 = !{!569, !569}
!591 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !420, file: !582, line: 55)
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !422, file: !582, line: 56)
!593 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !426, file: !582, line: 57)
!594 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !430, file: !582, line: 58)
!595 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !443, file: !582, line: 59)
!596 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !574, file: !582, line: 60)
!597 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !455, file: !582, line: 61)
!598 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !459, file: !582, line: 62)
!599 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !464, file: !582, line: 63)
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !466, file: !582, line: 64)
!601 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !470, file: !582, line: 65)
!602 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !474, file: !582, line: 67)
!603 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !478, file: !582, line: 68)
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !486, file: !582, line: 69)
!605 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !490, file: !582, line: 71)
!606 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !494, file: !582, line: 72)
!607 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !498, file: !582, line: 73)
!608 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !502, file: !582, line: 74)
!609 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !506, file: !582, line: 75)
!610 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !512, file: !582, line: 76)
!611 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !516, file: !582, line: 77)
!612 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !520, file: !582, line: 78)
!613 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !522, file: !582, line: 80)
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !530, file: !582, line: 81)
!615 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !616, file: !618, line: 414)
!616 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !617, file: !617, line: 1489, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!617 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_device_functions.h", directory: "/home/gkarlos")
!618 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_cmath.h", directory: "/home/gkarlos")
!619 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !620, file: !618, line: 415)
!620 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !617, file: !617, line: 1491, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!621 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !622, file: !618, line: 416)
!622 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !617, file: !617, line: 1493, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!623 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !624, file: !618, line: 417)
!624 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !617, file: !617, line: 1495, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!625 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !626, file: !618, line: 418)
!626 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !617, file: !617, line: 1498, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!627 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !628, file: !618, line: 419)
!628 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !617, file: !617, line: 1499, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!629 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !630, file: !618, line: 420)
!630 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !617, file: !617, line: 1501, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!631 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !632, file: !618, line: 421)
!632 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !617, file: !617, line: 1503, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !634, file: !618, line: 422)
!634 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !617, file: !617, line: 1505, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!635 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !636, file: !618, line: 423)
!636 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !617, file: !617, line: 1513, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!637 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !638, file: !618, line: 424)
!638 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !617, file: !617, line: 1517, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!639 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !640, file: !618, line: 425)
!640 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !617, file: !617, line: 1521, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !642, file: !618, line: 426)
!642 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !617, file: !617, line: 1530, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!643 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !644, file: !618, line: 427)
!644 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !617, file: !617, line: 1535, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!645 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !646, file: !618, line: 428)
!646 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !617, file: !617, line: 1542, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!647 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !648, file: !618, line: 429)
!648 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !617, file: !617, line: 1543, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !650, file: !618, line: 430)
!650 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !617, file: !617, line: 1545, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!651 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !652, file: !618, line: 431)
!652 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !617, file: !617, line: 1546, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!653 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !654, file: !618, line: 432)
!654 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !617, file: !617, line: 1548, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!655 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !656, file: !618, line: 433)
!656 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !617, file: !617, line: 1558, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!657 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !658, file: !618, line: 434)
!658 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !617, file: !617, line: 1562, type: !201, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!659 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !660, file: !618, line: 435)
!660 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !617, file: !617, line: 1566, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!661 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !662, file: !618, line: 436)
!662 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !617, file: !617, line: 1568, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!663 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !664, file: !618, line: 437)
!664 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !617, file: !617, line: 1570, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!665 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !666, file: !618, line: 438)
!666 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !617, file: !617, line: 1572, type: !215, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!667 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !668, file: !618, line: 439)
!668 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !617, file: !617, line: 1574, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!669 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !670, file: !618, line: 440)
!670 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !617, file: !617, line: 1576, type: !211, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!671 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !672, file: !618, line: 441)
!672 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !617, file: !617, line: 1589, type: !254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!673 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !674, file: !618, line: 442)
!674 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !617, file: !617, line: 1591, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!675 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !676, file: !618, line: 443)
!676 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !617, file: !617, line: 1600, type: !265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!677 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !678, file: !618, line: 444)
!678 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !617, file: !617, line: 1602, type: !265, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!679 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !680, file: !618, line: 445)
!680 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !617, file: !617, line: 1605, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!681 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !682, file: !618, line: 446)
!682 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !617, file: !617, line: 1607, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!683 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !684, file: !618, line: 447)
!684 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !617, file: !617, line: 1609, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!685 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !686, file: !618, line: 448)
!686 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !617, file: !617, line: 1613, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!687 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !688, file: !618, line: 449)
!688 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !617, file: !617, line: 1614, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!689 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !690, file: !618, line: 450)
!690 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !617, file: !617, line: 1619, type: !279, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!691 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !692, file: !618, line: 451)
!692 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !617, file: !617, line: 1621, type: !279, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!693 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !694, file: !618, line: 452)
!694 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !617, file: !617, line: 1641, type: !287, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!695 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !696, file: !618, line: 453)
!696 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !617, file: !617, line: 1643, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!697 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !698, file: !618, line: 454)
!698 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !617, file: !617, line: 1647, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!699 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !700, file: !618, line: 455)
!700 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !617, file: !617, line: 1673, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !702, file: !618, line: 456)
!702 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !617, file: !617, line: 1681, type: !169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!703 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !704, file: !618, line: 457)
!704 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !617, file: !617, line: 1687, type: !311, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!705 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !706, file: !618, line: 458)
!706 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !617, file: !617, line: 1697, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!707 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !708, file: !618, line: 459)
!708 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !617, file: !617, line: 1717, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!709 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !710, file: !618, line: 462)
!710 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !617, file: !617, line: 1731, type: !319, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!711 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !712, file: !618, line: 464)
!712 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !617, file: !617, line: 1721, type: !254, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!713 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !714, file: !618, line: 465)
!714 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !617, file: !617, line: 1752, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !716, file: !618, line: 466)
!716 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !617, file: !617, line: 1756, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!717 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !718, file: !618, line: 467)
!718 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !617, file: !617, line: 1760, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!719 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !720, file: !618, line: 468)
!720 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !617, file: !617, line: 1762, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!721 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !722, file: !618, line: 469)
!722 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !617, file: !617, line: 1764, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !724, file: !618, line: 470)
!724 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !617, file: !617, line: 1766, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!725 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !149, entity: !726, file: !618, line: 471)
!726 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !617, file: !617, line: 1768, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!727 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!728 = !{i32 2, !"Dwarf Version", i32 4}
!729 = !{i32 2, !"Debug Info Version", i32 3}
!730 = !{i32 1, !"wchar_size", i32 4}
!731 = !{!"clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)"}
!732 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 44, type: !733, scopeLine: 45, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!733 = !DISubroutineType(types: !734)
!734 = !{!154, !154, !510}
!735 = !{}
!736 = !DILocalVariable(name: "argc", arg: 1, scope: !732, file: !3, line: 44, type: !154)
!737 = !DILocation(line: 44, column: 14, scope: !732)
!738 = !DILocalVariable(name: "argv", arg: 2, scope: !732, file: !3, line: 44, type: !510)
!739 = !DILocation(line: 44, column: 26, scope: !732)
!740 = !DILocalVariable(name: "verbose", scope: !732, file: !3, line: 46, type: !154)
!741 = !DILocation(line: 46, column: 9, scope: !732)
!742 = !DILocation(line: 47, column: 9, scope: !743)
!743 = distinct !DILexicalBlock(scope: !732, file: !3, line: 47, column: 9)
!744 = !DILocation(line: 47, column: 14, scope: !743)
!745 = !DILocation(line: 47, column: 9, scope: !732)
!746 = !DILocation(line: 48, column: 9, scope: !747)
!747 = distinct !DILexicalBlock(scope: !743, file: !3, line: 47, column: 19)
!748 = !DILocation(line: 49, column: 9, scope: !747)
!749 = !DILocation(line: 50, column: 9, scope: !747)
!750 = !DILocation(line: 51, column: 9, scope: !747)
!751 = !DILocation(line: 52, column: 9, scope: !747)
!752 = !DILocation(line: 53, column: 9, scope: !747)
!753 = !DILocation(line: 54, column: 9, scope: !747)
!754 = !DILocation(line: 55, column: 9, scope: !747)
!755 = !DILocation(line: 56, column: 9, scope: !747)
!756 = !DILocation(line: 57, column: 9, scope: !747)
!757 = !DILocation(line: 58, column: 9, scope: !747)
!758 = !DILocation(line: 59, column: 9, scope: !747)
!759 = !DILocation(line: 60, column: 9, scope: !747)
!760 = !DILocation(line: 61, column: 9, scope: !747)
!761 = !DILocation(line: 62, column: 9, scope: !747)
!762 = !DILocation(line: 63, column: 9, scope: !747)
!763 = !DILocation(line: 64, column: 9, scope: !747)
!764 = !DILocation(line: 65, column: 9, scope: !747)
!765 = !DILocation(line: 66, column: 9, scope: !747)
!766 = !DILocation(line: 67, column: 9, scope: !747)
!767 = !DILocation(line: 68, column: 9, scope: !747)
!768 = !DILocation(line: 74, column: 21, scope: !732)
!769 = !DILocation(line: 74, column: 5, scope: !732)
!770 = !DILocation(line: 75, column: 9, scope: !771)
!771 = distinct !DILexicalBlock(scope: !732, file: !3, line: 75, column: 9)
!772 = !DILocation(line: 75, column: 14, scope: !771)
!773 = !DILocation(line: 75, column: 9, scope: !732)
!774 = !DILocation(line: 76, column: 21, scope: !775)
!775 = distinct !DILexicalBlock(scope: !776, file: !3, line: 76, column: 13)
!776 = distinct !DILexicalBlock(scope: !771, file: !3, line: 75, column: 19)
!777 = !DILocation(line: 76, column: 14, scope: !775)
!778 = !DILocation(line: 76, column: 13, scope: !776)
!779 = !DILocation(line: 76, column: 44, scope: !775)
!780 = !DILocation(line: 76, column: 36, scope: !775)
!781 = !DILocation(line: 77, column: 5, scope: !776)
!782 = !DILocation(line: 79, column: 5, scope: !732)
!783 = !DILocalVariable(name: "time_start", scope: !732, file: !3, line: 81, type: !784)
!784 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !785, line: 8, size: 128, flags: DIFlagTypePassByValue, elements: !786, identifier: "_ZTS7timeval")
!785 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_timeval.h", directory: "")
!786 = !{!787, !790}
!787 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !784, file: !785, line: 10, baseType: !788, size: 64)
!788 = !DIDerivedType(tag: DW_TAG_typedef, name: "__time_t", file: !789, line: 148, baseType: !251)
!789 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types.h", directory: "")
!790 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !784, file: !785, line: 11, baseType: !791, size: 64, offset: 64)
!791 = !DIDerivedType(tag: DW_TAG_typedef, name: "__suseconds_t", file: !789, line: 150, baseType: !251)
!792 = !DILocation(line: 81, column: 20, scope: !732)
!793 = !DILocation(line: 82, column: 5, scope: !732)
!794 = !DILocation(line: 85, column: 5, scope: !732)
!795 = !DILocalVariable(name: "time_end", scope: !732, file: !3, line: 88, type: !784)
!796 = !DILocation(line: 88, column: 20, scope: !732)
!797 = !DILocation(line: 89, column: 5, scope: !732)
!798 = !DILocalVariable(name: "time_total", scope: !732, file: !3, line: 90, type: !7)
!799 = !DILocation(line: 90, column: 18, scope: !732)
!800 = !DILocation(line: 90, column: 41, scope: !732)
!801 = !DILocation(line: 90, column: 48, scope: !732)
!802 = !DILocation(line: 90, column: 69, scope: !732)
!803 = !DILocation(line: 90, column: 58, scope: !732)
!804 = !DILocation(line: 90, column: 92, scope: !732)
!805 = !DILocation(line: 90, column: 99, scope: !732)
!806 = !DILocation(line: 90, column: 122, scope: !732)
!807 = !DILocation(line: 90, column: 109, scope: !732)
!808 = !DILocation(line: 90, column: 78, scope: !732)
!809 = !DILocation(line: 90, column: 31, scope: !732)
!810 = !DILocation(line: 92, column: 9, scope: !811)
!811 = distinct !DILexicalBlock(scope: !732, file: !3, line: 92, column: 9)
!812 = !DILocation(line: 92, column: 9, scope: !732)
!813 = !DILocation(line: 93, column: 9, scope: !814)
!814 = distinct !DILexicalBlock(scope: !811, file: !3, line: 92, column: 18)
!815 = !DILocation(line: 94, column: 18, scope: !814)
!816 = !DILocation(line: 94, column: 21, scope: !814)
!817 = !DILocation(line: 94, column: 27, scope: !814)
!818 = !DILocation(line: 94, column: 9, scope: !814)
!819 = !DILocation(line: 96, column: 9, scope: !814)
!820 = !DILocation(line: 97, column: 18, scope: !814)
!821 = !DILocation(line: 97, column: 21, scope: !814)
!822 = !DILocation(line: 97, column: 27, scope: !814)
!823 = !DILocation(line: 97, column: 9, scope: !814)
!824 = !DILocation(line: 99, column: 9, scope: !814)
!825 = !DILocation(line: 100, column: 18, scope: !814)
!826 = !DILocation(line: 100, column: 21, scope: !814)
!827 = !DILocation(line: 100, column: 9, scope: !814)
!828 = !DILocation(line: 101, column: 5, scope: !814)
!829 = !DILocation(line: 102, column: 5, scope: !732)
!830 = !DILocation(line: 103, column: 9, scope: !831)
!831 = distinct !DILexicalBlock(scope: !732, file: !3, line: 103, column: 9)
!832 = !DILocation(line: 103, column: 9, scope: !732)
!833 = !DILocation(line: 104, column: 9, scope: !834)
!834 = distinct !DILexicalBlock(scope: !831, file: !3, line: 103, column: 18)
!835 = !DILocation(line: 105, column: 18, scope: !834)
!836 = !DILocation(line: 105, column: 27, scope: !834)
!837 = !DILocation(line: 105, column: 9, scope: !834)
!838 = !DILocation(line: 106, column: 5, scope: !834)
!839 = !DILocation(line: 107, column: 67, scope: !732)
!840 = !DILocation(line: 107, column: 78, scope: !732)
!841 = !DILocation(line: 107, column: 5, scope: !732)
!842 = !DILocation(line: 108, column: 47, scope: !732)
!843 = !DILocation(line: 108, column: 63, scope: !732)
!844 = !DILocation(line: 108, column: 5, scope: !732)
!845 = !DILocation(line: 113, column: 10, scope: !732)
!846 = !DILocation(line: 113, column: 5, scope: !732)
!847 = !DILocation(line: 114, column: 10, scope: !732)
!848 = !DILocation(line: 114, column: 5, scope: !732)
!849 = !DILocation(line: 115, column: 10, scope: !732)
!850 = !DILocation(line: 115, column: 5, scope: !732)
!851 = !DILocation(line: 116, column: 1, scope: !732)
!852 = distinct !DISubprogram(name: "InitProblemOnce", linkageName: "_Z15InitProblemOncePc", scope: !3, file: !3, line: 163, type: !853, scopeLine: 164, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!853 = !DISubroutineType(types: !854)
!854 = !{null, !462}
!855 = !DILocalVariable(name: "filename", arg: 1, scope: !852, file: !3, line: 163, type: !462)
!856 = !DILocation(line: 163, column: 28, scope: !852)
!857 = !DILocation(line: 171, column: 13, scope: !852)
!858 = !DILocation(line: 171, column: 7, scope: !852)
!859 = !DILocation(line: 171, column: 5, scope: !852)
!860 = !DILocation(line: 173, column: 9, scope: !852)
!861 = !DILocation(line: 173, column: 2, scope: !852)
!862 = !DILocation(line: 175, column: 23, scope: !852)
!863 = !DILocation(line: 175, column: 30, scope: !852)
!864 = !DILocation(line: 175, column: 28, scope: !852)
!865 = !DILocation(line: 175, column: 35, scope: !852)
!866 = !DILocation(line: 175, column: 16, scope: !852)
!867 = !DILocation(line: 175, column: 6, scope: !852)
!868 = !DILocation(line: 175, column: 4, scope: !852)
!869 = !DILocation(line: 177, column: 10, scope: !852)
!870 = !DILocation(line: 177, column: 13, scope: !852)
!871 = !DILocation(line: 177, column: 19, scope: !852)
!872 = !DILocation(line: 177, column: 2, scope: !852)
!873 = !DILocation(line: 180, column: 23, scope: !852)
!874 = !DILocation(line: 180, column: 28, scope: !852)
!875 = !DILocation(line: 180, column: 16, scope: !852)
!876 = !DILocation(line: 180, column: 6, scope: !852)
!877 = !DILocation(line: 180, column: 4, scope: !852)
!878 = !DILocation(line: 182, column: 10, scope: !852)
!879 = !DILocation(line: 182, column: 13, scope: !852)
!880 = !DILocation(line: 182, column: 2, scope: !852)
!881 = !DILocation(line: 186, column: 24, scope: !852)
!882 = !DILocation(line: 186, column: 31, scope: !852)
!883 = !DILocation(line: 186, column: 29, scope: !852)
!884 = !DILocation(line: 186, column: 36, scope: !852)
!885 = !DILocation(line: 186, column: 17, scope: !852)
!886 = !DILocation(line: 186, column: 7, scope: !852)
!887 = !DILocation(line: 186, column: 5, scope: !852)
!888 = !DILocation(line: 187, column: 1, scope: !852)
!889 = distinct !DISubprogram(name: "InitPerRun", linkageName: "_Z10InitPerRunv", scope: !3, file: !3, line: 194, type: !412, scopeLine: 195, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!890 = !DILocalVariable(name: "i", scope: !889, file: !3, line: 196, type: !154)
!891 = !DILocation(line: 196, column: 6, scope: !889)
!892 = !DILocation(line: 197, column: 8, scope: !893)
!893 = distinct !DILexicalBlock(scope: !889, file: !3, line: 197, column: 2)
!894 = !DILocation(line: 197, column: 7, scope: !893)
!895 = !DILocation(line: 197, column: 12, scope: !896)
!896 = distinct !DILexicalBlock(scope: !893, file: !3, line: 197, column: 2)
!897 = !DILocation(line: 197, column: 14, scope: !896)
!898 = !DILocation(line: 197, column: 19, scope: !896)
!899 = !DILocation(line: 197, column: 18, scope: !896)
!900 = !DILocation(line: 197, column: 13, scope: !896)
!901 = !DILocation(line: 197, column: 2, scope: !893)
!902 = !DILocation(line: 198, column: 6, scope: !896)
!903 = !DILocation(line: 198, column: 8, scope: !896)
!904 = !DILocation(line: 198, column: 7, scope: !896)
!905 = !DILocation(line: 198, column: 11, scope: !896)
!906 = !DILocation(line: 198, column: 4, scope: !896)
!907 = !DILocation(line: 197, column: 26, scope: !896)
!908 = !DILocation(line: 197, column: 2, scope: !896)
!909 = distinct !{!909, !901, !910}
!910 = !DILocation(line: 198, column: 13, scope: !893)
!911 = !DILocation(line: 199, column: 1, scope: !889)
!912 = distinct !DISubprogram(name: "ForwardSub", linkageName: "_Z10ForwardSubv", scope: !3, file: !3, line: 246, type: !412, scopeLine: 247, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!913 = !DILocalVariable(name: "t", scope: !912, file: !3, line: 248, type: !154)
!914 = !DILocation(line: 248, column: 6, scope: !912)
!915 = !DILocalVariable(name: "m_cuda", scope: !912, file: !3, line: 249, type: !125)
!916 = !DILocation(line: 249, column: 12, scope: !912)
!917 = !DILocalVariable(name: "a_cuda", scope: !912, file: !3, line: 249, type: !125)
!918 = !DILocation(line: 249, column: 20, scope: !912)
!919 = !DILocalVariable(name: "b_cuda", scope: !912, file: !3, line: 249, type: !125)
!920 = !DILocation(line: 249, column: 28, scope: !912)
!921 = !DILocation(line: 252, column: 13, scope: !912)
!922 = !DILocation(line: 252, column: 32, scope: !912)
!923 = !DILocation(line: 252, column: 39, scope: !912)
!924 = !DILocation(line: 252, column: 37, scope: !912)
!925 = !DILocation(line: 252, column: 44, scope: !912)
!926 = !DILocation(line: 252, column: 2, scope: !912)
!927 = !DILocation(line: 254, column: 13, scope: !912)
!928 = !DILocation(line: 254, column: 32, scope: !912)
!929 = !DILocation(line: 254, column: 39, scope: !912)
!930 = !DILocation(line: 254, column: 37, scope: !912)
!931 = !DILocation(line: 254, column: 44, scope: !912)
!932 = !DILocation(line: 254, column: 2, scope: !912)
!933 = !DILocation(line: 256, column: 13, scope: !912)
!934 = !DILocation(line: 256, column: 32, scope: !912)
!935 = !DILocation(line: 256, column: 37, scope: !912)
!936 = !DILocation(line: 256, column: 2, scope: !912)
!937 = !DILocation(line: 259, column: 13, scope: !912)
!938 = !DILocation(line: 259, column: 21, scope: !912)
!939 = !DILocation(line: 259, column: 24, scope: !912)
!940 = !DILocation(line: 259, column: 31, scope: !912)
!941 = !DILocation(line: 259, column: 29, scope: !912)
!942 = !DILocation(line: 259, column: 36, scope: !912)
!943 = !DILocation(line: 259, column: 2, scope: !912)
!944 = !DILocation(line: 260, column: 13, scope: !912)
!945 = !DILocation(line: 260, column: 21, scope: !912)
!946 = !DILocation(line: 260, column: 24, scope: !912)
!947 = !DILocation(line: 260, column: 31, scope: !912)
!948 = !DILocation(line: 260, column: 29, scope: !912)
!949 = !DILocation(line: 260, column: 36, scope: !912)
!950 = !DILocation(line: 260, column: 2, scope: !912)
!951 = !DILocation(line: 261, column: 13, scope: !912)
!952 = !DILocation(line: 261, column: 21, scope: !912)
!953 = !DILocation(line: 261, column: 24, scope: !912)
!954 = !DILocation(line: 261, column: 29, scope: !912)
!955 = !DILocation(line: 261, column: 2, scope: !912)
!956 = !DILocalVariable(name: "block_size", scope: !912, file: !3, line: 263, type: !154)
!957 = !DILocation(line: 263, column: 6, scope: !912)
!958 = !DILocalVariable(name: "grid_size", scope: !912, file: !3, line: 263, type: !154)
!959 = !DILocation(line: 263, column: 17, scope: !912)
!960 = !DILocation(line: 265, column: 13, scope: !912)
!961 = !DILocation(line: 266, column: 15, scope: !912)
!962 = !DILocation(line: 266, column: 20, scope: !912)
!963 = !DILocation(line: 266, column: 19, scope: !912)
!964 = !DILocation(line: 266, column: 37, scope: !912)
!965 = !DILocation(line: 266, column: 42, scope: !912)
!966 = !DILocation(line: 266, column: 41, scope: !912)
!967 = !DILocation(line: 266, column: 36, scope: !912)
!968 = !DILocation(line: 266, column: 35, scope: !912)
!969 = !DILocation(line: 266, column: 32, scope: !912)
!970 = !DILocation(line: 266, column: 12, scope: !912)
!971 = !DILocalVariable(name: "dimBlock", scope: !912, file: !3, line: 270, type: !972)
!972 = !DIDerivedType(tag: DW_TAG_typedef, name: "dim3", file: !973, line: 430, baseType: !974)
!973 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!974 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !973, line: 416, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !975, identifier: "_ZTS4dim3")
!975 = !{!976, !977, !978, !979, !983, !992}
!976 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !974, file: !973, line: 418, baseType: !7, size: 32)
!977 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !974, file: !973, line: 418, baseType: !7, size: 32, offset: 32)
!978 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !974, file: !973, line: 418, baseType: !7, size: 32, offset: 64)
!979 = !DISubprogram(name: "dim3", scope: !974, file: !973, line: 423, type: !980, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!980 = !DISubroutineType(types: !981)
!981 = !{null, !982, !7, !7, !7}
!982 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !974, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!983 = !DISubprogram(name: "dim3", scope: !974, file: !973, line: 425, type: !984, scopeLine: 425, flags: DIFlagPrototyped, spFlags: 0)
!984 = !DISubroutineType(types: !985)
!985 = !{null, !982, !986}
!986 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !973, line: 382, baseType: !987)
!987 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !973, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !988, identifier: "_ZTS5uint3")
!988 = !{!989, !990, !991}
!989 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !987, file: !973, line: 192, baseType: !7, size: 32)
!990 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !987, file: !973, line: 192, baseType: !7, size: 32, offset: 32)
!991 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !987, file: !973, line: 192, baseType: !7, size: 32, offset: 64)
!992 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !974, file: !973, line: 426, type: !993, scopeLine: 426, flags: DIFlagPrototyped, spFlags: 0)
!993 = !DISubroutineType(types: !994)
!994 = !{!986, !982}
!995 = !DILocation(line: 270, column: 7, scope: !912)
!996 = !DILocation(line: 270, column: 16, scope: !912)
!997 = !DILocalVariable(name: "dimGrid", scope: !912, file: !3, line: 271, type: !972)
!998 = !DILocation(line: 271, column: 7, scope: !912)
!999 = !DILocation(line: 271, column: 15, scope: !912)
!1000 = !DILocalVariable(name: "blockSize2d", scope: !912, file: !3, line: 274, type: !154)
!1001 = !DILocation(line: 274, column: 6, scope: !912)
!1002 = !DILocalVariable(name: "gridSize2d", scope: !912, file: !3, line: 274, type: !154)
!1003 = !DILocation(line: 274, column: 19, scope: !912)
!1004 = !DILocation(line: 275, column: 14, scope: !912)
!1005 = !DILocation(line: 276, column: 16, scope: !912)
!1006 = !DILocation(line: 276, column: 21, scope: !912)
!1007 = !DILocation(line: 276, column: 20, scope: !912)
!1008 = !DILocation(line: 276, column: 39, scope: !912)
!1009 = !DILocation(line: 276, column: 44, scope: !912)
!1010 = !DILocation(line: 276, column: 43, scope: !912)
!1011 = !DILocation(line: 276, column: 38, scope: !912)
!1012 = !DILocation(line: 276, column: 37, scope: !912)
!1013 = !DILocation(line: 276, column: 36, scope: !912)
!1014 = !DILocation(line: 276, column: 34, scope: !912)
!1015 = !DILocation(line: 276, column: 13, scope: !912)
!1016 = !DILocalVariable(name: "dimBlockXY", scope: !912, file: !3, line: 278, type: !972)
!1017 = !DILocation(line: 278, column: 7, scope: !912)
!1018 = !DILocation(line: 278, column: 18, scope: !912)
!1019 = !DILocation(line: 278, column: 30, scope: !912)
!1020 = !DILocalVariable(name: "dimGridXY", scope: !912, file: !3, line: 279, type: !972)
!1021 = !DILocation(line: 279, column: 7, scope: !912)
!1022 = !DILocation(line: 279, column: 17, scope: !912)
!1023 = !DILocation(line: 279, column: 28, scope: !912)
!1024 = !DILocalVariable(name: "time_start", scope: !912, file: !3, line: 282, type: !784)
!1025 = !DILocation(line: 282, column: 20, scope: !912)
!1026 = !DILocation(line: 283, column: 5, scope: !912)
!1027 = !DILocation(line: 284, column: 8, scope: !1028)
!1028 = distinct !DILexicalBlock(scope: !912, file: !3, line: 284, column: 2)
!1029 = !DILocation(line: 284, column: 7, scope: !1028)
!1030 = !DILocation(line: 284, column: 12, scope: !1031)
!1031 = distinct !DILexicalBlock(scope: !1028, file: !3, line: 284, column: 2)
!1032 = !DILocation(line: 284, column: 15, scope: !1031)
!1033 = !DILocation(line: 284, column: 19, scope: !1031)
!1034 = !DILocation(line: 284, column: 13, scope: !1031)
!1035 = !DILocation(line: 284, column: 2, scope: !1028)
!1036 = !DILocation(line: 285, column: 10, scope: !1037)
!1037 = distinct !DILexicalBlock(scope: !1031, file: !3, line: 284, column: 29)
!1038 = !DILocation(line: 285, column: 18, scope: !1037)
!1039 = !DILocation(line: 285, column: 7, scope: !1037)
!1040 = !DILocation(line: 285, column: 3, scope: !1037)
!1041 = !DILocation(line: 285, column: 30, scope: !1037)
!1042 = !DILocation(line: 285, column: 37, scope: !1037)
!1043 = !DILocation(line: 285, column: 44, scope: !1037)
!1044 = !DILocation(line: 285, column: 49, scope: !1037)
!1045 = !DILocation(line: 286, column: 3, scope: !1037)
!1046 = !DILocation(line: 287, column: 10, scope: !1037)
!1047 = !DILocation(line: 287, column: 20, scope: !1037)
!1048 = !DILocation(line: 287, column: 7, scope: !1037)
!1049 = !DILocation(line: 287, column: 3, scope: !1037)
!1050 = !DILocation(line: 287, column: 34, scope: !1037)
!1051 = !DILocation(line: 287, column: 41, scope: !1037)
!1052 = !DILocation(line: 287, column: 48, scope: !1037)
!1053 = !DILocation(line: 287, column: 55, scope: !1037)
!1054 = !DILocation(line: 287, column: 60, scope: !1037)
!1055 = !DILocation(line: 287, column: 65, scope: !1037)
!1056 = !DILocation(line: 287, column: 64, scope: !1037)
!1057 = !DILocation(line: 287, column: 67, scope: !1037)
!1058 = !DILocation(line: 288, column: 3, scope: !1037)
!1059 = !DILocation(line: 289, column: 3, scope: !1037)
!1060 = !DILocation(line: 290, column: 2, scope: !1037)
!1061 = !DILocation(line: 284, column: 25, scope: !1031)
!1062 = !DILocation(line: 284, column: 2, scope: !1031)
!1063 = distinct !{!1063, !1035, !1064}
!1064 = !DILocation(line: 290, column: 2, scope: !1028)
!1065 = !DILocalVariable(name: "time_end", scope: !912, file: !3, line: 292, type: !784)
!1066 = !DILocation(line: 292, column: 17, scope: !912)
!1067 = !DILocation(line: 293, column: 5, scope: !912)
!1068 = !DILocation(line: 294, column: 33, scope: !912)
!1069 = !DILocation(line: 294, column: 40, scope: !912)
!1070 = !DILocation(line: 294, column: 61, scope: !912)
!1071 = !DILocation(line: 294, column: 50, scope: !912)
!1072 = !DILocation(line: 294, column: 84, scope: !912)
!1073 = !DILocation(line: 294, column: 91, scope: !912)
!1074 = !DILocation(line: 294, column: 114, scope: !912)
!1075 = !DILocation(line: 294, column: 101, scope: !912)
!1076 = !DILocation(line: 294, column: 70, scope: !912)
!1077 = !DILocation(line: 294, column: 23, scope: !912)
!1078 = !DILocation(line: 294, column: 21, scope: !912)
!1079 = !DILocation(line: 297, column: 13, scope: !912)
!1080 = !DILocation(line: 297, column: 16, scope: !912)
!1081 = !DILocation(line: 297, column: 24, scope: !912)
!1082 = !DILocation(line: 297, column: 31, scope: !912)
!1083 = !DILocation(line: 297, column: 29, scope: !912)
!1084 = !DILocation(line: 297, column: 36, scope: !912)
!1085 = !DILocation(line: 297, column: 2, scope: !912)
!1086 = !DILocation(line: 298, column: 13, scope: !912)
!1087 = !DILocation(line: 298, column: 16, scope: !912)
!1088 = !DILocation(line: 298, column: 24, scope: !912)
!1089 = !DILocation(line: 298, column: 31, scope: !912)
!1090 = !DILocation(line: 298, column: 29, scope: !912)
!1091 = !DILocation(line: 298, column: 36, scope: !912)
!1092 = !DILocation(line: 298, column: 2, scope: !912)
!1093 = !DILocation(line: 299, column: 13, scope: !912)
!1094 = !DILocation(line: 299, column: 16, scope: !912)
!1095 = !DILocation(line: 299, column: 24, scope: !912)
!1096 = !DILocation(line: 299, column: 29, scope: !912)
!1097 = !DILocation(line: 299, column: 2, scope: !912)
!1098 = !DILocation(line: 300, column: 11, scope: !912)
!1099 = !DILocation(line: 300, column: 2, scope: !912)
!1100 = !DILocation(line: 301, column: 11, scope: !912)
!1101 = !DILocation(line: 301, column: 2, scope: !912)
!1102 = !DILocation(line: 302, column: 11, scope: !912)
!1103 = !DILocation(line: 302, column: 2, scope: !912)
!1104 = !DILocation(line: 303, column: 1, scope: !912)
!1105 = distinct !DISubprogram(name: "PrintMat", linkageName: "_Z8PrintMatPfii", scope: !3, file: !3, line: 341, type: !1106, scopeLine: 342, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1106 = !DISubroutineType(types: !1107)
!1107 = !{null, !125, !154, !154}
!1108 = !DILocalVariable(name: "ary", arg: 1, scope: !1105, file: !3, line: 341, type: !125)
!1109 = !DILocation(line: 341, column: 22, scope: !1105)
!1110 = !DILocalVariable(name: "nrow", arg: 2, scope: !1105, file: !3, line: 341, type: !154)
!1111 = !DILocation(line: 341, column: 31, scope: !1105)
!1112 = !DILocalVariable(name: "ncol", arg: 3, scope: !1105, file: !3, line: 341, type: !154)
!1113 = !DILocation(line: 341, column: 41, scope: !1105)
!1114 = !DILocalVariable(name: "i", scope: !1105, file: !3, line: 343, type: !154)
!1115 = !DILocation(line: 343, column: 6, scope: !1105)
!1116 = !DILocalVariable(name: "j", scope: !1105, file: !3, line: 343, type: !154)
!1117 = !DILocation(line: 343, column: 9, scope: !1105)
!1118 = !DILocation(line: 345, column: 8, scope: !1119)
!1119 = distinct !DILexicalBlock(scope: !1105, file: !3, line: 345, column: 2)
!1120 = !DILocation(line: 345, column: 7, scope: !1119)
!1121 = !DILocation(line: 345, column: 12, scope: !1122)
!1122 = distinct !DILexicalBlock(scope: !1119, file: !3, line: 345, column: 2)
!1123 = !DILocation(line: 345, column: 14, scope: !1122)
!1124 = !DILocation(line: 345, column: 13, scope: !1122)
!1125 = !DILocation(line: 345, column: 2, scope: !1119)
!1126 = !DILocation(line: 346, column: 9, scope: !1127)
!1127 = distinct !DILexicalBlock(scope: !1128, file: !3, line: 346, column: 3)
!1128 = distinct !DILexicalBlock(scope: !1122, file: !3, line: 345, column: 25)
!1129 = !DILocation(line: 346, column: 8, scope: !1127)
!1130 = !DILocation(line: 346, column: 13, scope: !1131)
!1131 = distinct !DILexicalBlock(scope: !1127, file: !3, line: 346, column: 3)
!1132 = !DILocation(line: 346, column: 15, scope: !1131)
!1133 = !DILocation(line: 346, column: 14, scope: !1131)
!1134 = !DILocation(line: 346, column: 3, scope: !1127)
!1135 = !DILocation(line: 347, column: 23, scope: !1136)
!1136 = distinct !DILexicalBlock(scope: !1131, file: !3, line: 346, column: 26)
!1137 = !DILocation(line: 347, column: 27, scope: !1136)
!1138 = !DILocation(line: 347, column: 32, scope: !1136)
!1139 = !DILocation(line: 347, column: 31, scope: !1136)
!1140 = !DILocation(line: 347, column: 26, scope: !1136)
!1141 = !DILocation(line: 347, column: 34, scope: !1136)
!1142 = !DILocation(line: 347, column: 33, scope: !1136)
!1143 = !DILocation(line: 347, column: 21, scope: !1136)
!1144 = !DILocation(line: 347, column: 4, scope: !1136)
!1145 = !DILocation(line: 348, column: 3, scope: !1136)
!1146 = !DILocation(line: 346, column: 22, scope: !1131)
!1147 = !DILocation(line: 346, column: 3, scope: !1131)
!1148 = distinct !{!1148, !1134, !1149}
!1149 = !DILocation(line: 348, column: 3, scope: !1127)
!1150 = !DILocation(line: 349, column: 3, scope: !1128)
!1151 = !DILocation(line: 350, column: 2, scope: !1128)
!1152 = !DILocation(line: 345, column: 21, scope: !1122)
!1153 = !DILocation(line: 345, column: 2, scope: !1122)
!1154 = distinct !{!1154, !1125, !1155}
!1155 = !DILocation(line: 350, column: 2, scope: !1119)
!1156 = !DILocation(line: 351, column: 2, scope: !1105)
!1157 = !DILocation(line: 352, column: 1, scope: !1105)
!1158 = distinct !DISubprogram(name: "PrintAry", linkageName: "_Z8PrintAryPfi", scope: !3, file: !3, line: 372, type: !1159, scopeLine: 373, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1159 = !DISubroutineType(types: !1160)
!1160 = !{null, !125, !154}
!1161 = !DILocalVariable(name: "ary", arg: 1, scope: !1158, file: !3, line: 372, type: !125)
!1162 = !DILocation(line: 372, column: 22, scope: !1158)
!1163 = !DILocalVariable(name: "ary_size", arg: 2, scope: !1158, file: !3, line: 372, type: !154)
!1164 = !DILocation(line: 372, column: 31, scope: !1158)
!1165 = !DILocalVariable(name: "i", scope: !1158, file: !3, line: 374, type: !154)
!1166 = !DILocation(line: 374, column: 6, scope: !1158)
!1167 = !DILocation(line: 375, column: 8, scope: !1168)
!1168 = distinct !DILexicalBlock(scope: !1158, file: !3, line: 375, column: 2)
!1169 = !DILocation(line: 375, column: 7, scope: !1168)
!1170 = !DILocation(line: 375, column: 12, scope: !1171)
!1171 = distinct !DILexicalBlock(scope: !1168, file: !3, line: 375, column: 2)
!1172 = !DILocation(line: 375, column: 14, scope: !1171)
!1173 = !DILocation(line: 375, column: 13, scope: !1171)
!1174 = !DILocation(line: 375, column: 2, scope: !1168)
!1175 = !DILocation(line: 376, column: 19, scope: !1176)
!1176 = distinct !DILexicalBlock(scope: !1171, file: !3, line: 375, column: 29)
!1177 = !DILocation(line: 376, column: 23, scope: !1176)
!1178 = !DILocation(line: 376, column: 3, scope: !1176)
!1179 = !DILocation(line: 377, column: 2, scope: !1176)
!1180 = !DILocation(line: 375, column: 25, scope: !1171)
!1181 = !DILocation(line: 375, column: 2, scope: !1171)
!1182 = distinct !{!1182, !1174, !1183}
!1183 = !DILocation(line: 377, column: 2, scope: !1168)
!1184 = !DILocation(line: 378, column: 2, scope: !1158)
!1185 = !DILocation(line: 379, column: 1, scope: !1158)
!1186 = distinct !DISubprogram(name: "BackSub", linkageName: "_Z7BackSubv", scope: !3, file: !3, line: 310, type: !412, scopeLine: 311, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1187 = !DILocation(line: 313, column: 30, scope: !1186)
!1188 = !DILocation(line: 313, column: 35, scope: !1186)
!1189 = !DILocation(line: 313, column: 23, scope: !1186)
!1190 = !DILocation(line: 313, column: 13, scope: !1186)
!1191 = !DILocation(line: 313, column: 11, scope: !1186)
!1192 = !DILocalVariable(name: "i", scope: !1186, file: !3, line: 315, type: !154)
!1193 = !DILocation(line: 315, column: 6, scope: !1186)
!1194 = !DILocalVariable(name: "j", scope: !1186, file: !3, line: 315, type: !154)
!1195 = !DILocation(line: 315, column: 8, scope: !1186)
!1196 = !DILocation(line: 316, column: 7, scope: !1197)
!1197 = distinct !DILexicalBlock(scope: !1186, file: !3, line: 316, column: 2)
!1198 = !DILocation(line: 316, column: 6, scope: !1197)
!1199 = !DILocation(line: 316, column: 10, scope: !1200)
!1200 = distinct !DILexicalBlock(scope: !1197, file: !3, line: 316, column: 2)
!1201 = !DILocation(line: 316, column: 12, scope: !1200)
!1202 = !DILocation(line: 316, column: 11, scope: !1200)
!1203 = !DILocation(line: 316, column: 2, scope: !1197)
!1204 = !DILocation(line: 317, column: 22, scope: !1205)
!1205 = distinct !DILexicalBlock(scope: !1200, file: !3, line: 316, column: 21)
!1206 = !DILocation(line: 317, column: 24, scope: !1205)
!1207 = !DILocation(line: 317, column: 29, scope: !1205)
!1208 = !DILocation(line: 317, column: 28, scope: !1205)
!1209 = !DILocation(line: 317, column: 30, scope: !1205)
!1210 = !DILocation(line: 317, column: 3, scope: !1205)
!1211 = !DILocation(line: 317, column: 12, scope: !1205)
!1212 = !DILocation(line: 317, column: 17, scope: !1205)
!1213 = !DILocation(line: 317, column: 16, scope: !1205)
!1214 = !DILocation(line: 317, column: 18, scope: !1205)
!1215 = !DILocation(line: 317, column: 21, scope: !1205)
!1216 = !DILocation(line: 318, column: 8, scope: !1217)
!1217 = distinct !DILexicalBlock(scope: !1205, file: !3, line: 318, column: 3)
!1218 = !DILocation(line: 318, column: 7, scope: !1217)
!1219 = !DILocation(line: 318, column: 11, scope: !1220)
!1220 = distinct !DILexicalBlock(scope: !1217, file: !3, line: 318, column: 3)
!1221 = !DILocation(line: 318, column: 13, scope: !1220)
!1222 = !DILocation(line: 318, column: 12, scope: !1220)
!1223 = !DILocation(line: 318, column: 3, scope: !1217)
!1224 = !DILocation(line: 320, column: 26, scope: !1225)
!1225 = distinct !DILexicalBlock(scope: !1220, file: !3, line: 319, column: 3)
!1226 = !DILocation(line: 320, column: 28, scope: !1225)
!1227 = !DILocation(line: 320, column: 34, scope: !1225)
!1228 = !DILocation(line: 320, column: 39, scope: !1225)
!1229 = !DILocation(line: 320, column: 38, scope: !1225)
!1230 = !DILocation(line: 320, column: 40, scope: !1225)
!1231 = !DILocation(line: 320, column: 32, scope: !1225)
!1232 = !DILocation(line: 320, column: 27, scope: !1225)
!1233 = !DILocation(line: 320, column: 45, scope: !1225)
!1234 = !DILocation(line: 320, column: 50, scope: !1225)
!1235 = !DILocation(line: 320, column: 49, scope: !1225)
!1236 = !DILocation(line: 320, column: 51, scope: !1225)
!1237 = !DILocation(line: 320, column: 43, scope: !1225)
!1238 = !DILocation(line: 320, column: 24, scope: !1225)
!1239 = !DILocation(line: 320, column: 58, scope: !1225)
!1240 = !DILocation(line: 320, column: 67, scope: !1225)
!1241 = !DILocation(line: 320, column: 72, scope: !1225)
!1242 = !DILocation(line: 320, column: 71, scope: !1225)
!1243 = !DILocation(line: 320, column: 73, scope: !1225)
!1244 = !DILocation(line: 320, column: 56, scope: !1225)
!1245 = !DILocation(line: 320, column: 4, scope: !1225)
!1246 = !DILocation(line: 320, column: 13, scope: !1225)
!1247 = !DILocation(line: 320, column: 18, scope: !1225)
!1248 = !DILocation(line: 320, column: 17, scope: !1225)
!1249 = !DILocation(line: 320, column: 19, scope: !1225)
!1250 = !DILocation(line: 320, column: 22, scope: !1225)
!1251 = !DILocation(line: 321, column: 3, scope: !1225)
!1252 = !DILocation(line: 318, column: 16, scope: !1220)
!1253 = !DILocation(line: 318, column: 3, scope: !1220)
!1254 = distinct !{!1254, !1223, !1255}
!1255 = !DILocation(line: 321, column: 3, scope: !1217)
!1256 = !DILocation(line: 322, column: 22, scope: !1205)
!1257 = !DILocation(line: 322, column: 31, scope: !1205)
!1258 = !DILocation(line: 322, column: 36, scope: !1205)
!1259 = !DILocation(line: 322, column: 35, scope: !1205)
!1260 = !DILocation(line: 322, column: 37, scope: !1205)
!1261 = !DILocation(line: 322, column: 44, scope: !1205)
!1262 = !DILocation(line: 322, column: 46, scope: !1205)
!1263 = !DILocation(line: 322, column: 52, scope: !1205)
!1264 = !DILocation(line: 322, column: 57, scope: !1205)
!1265 = !DILocation(line: 322, column: 56, scope: !1205)
!1266 = !DILocation(line: 322, column: 58, scope: !1205)
!1267 = !DILocation(line: 322, column: 50, scope: !1205)
!1268 = !DILocation(line: 322, column: 45, scope: !1205)
!1269 = !DILocation(line: 322, column: 63, scope: !1205)
!1270 = !DILocation(line: 322, column: 68, scope: !1205)
!1271 = !DILocation(line: 322, column: 67, scope: !1205)
!1272 = !DILocation(line: 322, column: 69, scope: !1205)
!1273 = !DILocation(line: 322, column: 61, scope: !1205)
!1274 = !DILocation(line: 322, column: 42, scope: !1205)
!1275 = !DILocation(line: 322, column: 40, scope: !1205)
!1276 = !DILocation(line: 322, column: 3, scope: !1205)
!1277 = !DILocation(line: 322, column: 12, scope: !1205)
!1278 = !DILocation(line: 322, column: 17, scope: !1205)
!1279 = !DILocation(line: 322, column: 16, scope: !1205)
!1280 = !DILocation(line: 322, column: 18, scope: !1205)
!1281 = !DILocation(line: 322, column: 21, scope: !1205)
!1282 = !DILocation(line: 323, column: 2, scope: !1205)
!1283 = !DILocation(line: 316, column: 18, scope: !1200)
!1284 = !DILocation(line: 316, column: 2, scope: !1200)
!1285 = distinct !{!1285, !1203, !1286}
!1286 = !DILocation(line: 323, column: 2, scope: !1197)
!1287 = !DILocation(line: 324, column: 1, scope: !1186)
!1288 = distinct !DISubprogram(name: "PrintDeviceProperties", linkageName: "_Z21PrintDevicePropertiesv", scope: !3, file: !3, line: 121, type: !412, scopeLine: 121, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1289 = !DILocalVariable(name: "deviceProp", scope: !1288, file: !3, line: 122, type: !1290)
!1290 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !6, line: 1647, size: 5696, flags: DIFlagTypePassByValue, elements: !1291, identifier: "_ZTS14cudaDeviceProp")
!1291 = !{!1292, !1296, !1305, !1309, !1310, !1311, !1312, !1313, !1314, !1315, !1316, !1320, !1321, !1322, !1323, !1324, !1325, !1326, !1327, !1328, !1329, !1330, !1331, !1332, !1333, !1334, !1335, !1336, !1340, !1341, !1342, !1343, !1344, !1345, !1346, !1347, !1348, !1349, !1350, !1351, !1352, !1353, !1354, !1355, !1356, !1357, !1358, !1359, !1360, !1361, !1362, !1363, !1364, !1365, !1366, !1367, !1368, !1369, !1370, !1371, !1372, !1373, !1374, !1375, !1376, !1377, !1378, !1379, !1380, !1381, !1382, !1383, !1384, !1385, !1386, !1387}
!1292 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !1290, file: !6, line: 1649, baseType: !1293, size: 2048)
!1293 = !DICompositeType(tag: DW_TAG_array_type, baseType: !296, size: 2048, elements: !1294)
!1294 = !{!1295}
!1295 = !DISubrange(count: 256)
!1296 = !DIDerivedType(tag: DW_TAG_member, name: "uuid", scope: !1290, file: !6, line: 1650, baseType: !1297, size: 128, offset: 2048)
!1297 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaUUID_t", file: !6, line: 1642, baseType: !1298)
!1298 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "CUuuid_st", file: !1299, line: 278, size: 128, flags: DIFlagTypePassByValue, elements: !1300, identifier: "_ZTS9CUuuid_st")
!1299 = !DIFile(filename: "/usr/local/cuda/include/cuda.h", directory: "")
!1300 = !{!1301}
!1301 = !DIDerivedType(tag: DW_TAG_member, name: "bytes", scope: !1298, file: !1299, line: 279, baseType: !1302, size: 128)
!1302 = !DICompositeType(tag: DW_TAG_array_type, baseType: !296, size: 128, elements: !1303)
!1303 = !{!1304}
!1304 = !DISubrange(count: 16)
!1305 = !DIDerivedType(tag: DW_TAG_member, name: "luid", scope: !1290, file: !6, line: 1651, baseType: !1306, size: 64, offset: 2176)
!1306 = !DICompositeType(tag: DW_TAG_array_type, baseType: !296, size: 64, elements: !1307)
!1307 = !{!1308}
!1308 = !DISubrange(count: 8)
!1309 = !DIDerivedType(tag: DW_TAG_member, name: "luidDeviceNodeMask", scope: !1290, file: !6, line: 1652, baseType: !7, size: 32, offset: 2240)
!1310 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !1290, file: !6, line: 1653, baseType: !435, size: 64, offset: 2304)
!1311 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !1290, file: !6, line: 1654, baseType: !435, size: 64, offset: 2368)
!1312 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !1290, file: !6, line: 1655, baseType: !154, size: 32, offset: 2432)
!1313 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !1290, file: !6, line: 1656, baseType: !154, size: 32, offset: 2464)
!1314 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !1290, file: !6, line: 1657, baseType: !435, size: 64, offset: 2496)
!1315 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !1290, file: !6, line: 1658, baseType: !154, size: 32, offset: 2560)
!1316 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !1290, file: !6, line: 1659, baseType: !1317, size: 96, offset: 2592)
!1317 = !DICompositeType(tag: DW_TAG_array_type, baseType: !154, size: 96, elements: !1318)
!1318 = !{!1319}
!1319 = !DISubrange(count: 3)
!1320 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !1290, file: !6, line: 1660, baseType: !1317, size: 96, offset: 2688)
!1321 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !1290, file: !6, line: 1661, baseType: !154, size: 32, offset: 2784)
!1322 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !1290, file: !6, line: 1662, baseType: !435, size: 64, offset: 2816)
!1323 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !1290, file: !6, line: 1663, baseType: !154, size: 32, offset: 2880)
!1324 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !1290, file: !6, line: 1664, baseType: !154, size: 32, offset: 2912)
!1325 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !1290, file: !6, line: 1665, baseType: !435, size: 64, offset: 2944)
!1326 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !1290, file: !6, line: 1666, baseType: !435, size: 64, offset: 3008)
!1327 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !1290, file: !6, line: 1667, baseType: !154, size: 32, offset: 3072)
!1328 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !1290, file: !6, line: 1668, baseType: !154, size: 32, offset: 3104)
!1329 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !1290, file: !6, line: 1669, baseType: !154, size: 32, offset: 3136)
!1330 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !1290, file: !6, line: 1670, baseType: !154, size: 32, offset: 3168)
!1331 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !1290, file: !6, line: 1671, baseType: !154, size: 32, offset: 3200)
!1332 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !1290, file: !6, line: 1672, baseType: !154, size: 32, offset: 3232)
!1333 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !1290, file: !6, line: 1673, baseType: !154, size: 32, offset: 3264)
!1334 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !1290, file: !6, line: 1674, baseType: !154, size: 32, offset: 3296)
!1335 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !1290, file: !6, line: 1675, baseType: !154, size: 32, offset: 3328)
!1336 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !1290, file: !6, line: 1676, baseType: !1337, size: 64, offset: 3360)
!1337 = !DICompositeType(tag: DW_TAG_array_type, baseType: !154, size: 64, elements: !1338)
!1338 = !{!1339}
!1339 = !DISubrange(count: 2)
!1340 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !1290, file: !6, line: 1677, baseType: !1337, size: 64, offset: 3424)
!1341 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !1290, file: !6, line: 1678, baseType: !1317, size: 96, offset: 3488)
!1342 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !1290, file: !6, line: 1679, baseType: !1337, size: 64, offset: 3584)
!1343 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !1290, file: !6, line: 1680, baseType: !1317, size: 96, offset: 3648)
!1344 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !1290, file: !6, line: 1681, baseType: !1317, size: 96, offset: 3744)
!1345 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !1290, file: !6, line: 1682, baseType: !154, size: 32, offset: 3840)
!1346 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !1290, file: !6, line: 1683, baseType: !1337, size: 64, offset: 3872)
!1347 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !1290, file: !6, line: 1684, baseType: !1317, size: 96, offset: 3936)
!1348 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !1290, file: !6, line: 1685, baseType: !1337, size: 64, offset: 4032)
!1349 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !1290, file: !6, line: 1686, baseType: !154, size: 32, offset: 4096)
!1350 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !1290, file: !6, line: 1687, baseType: !1337, size: 64, offset: 4128)
!1351 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !1290, file: !6, line: 1688, baseType: !1317, size: 96, offset: 4192)
!1352 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !1290, file: !6, line: 1689, baseType: !1337, size: 64, offset: 4288)
!1353 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !1290, file: !6, line: 1690, baseType: !1317, size: 96, offset: 4352)
!1354 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !1290, file: !6, line: 1691, baseType: !154, size: 32, offset: 4448)
!1355 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !1290, file: !6, line: 1692, baseType: !1337, size: 64, offset: 4480)
!1356 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !1290, file: !6, line: 1693, baseType: !435, size: 64, offset: 4544)
!1357 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !1290, file: !6, line: 1694, baseType: !154, size: 32, offset: 4608)
!1358 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !1290, file: !6, line: 1695, baseType: !154, size: 32, offset: 4640)
!1359 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !1290, file: !6, line: 1696, baseType: !154, size: 32, offset: 4672)
!1360 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !1290, file: !6, line: 1697, baseType: !154, size: 32, offset: 4704)
!1361 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !1290, file: !6, line: 1698, baseType: !154, size: 32, offset: 4736)
!1362 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !1290, file: !6, line: 1699, baseType: !154, size: 32, offset: 4768)
!1363 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !1290, file: !6, line: 1700, baseType: !154, size: 32, offset: 4800)
!1364 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !1290, file: !6, line: 1701, baseType: !154, size: 32, offset: 4832)
!1365 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !1290, file: !6, line: 1702, baseType: !154, size: 32, offset: 4864)
!1366 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !1290, file: !6, line: 1703, baseType: !154, size: 32, offset: 4896)
!1367 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !1290, file: !6, line: 1704, baseType: !154, size: 32, offset: 4928)
!1368 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !1290, file: !6, line: 1705, baseType: !154, size: 32, offset: 4960)
!1369 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !1290, file: !6, line: 1706, baseType: !154, size: 32, offset: 4992)
!1370 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !1290, file: !6, line: 1707, baseType: !154, size: 32, offset: 5024)
!1371 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !1290, file: !6, line: 1708, baseType: !154, size: 32, offset: 5056)
!1372 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !1290, file: !6, line: 1709, baseType: !435, size: 64, offset: 5120)
!1373 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !1290, file: !6, line: 1710, baseType: !154, size: 32, offset: 5184)
!1374 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !1290, file: !6, line: 1711, baseType: !154, size: 32, offset: 5216)
!1375 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !1290, file: !6, line: 1712, baseType: !154, size: 32, offset: 5248)
!1376 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !1290, file: !6, line: 1713, baseType: !154, size: 32, offset: 5280)
!1377 = !DIDerivedType(tag: DW_TAG_member, name: "hostNativeAtomicSupported", scope: !1290, file: !6, line: 1714, baseType: !154, size: 32, offset: 5312)
!1378 = !DIDerivedType(tag: DW_TAG_member, name: "singleToDoublePrecisionPerfRatio", scope: !1290, file: !6, line: 1715, baseType: !154, size: 32, offset: 5344)
!1379 = !DIDerivedType(tag: DW_TAG_member, name: "pageableMemoryAccess", scope: !1290, file: !6, line: 1716, baseType: !154, size: 32, offset: 5376)
!1380 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentManagedAccess", scope: !1290, file: !6, line: 1717, baseType: !154, size: 32, offset: 5408)
!1381 = !DIDerivedType(tag: DW_TAG_member, name: "computePreemptionSupported", scope: !1290, file: !6, line: 1718, baseType: !154, size: 32, offset: 5440)
!1382 = !DIDerivedType(tag: DW_TAG_member, name: "canUseHostPointerForRegisteredMem", scope: !1290, file: !6, line: 1719, baseType: !154, size: 32, offset: 5472)
!1383 = !DIDerivedType(tag: DW_TAG_member, name: "cooperativeLaunch", scope: !1290, file: !6, line: 1720, baseType: !154, size: 32, offset: 5504)
!1384 = !DIDerivedType(tag: DW_TAG_member, name: "cooperativeMultiDeviceLaunch", scope: !1290, file: !6, line: 1721, baseType: !154, size: 32, offset: 5536)
!1385 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlockOptin", scope: !1290, file: !6, line: 1722, baseType: !435, size: 64, offset: 5568)
!1386 = !DIDerivedType(tag: DW_TAG_member, name: "pageableMemoryAccessUsesHostPageTables", scope: !1290, file: !6, line: 1723, baseType: !154, size: 32, offset: 5632)
!1387 = !DIDerivedType(tag: DW_TAG_member, name: "directManagedMemAccessFromHost", scope: !1290, file: !6, line: 1724, baseType: !154, size: 32, offset: 5664)
!1388 = !DILocation(line: 122, column: 17, scope: !1288)
!1389 = !DILocalVariable(name: "nDevCount", scope: !1288, file: !3, line: 123, type: !154)
!1390 = !DILocation(line: 123, column: 6, scope: !1288)
!1391 = !DILocation(line: 125, column: 2, scope: !1288)
!1392 = !DILocation(line: 126, column: 36, scope: !1288)
!1393 = !DILocation(line: 126, column: 2, scope: !1288)
!1394 = !DILocalVariable(name: "nDeviceIdx", scope: !1395, file: !3, line: 127, type: !154)
!1395 = distinct !DILexicalBlock(scope: !1288, file: !3, line: 127, column: 2)
!1396 = !DILocation(line: 127, column: 11, scope: !1395)
!1397 = !DILocation(line: 127, column: 7, scope: !1395)
!1398 = !DILocation(line: 127, column: 27, scope: !1399)
!1399 = distinct !DILexicalBlock(scope: !1395, file: !3, line: 127, column: 2)
!1400 = !DILocation(line: 127, column: 40, scope: !1399)
!1401 = !DILocation(line: 127, column: 38, scope: !1399)
!1402 = !DILocation(line: 127, column: 2, scope: !1395)
!1403 = !DILocation(line: 129, column: 6, scope: !1404)
!1404 = distinct !DILexicalBlock(scope: !1399, file: !3, line: 128, column: 2)
!1405 = !DILocation(line: 130, column: 62, scope: !1406)
!1406 = distinct !DILexicalBlock(scope: !1404, file: !3, line: 130, column: 10)
!1407 = !DILocation(line: 130, column: 25, scope: !1406)
!1408 = !DILocation(line: 130, column: 22, scope: !1406)
!1409 = !DILocation(line: 130, column: 10, scope: !1404)
!1410 = !DILocation(line: 132, column: 52, scope: !1411)
!1411 = distinct !DILexicalBlock(scope: !1406, file: !3, line: 131, column: 10)
!1412 = !DILocation(line: 132, column: 41, scope: !1411)
!1413 = !DILocation(line: 132, column: 5, scope: !1411)
!1414 = !DILocation(line: 133, column: 8, scope: !1411)
!1415 = !DILocation(line: 134, column: 67, scope: !1411)
!1416 = !DILocation(line: 134, column: 81, scope: !1411)
!1417 = !DILocation(line: 134, column: 8, scope: !1411)
!1418 = !DILocation(line: 135, column: 78, scope: !1411)
!1419 = !DILocation(line: 135, column: 95, scope: !1411)
!1420 = !DILocation(line: 135, column: 8, scope: !1411)
!1421 = !DILocation(line: 136, column: 77, scope: !1411)
!1422 = !DILocation(line: 136, column: 8, scope: !1411)
!1423 = !DILocation(line: 137, column: 65, scope: !1411)
!1424 = !DILocation(line: 137, column: 8, scope: !1411)
!1425 = !DILocation(line: 138, column: 66, scope: !1411)
!1426 = !DILocation(line: 138, column: 8, scope: !1411)
!1427 = !DILocation(line: 139, column: 68, scope: !1411)
!1428 = !DILocation(line: 139, column: 8, scope: !1411)
!1429 = !DILocation(line: 140, column: 79, scope: !1411)
!1430 = !DILocation(line: 140, column: 68, scope: !1411)
!1431 = !DILocation(line: 140, column: 108, scope: !1411)
!1432 = !DILocation(line: 140, column: 97, scope: !1411)
!1433 = !DILocation(line: 140, column: 137, scope: !1411)
!1434 = !DILocation(line: 140, column: 126, scope: !1411)
!1435 = !DILocation(line: 140, column: 8, scope: !1411)
!1436 = !DILocation(line: 141, column: 78, scope: !1411)
!1437 = !DILocation(line: 141, column: 67, scope: !1411)
!1438 = !DILocation(line: 141, column: 105, scope: !1411)
!1439 = !DILocation(line: 141, column: 94, scope: !1411)
!1440 = !DILocation(line: 141, column: 132, scope: !1411)
!1441 = !DILocation(line: 141, column: 121, scope: !1411)
!1442 = !DILocation(line: 141, column: 8, scope: !1411)
!1443 = !DILocation(line: 142, column: 73, scope: !1411)
!1444 = !DILocation(line: 142, column: 8, scope: !1411)
!1445 = !DILocation(line: 143, column: 58, scope: !1411)
!1446 = !DILocation(line: 143, column: 76, scope: !1411)
!1447 = !DILocation(line: 143, column: 8, scope: !1411)
!1448 = !DILocation(line: 144, column: 61, scope: !1411)
!1449 = !DILocation(line: 144, column: 8, scope: !1411)
!1450 = !DILocation(line: 145, column: 69, scope: !1411)
!1451 = !DILocation(line: 145, column: 8, scope: !1411)
!1452 = !DILocation(line: 146, column: 62, scope: !1411)
!1453 = !DILocation(line: 146, column: 50, scope: !1411)
!1454 = !DILocation(line: 146, column: 8, scope: !1411)
!1455 = !DILocation(line: 147, column: 73, scope: !1411)
!1456 = !DILocation(line: 147, column: 8, scope: !1411)
!1457 = !DILocation(line: 148, column: 4, scope: !1411)
!1458 = !DILocation(line: 150, column: 45, scope: !1406)
!1459 = !DILocation(line: 150, column: 26, scope: !1406)
!1460 = !DILocation(line: 150, column: 10, scope: !1406)
!1461 = !DILocation(line: 151, column: 2, scope: !1404)
!1462 = !DILocation(line: 127, column: 51, scope: !1399)
!1463 = !DILocation(line: 127, column: 2, scope: !1399)
!1464 = distinct !{!1464, !1402, !1465}
!1465 = !DILocation(line: 151, column: 2, scope: !1395)
!1466 = !DILocation(line: 152, column: 1, scope: !1288)
!1467 = distinct !DISubprogram(name: "InitMat", linkageName: "_Z7InitMatPfii", scope: !3, file: !3, line: 326, type: !1106, scopeLine: 327, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1468 = !DILocalVariable(name: "ary", arg: 1, scope: !1467, file: !3, line: 326, type: !125)
!1469 = !DILocation(line: 326, column: 21, scope: !1467)
!1470 = !DILocalVariable(name: "nrow", arg: 2, scope: !1467, file: !3, line: 326, type: !154)
!1471 = !DILocation(line: 326, column: 30, scope: !1467)
!1472 = !DILocalVariable(name: "ncol", arg: 3, scope: !1467, file: !3, line: 326, type: !154)
!1473 = !DILocation(line: 326, column: 40, scope: !1467)
!1474 = !DILocalVariable(name: "i", scope: !1467, file: !3, line: 328, type: !154)
!1475 = !DILocation(line: 328, column: 6, scope: !1467)
!1476 = !DILocalVariable(name: "j", scope: !1467, file: !3, line: 328, type: !154)
!1477 = !DILocation(line: 328, column: 9, scope: !1467)
!1478 = !DILocation(line: 330, column: 8, scope: !1479)
!1479 = distinct !DILexicalBlock(scope: !1467, file: !3, line: 330, column: 2)
!1480 = !DILocation(line: 330, column: 7, scope: !1479)
!1481 = !DILocation(line: 330, column: 12, scope: !1482)
!1482 = distinct !DILexicalBlock(scope: !1479, file: !3, line: 330, column: 2)
!1483 = !DILocation(line: 330, column: 14, scope: !1482)
!1484 = !DILocation(line: 330, column: 13, scope: !1482)
!1485 = !DILocation(line: 330, column: 2, scope: !1479)
!1486 = !DILocation(line: 331, column: 9, scope: !1487)
!1487 = distinct !DILexicalBlock(scope: !1488, file: !3, line: 331, column: 3)
!1488 = distinct !DILexicalBlock(scope: !1482, file: !3, line: 330, column: 25)
!1489 = !DILocation(line: 331, column: 8, scope: !1487)
!1490 = !DILocation(line: 331, column: 13, scope: !1491)
!1491 = distinct !DILexicalBlock(scope: !1487, file: !3, line: 331, column: 3)
!1492 = !DILocation(line: 331, column: 15, scope: !1491)
!1493 = !DILocation(line: 331, column: 14, scope: !1491)
!1494 = !DILocation(line: 331, column: 3, scope: !1487)
!1495 = !DILocation(line: 332, column: 11, scope: !1496)
!1496 = distinct !DILexicalBlock(scope: !1491, file: !3, line: 331, column: 26)
!1497 = !DILocation(line: 332, column: 22, scope: !1496)
!1498 = !DILocation(line: 332, column: 26, scope: !1496)
!1499 = !DILocation(line: 332, column: 31, scope: !1496)
!1500 = !DILocation(line: 332, column: 30, scope: !1496)
!1501 = !DILocation(line: 332, column: 25, scope: !1496)
!1502 = !DILocation(line: 332, column: 33, scope: !1496)
!1503 = !DILocation(line: 332, column: 32, scope: !1496)
!1504 = !DILocation(line: 332, column: 4, scope: !1496)
!1505 = !DILocation(line: 333, column: 3, scope: !1496)
!1506 = !DILocation(line: 331, column: 22, scope: !1491)
!1507 = !DILocation(line: 331, column: 3, scope: !1491)
!1508 = distinct !{!1508, !1494, !1509}
!1509 = !DILocation(line: 333, column: 3, scope: !1487)
!1510 = !DILocation(line: 334, column: 2, scope: !1488)
!1511 = !DILocation(line: 330, column: 21, scope: !1482)
!1512 = !DILocation(line: 330, column: 2, scope: !1482)
!1513 = distinct !{!1513, !1485, !1514}
!1514 = !DILocation(line: 334, column: 2, scope: !1479)
!1515 = !DILocation(line: 335, column: 1, scope: !1467)
!1516 = distinct !DISubprogram(name: "InitAry", linkageName: "_Z7InitAryPfi", scope: !3, file: !3, line: 359, type: !1159, scopeLine: 360, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1517 = !DILocalVariable(name: "ary", arg: 1, scope: !1516, file: !3, line: 359, type: !125)
!1518 = !DILocation(line: 359, column: 21, scope: !1516)
!1519 = !DILocalVariable(name: "ary_size", arg: 2, scope: !1516, file: !3, line: 359, type: !154)
!1520 = !DILocation(line: 359, column: 30, scope: !1516)
!1521 = !DILocalVariable(name: "i", scope: !1516, file: !3, line: 361, type: !154)
!1522 = !DILocation(line: 361, column: 6, scope: !1516)
!1523 = !DILocation(line: 363, column: 8, scope: !1524)
!1524 = distinct !DILexicalBlock(scope: !1516, file: !3, line: 363, column: 2)
!1525 = !DILocation(line: 363, column: 7, scope: !1524)
!1526 = !DILocation(line: 363, column: 12, scope: !1527)
!1527 = distinct !DILexicalBlock(scope: !1524, file: !3, line: 363, column: 2)
!1528 = !DILocation(line: 363, column: 14, scope: !1527)
!1529 = !DILocation(line: 363, column: 13, scope: !1527)
!1530 = !DILocation(line: 363, column: 2, scope: !1524)
!1531 = !DILocation(line: 364, column: 10, scope: !1532)
!1532 = distinct !DILexicalBlock(scope: !1527, file: !3, line: 363, column: 29)
!1533 = !DILocation(line: 364, column: 22, scope: !1532)
!1534 = !DILocation(line: 364, column: 26, scope: !1532)
!1535 = !DILocation(line: 364, column: 3, scope: !1532)
!1536 = !DILocation(line: 365, column: 2, scope: !1532)
!1537 = !DILocation(line: 363, column: 25, scope: !1527)
!1538 = !DILocation(line: 363, column: 2, scope: !1527)
!1539 = distinct !{!1539, !1530, !1540}
!1540 = !DILocation(line: 365, column: 2, scope: !1524)
!1541 = !DILocation(line: 366, column: 1, scope: !1516)
!1542 = distinct !DISubprogram(name: "Fan1", linkageName: "_Z4Fan1PfS_ii", scope: !3, file: !3, line: 209, type: !1543, scopeLine: 210, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1543 = !DISubroutineType(types: !1544)
!1544 = !{null, !125, !125, !154, !154}
!1545 = !DILocalVariable(name: "m_cuda", arg: 1, scope: !1542, file: !3, line: 209, type: !125)
!1546 = !DILocation(line: 209, column: 29, scope: !1542)
!1547 = !DILocalVariable(name: "a_cuda", arg: 2, scope: !1542, file: !3, line: 209, type: !125)
!1548 = !DILocation(line: 209, column: 44, scope: !1542)
!1549 = !DILocalVariable(name: "Size", arg: 3, scope: !1542, file: !3, line: 209, type: !154)
!1550 = !DILocation(line: 209, column: 56, scope: !1542)
!1551 = !DILocalVariable(name: "t", arg: 4, scope: !1542, file: !3, line: 209, type: !154)
!1552 = !DILocation(line: 209, column: 66, scope: !1542)
!1553 = !DILocation(line: 210, column: 1, scope: !1542)
!1554 = !DILocation(line: 216, column: 1, scope: !1542)
!1555 = distinct !DISubprogram(name: "Fan2", linkageName: "_Z4Fan2PfS_S_iii", scope: !3, file: !3, line: 223, type: !1556, scopeLine: 224, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1556 = !DISubroutineType(types: !1557)
!1557 = !{null, !125, !125, !125, !154, !154, !154}
!1558 = !DILocalVariable(name: "m_cuda", arg: 1, scope: !1555, file: !3, line: 223, type: !125)
!1559 = !DILocation(line: 223, column: 29, scope: !1555)
!1560 = !DILocalVariable(name: "a_cuda", arg: 2, scope: !1555, file: !3, line: 223, type: !125)
!1561 = !DILocation(line: 223, column: 44, scope: !1555)
!1562 = !DILocalVariable(name: "b_cuda", arg: 3, scope: !1555, file: !3, line: 223, type: !125)
!1563 = !DILocation(line: 223, column: 59, scope: !1555)
!1564 = !DILocalVariable(name: "Size", arg: 4, scope: !1555, file: !3, line: 223, type: !154)
!1565 = !DILocation(line: 223, column: 70, scope: !1555)
!1566 = !DILocalVariable(name: "j1", arg: 5, scope: !1555, file: !3, line: 223, type: !154)
!1567 = !DILocation(line: 223, column: 80, scope: !1555)
!1568 = !DILocalVariable(name: "t", arg: 6, scope: !1555, file: !3, line: 223, type: !154)
!1569 = !DILocation(line: 223, column: 88, scope: !1555)
!1570 = !DILocation(line: 224, column: 1, scope: !1555)
!1571 = !DILocation(line: 239, column: 1, scope: !1555)
!1572 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !974, file: !973, line: 423, type: !980, scopeLine: 423, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !979, retainedNodes: !735)
!1573 = !DILocalVariable(name: "this", arg: 1, scope: !1572, type: !1574, flags: DIFlagArtificial | DIFlagObjectPointer)
!1574 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !974, size: 64)
!1575 = !DILocation(line: 0, scope: !1572)
!1576 = !DILocalVariable(name: "vx", arg: 2, scope: !1572, file: !973, line: 423, type: !7)
!1577 = !DILocation(line: 423, column: 43, scope: !1572)
!1578 = !DILocalVariable(name: "vy", arg: 3, scope: !1572, file: !973, line: 423, type: !7)
!1579 = !DILocation(line: 423, column: 64, scope: !1572)
!1580 = !DILocalVariable(name: "vz", arg: 4, scope: !1572, file: !973, line: 423, type: !7)
!1581 = !DILocation(line: 423, column: 85, scope: !1572)
!1582 = !DILocation(line: 423, column: 95, scope: !1572)
!1583 = !DILocation(line: 423, column: 97, scope: !1572)
!1584 = !DILocation(line: 423, column: 102, scope: !1572)
!1585 = !DILocation(line: 423, column: 104, scope: !1572)
!1586 = !DILocation(line: 423, column: 109, scope: !1572)
!1587 = !DILocation(line: 423, column: 111, scope: !1572)
!1588 = !DILocation(line: 423, column: 116, scope: !1572)
!1589 = distinct !DISubprogram(name: "checkCUDAError", linkageName: "_Z14checkCUDAErrorPKc", scope: !3, file: !3, line: 380, type: !1590, scopeLine: 381, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !735)
!1590 = !DISubroutineType(types: !1591)
!1591 = !{null, !294}
!1592 = !DILocalVariable(name: "msg", arg: 1, scope: !1589, file: !3, line: 380, type: !294)
!1593 = !DILocation(line: 380, column: 33, scope: !1589)
!1594 = !DILocalVariable(name: "err", scope: !1589, file: !3, line: 382, type: !1595)
!1595 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 2069, baseType: !5)
!1596 = !DILocation(line: 382, column: 17, scope: !1589)
!1597 = !DILocation(line: 382, column: 23, scope: !1589)
!1598 = !DILocation(line: 383, column: 24, scope: !1599)
!1599 = distinct !DILexicalBlock(scope: !1589, file: !3, line: 383, column: 9)
!1600 = !DILocation(line: 383, column: 21, scope: !1599)
!1601 = !DILocation(line: 383, column: 9, scope: !1589)
!1602 = !DILocation(line: 385, column: 17, scope: !1603)
!1603 = distinct !DILexicalBlock(scope: !1599, file: !3, line: 384, column: 5)
!1604 = !DILocation(line: 385, column: 50, scope: !1603)
!1605 = !DILocation(line: 386, column: 55, scope: !1603)
!1606 = !DILocation(line: 386, column: 35, scope: !1603)
!1607 = !DILocation(line: 385, column: 9, scope: !1603)
!1608 = !DILocation(line: 387, column: 9, scope: !1603)
!1609 = !DILocation(line: 389, column: 1, scope: !1589)

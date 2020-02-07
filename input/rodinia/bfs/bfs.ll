; ModuleID = 'bfs.cu'
source_filename = "bfs.cu"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.Node = type { i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@no_of_nodes = dso_local global i32 0, align 4
@edge_list_size = dso_local global i32 0, align 4
@fp = dso_local global %struct._IO_FILE* null, align 8
@stderr = external dso_local global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [24 x i8] c"Usage: %s <input_file>\0A\00", align 1
@.str.1 = private unnamed_addr constant [14 x i8] c"Reading File\0A\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.3 = private unnamed_addr constant [26 x i8] c"Error Reading graph file\0A\00", align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"%d %d\00", align 1
@.str.6 = private unnamed_addr constant [11 x i8] c"Read File\0A\00", align 1
@.str.7 = private unnamed_addr constant [33 x i8] c"Copied Everything to GPU memory\0A\00", align 1
@.str.8 = private unnamed_addr constant [27 x i8] c"Start traversing the tree\0A\00", align 1
@.str.9 = private unnamed_addr constant [26 x i8] c"Kernel Executed %d times\0A\00", align 1
@.str.10 = private unnamed_addr constant [11 x i8] c"result.txt\00", align 1
@.str.11 = private unnamed_addr constant [2 x i8] c"w\00", align 1
@.str.12 = private unnamed_addr constant [13 x i8] c"%d) cost:%d\0A\00", align 1
@.str.13 = private unnamed_addr constant [29 x i8] c"Result stored in result.txt\0A\00", align 1

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z6KernelP4NodePiPbS2_S2_S1_i(%struct.Node* %0, i32* %1, i8* %2, i8* %3, i8* %4, i32* %5, i32 %6) #0 {
  %8 = alloca %struct.Node*, align 8
  %9 = alloca i32*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  %12 = alloca i8*, align 8
  %13 = alloca i32*, align 8
  %14 = alloca i32, align 4
  %15 = alloca %struct.dim3, align 8
  %16 = alloca %struct.dim3, align 8
  %17 = alloca i64, align 8
  %18 = alloca i8*, align 8
  %19 = alloca { i64, i32 }, align 8
  %20 = alloca { i64, i32 }, align 8
  store %struct.Node* %0, %struct.Node** %8, align 8
  store i32* %1, i32** %9, align 8
  store i8* %2, i8** %10, align 8
  store i8* %3, i8** %11, align 8
  store i8* %4, i8** %12, align 8
  store i32* %5, i32** %13, align 8
  store i32 %6, i32* %14, align 4
  %21 = alloca i8*, i64 7, align 16
  %22 = bitcast %struct.Node** %8 to i8*
  %23 = getelementptr i8*, i8** %21, i32 0
  store i8* %22, i8** %23
  %24 = bitcast i32** %9 to i8*
  %25 = getelementptr i8*, i8** %21, i32 1
  store i8* %24, i8** %25
  %26 = bitcast i8** %10 to i8*
  %27 = getelementptr i8*, i8** %21, i32 2
  store i8* %26, i8** %27
  %28 = bitcast i8** %11 to i8*
  %29 = getelementptr i8*, i8** %21, i32 3
  store i8* %28, i8** %29
  %30 = bitcast i8** %12 to i8*
  %31 = getelementptr i8*, i8** %21, i32 4
  store i8* %30, i8** %31
  %32 = bitcast i32** %13 to i8*
  %33 = getelementptr i8*, i8** %21, i32 5
  store i8* %32, i8** %33
  %34 = bitcast i32* %14 to i8*
  %35 = getelementptr i8*, i8** %21, i32 6
  store i8* %34, i8** %35
  %36 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %15, %struct.dim3* %16, i64* %17, i8** %18)
  %37 = load i64, i64* %17, align 8
  %38 = load i8*, i8** %18, align 8
  %39 = bitcast { i64, i32 }* %19 to i8*
  %40 = bitcast %struct.dim3* %15 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %39, i8* align 8 %40, i64 12, i1 false)
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 0
  %42 = load i64, i64* %41, align 8
  %43 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %19, i32 0, i32 1
  %44 = load i32, i32* %43, align 8
  %45 = bitcast { i64, i32 }* %20 to i8*
  %46 = bitcast %struct.dim3* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %45, i8* align 8 %46, i64 12, i1 false)
  %47 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 0
  %48 = load i64, i64* %47, align 8
  %49 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %20, i32 0, i32 1
  %50 = load i32, i32* %49, align 8
  %51 = bitcast i8* %38 to %struct.CUstream_st*
  %52 = call i32 @cudaLaunchKernel(i8* bitcast (void (%struct.Node*, i32*, i8*, i8*, i8*, i32*, i32)* @_Z6KernelP4NodePiPbS2_S2_S1_i to i8*), i64 %42, i32 %44, i64 %48, i32 %50, i8** %21, i64 %37, %struct.CUstream_st* %51)
  br label %53

53:                                               ; preds = %7
  ret void
}

declare dso_local i32 @__cudaPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**)

declare dso_local i32 @cudaLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.CUstream_st*)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z7Kernel2PbS_S_S_i(i8* %0, i8* %1, i8* %2, i8* %3, i32 %4) #0 {
  %6 = alloca i8*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i8*, align 8
  %9 = alloca i8*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %struct.dim3, align 8
  %12 = alloca %struct.dim3, align 8
  %13 = alloca i64, align 8
  %14 = alloca i8*, align 8
  %15 = alloca { i64, i32 }, align 8
  %16 = alloca { i64, i32 }, align 8
  store i8* %0, i8** %6, align 8
  store i8* %1, i8** %7, align 8
  store i8* %2, i8** %8, align 8
  store i8* %3, i8** %9, align 8
  store i32 %4, i32* %10, align 4
  %17 = alloca i8*, i64 5, align 16
  %18 = bitcast i8** %6 to i8*
  %19 = getelementptr i8*, i8** %17, i32 0
  store i8* %18, i8** %19
  %20 = bitcast i8** %7 to i8*
  %21 = getelementptr i8*, i8** %17, i32 1
  store i8* %20, i8** %21
  %22 = bitcast i8** %8 to i8*
  %23 = getelementptr i8*, i8** %17, i32 2
  store i8* %22, i8** %23
  %24 = bitcast i8** %9 to i8*
  %25 = getelementptr i8*, i8** %17, i32 3
  store i8* %24, i8** %25
  %26 = bitcast i32* %10 to i8*
  %27 = getelementptr i8*, i8** %17, i32 4
  store i8* %26, i8** %27
  %28 = call i32 @__cudaPopCallConfiguration(%struct.dim3* %11, %struct.dim3* %12, i64* %13, i8** %14)
  %29 = load i64, i64* %13, align 8
  %30 = load i8*, i8** %14, align 8
  %31 = bitcast { i64, i32 }* %15 to i8*
  %32 = bitcast %struct.dim3* %11 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %31, i8* align 8 %32, i64 12, i1 false)
  %33 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 0
  %34 = load i64, i64* %33, align 8
  %35 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %15, i32 0, i32 1
  %36 = load i32, i32* %35, align 8
  %37 = bitcast { i64, i32 }* %16 to i8*
  %38 = bitcast %struct.dim3* %12 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %37, i8* align 8 %38, i64 12, i1 false)
  %39 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 0
  %40 = load i64, i64* %39, align 8
  %41 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %16, i32 0, i32 1
  %42 = load i32, i32* %41, align 8
  %43 = bitcast i8* %30 to %struct.CUstream_st*
  %44 = call i32 @cudaLaunchKernel(i8* bitcast (void (i8*, i8*, i8*, i8*, i32)* @_Z7Kernel2PbS_S_S_i to i8*), i64 %34, i32 %36, i64 %40, i32 %42, i8** %17, i64 %29, %struct.CUstream_st* %43)
  br label %45

45:                                               ; preds = %5
  ret void
}

; Function Attrs: noinline norecurse optnone uwtable
define dso_local i32 @main(i32 %0, i8** %1) #2 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  store i32 0, i32* @no_of_nodes, align 4
  store i32 0, i32* @edge_list_size, align 4
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  call void @_Z8BFSGraphiPPc(i32 %5, i8** %6)
  ret i32 0
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z8BFSGraphiPPc(i32 %0, i8** %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %struct.Node*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  %12 = alloca i8*, align 8
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32*, align 8
  %19 = alloca i32, align 4
  %20 = alloca %struct.Node*, align 8
  %21 = alloca i32*, align 8
  %22 = alloca i8*, align 8
  %23 = alloca i8*, align 8
  %24 = alloca i8*, align 8
  %25 = alloca i32*, align 8
  %26 = alloca i32, align 4
  %27 = alloca i32*, align 8
  %28 = alloca i8*, align 8
  %29 = alloca %struct.dim3, align 4
  %30 = alloca %struct.dim3, align 4
  %31 = alloca i32, align 4
  %32 = alloca i8, align 1
  %33 = alloca %struct.dim3, align 4
  %34 = alloca %struct.dim3, align 4
  %35 = alloca { i64, i32 }, align 4
  %36 = alloca { i64, i32 }, align 4
  %37 = alloca %struct.dim3, align 4
  %38 = alloca %struct.dim3, align 4
  %39 = alloca { i64, i32 }, align 4
  %40 = alloca { i64, i32 }, align 4
  %41 = alloca %struct._IO_FILE*, align 8
  %42 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %43 = load i32, i32* %3, align 4
  %44 = icmp ne i32 %43, 2
  br i1 %44, label %45, label %48

45:                                               ; preds = %2
  %46 = load i32, i32* %3, align 4
  %47 = load i8**, i8*** %4, align 8
  call void @_Z5UsageiPPc(i32 %46, i8** %47)
  call void @exit(i32 0) #8
  unreachable

48:                                               ; preds = %2
  %49 = load i8**, i8*** %4, align 8
  %50 = getelementptr inbounds i8*, i8** %49, i64 1
  %51 = load i8*, i8** %50, align 8
  store i8* %51, i8** %5, align 8
  %52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.1, i64 0, i64 0))
  %53 = load i8*, i8** %5, align 8
  %54 = call %struct._IO_FILE* @fopen(i8* %53, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i64 0, i64 0))
  store %struct._IO_FILE* %54, %struct._IO_FILE** @fp, align 8
  %55 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %56 = icmp ne %struct._IO_FILE* %55, null
  br i1 %56, label %59, label %57

57:                                               ; preds = %48
  %58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.3, i64 0, i64 0))
  br label %380

59:                                               ; preds = %48
  store i32 0, i32* %6, align 4
  %60 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %61 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %60, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i32* @no_of_nodes)
  store i32 1, i32* %7, align 4
  %62 = load i32, i32* @no_of_nodes, align 4
  store i32 %62, i32* %8, align 4
  %63 = load i32, i32* @no_of_nodes, align 4
  %64 = icmp sgt i32 %63, 512
  br i1 %64, label %65, label %71

65:                                               ; preds = %59
  %66 = load i32, i32* @no_of_nodes, align 4
  %67 = sitofp i32 %66 to double
  %68 = fdiv double %67, 5.120000e+02
  %69 = call double @llvm.ceil.f64(double %68)
  %70 = fptosi double %69 to i32
  store i32 %70, i32* %7, align 4
  store i32 512, i32* %8, align 4
  br label %71

71:                                               ; preds = %65, %59
  %72 = load i32, i32* @no_of_nodes, align 4
  %73 = sext i32 %72 to i64
  %74 = mul i64 8, %73
  %75 = call noalias i8* @malloc(i64 %74) #9
  %76 = bitcast i8* %75 to %struct.Node*
  store %struct.Node* %76, %struct.Node** %9, align 8
  %77 = load i32, i32* @no_of_nodes, align 4
  %78 = sext i32 %77 to i64
  %79 = mul i64 1, %78
  %80 = call noalias i8* @malloc(i64 %79) #9
  store i8* %80, i8** %10, align 8
  %81 = load i32, i32* @no_of_nodes, align 4
  %82 = sext i32 %81 to i64
  %83 = mul i64 1, %82
  %84 = call noalias i8* @malloc(i64 %83) #9
  store i8* %84, i8** %11, align 8
  %85 = load i32, i32* @no_of_nodes, align 4
  %86 = sext i32 %85 to i64
  %87 = mul i64 1, %86
  %88 = call noalias i8* @malloc(i64 %87) #9
  store i8* %88, i8** %12, align 8
  store i32 0, i32* %15, align 4
  br label %89

89:                                               ; preds = %120, %71
  %90 = load i32, i32* %15, align 4
  %91 = load i32, i32* @no_of_nodes, align 4
  %92 = icmp ult i32 %90, %91
  br i1 %92, label %93, label %123

93:                                               ; preds = %89
  %94 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %95 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %94, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.5, i64 0, i64 0), i32* %13, i32* %14)
  %96 = load i32, i32* %13, align 4
  %97 = load %struct.Node*, %struct.Node** %9, align 8
  %98 = load i32, i32* %15, align 4
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds %struct.Node, %struct.Node* %97, i64 %99
  %101 = getelementptr inbounds %struct.Node, %struct.Node* %100, i32 0, i32 0
  store i32 %96, i32* %101, align 4
  %102 = load i32, i32* %14, align 4
  %103 = load %struct.Node*, %struct.Node** %9, align 8
  %104 = load i32, i32* %15, align 4
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds %struct.Node, %struct.Node* %103, i64 %105
  %107 = getelementptr inbounds %struct.Node, %struct.Node* %106, i32 0, i32 1
  store i32 %102, i32* %107, align 4
  %108 = load i8*, i8** %10, align 8
  %109 = load i32, i32* %15, align 4
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i8, i8* %108, i64 %110
  store i8 0, i8* %111, align 1
  %112 = load i8*, i8** %11, align 8
  %113 = load i32, i32* %15, align 4
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i8, i8* %112, i64 %114
  store i8 0, i8* %115, align 1
  %116 = load i8*, i8** %12, align 8
  %117 = load i32, i32* %15, align 4
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8* %116, i64 %118
  store i8 0, i8* %119, align 1
  br label %120

120:                                              ; preds = %93
  %121 = load i32, i32* %15, align 4
  %122 = add i32 %121, 1
  store i32 %122, i32* %15, align 4
  br label %89

123:                                              ; preds = %89
  %124 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %125 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %124, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i32* %6)
  store i32 0, i32* %6, align 4
  %126 = load i8*, i8** %10, align 8
  %127 = load i32, i32* %6, align 4
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds i8, i8* %126, i64 %128
  store i8 1, i8* %129, align 1
  %130 = load i8*, i8** %12, align 8
  %131 = load i32, i32* %6, align 4
  %132 = sext i32 %131 to i64
  %133 = getelementptr inbounds i8, i8* %130, i64 %132
  store i8 1, i8* %133, align 1
  %134 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %135 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %134, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i32* @edge_list_size)
  %136 = load i32, i32* @edge_list_size, align 4
  %137 = sext i32 %136 to i64
  %138 = mul i64 4, %137
  %139 = call noalias i8* @malloc(i64 %138) #9
  %140 = bitcast i8* %139 to i32*
  store i32* %140, i32** %18, align 8
  store i32 0, i32* %19, align 4
  br label %141

141:                                              ; preds = %155, %123
  %142 = load i32, i32* %19, align 4
  %143 = load i32, i32* @edge_list_size, align 4
  %144 = icmp slt i32 %142, %143
  br i1 %144, label %145, label %158

145:                                              ; preds = %141
  %146 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %147 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %146, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i32* %16)
  %148 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %149 = call i32 (%struct._IO_FILE*, i8*, ...) @fscanf(%struct._IO_FILE* %148, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.4, i64 0, i64 0), i32* %17)
  %150 = load i32, i32* %16, align 4
  %151 = load i32*, i32** %18, align 8
  %152 = load i32, i32* %19, align 4
  %153 = sext i32 %152 to i64
  %154 = getelementptr inbounds i32, i32* %151, i64 %153
  store i32 %150, i32* %154, align 4
  br label %155

155:                                              ; preds = %145
  %156 = load i32, i32* %19, align 4
  %157 = add nsw i32 %156, 1
  store i32 %157, i32* %19, align 4
  br label %141

158:                                              ; preds = %141
  %159 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %160 = icmp ne %struct._IO_FILE* %159, null
  br i1 %160, label %161, label %164

161:                                              ; preds = %158
  %162 = load %struct._IO_FILE*, %struct._IO_FILE** @fp, align 8
  %163 = call i32 @fclose(%struct._IO_FILE* %162)
  br label %164

164:                                              ; preds = %161, %158
  %165 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.6, i64 0, i64 0))
  %166 = bitcast %struct.Node** %20 to i8**
  %167 = load i32, i32* @no_of_nodes, align 4
  %168 = sext i32 %167 to i64
  %169 = mul i64 8, %168
  %170 = call i32 @cudaMalloc(i8** %166, i64 %169)
  %171 = load %struct.Node*, %struct.Node** %20, align 8
  %172 = bitcast %struct.Node* %171 to i8*
  %173 = load %struct.Node*, %struct.Node** %9, align 8
  %174 = bitcast %struct.Node* %173 to i8*
  %175 = load i32, i32* @no_of_nodes, align 4
  %176 = sext i32 %175 to i64
  %177 = mul i64 8, %176
  %178 = call i32 @cudaMemcpy(i8* %172, i8* %174, i64 %177, i32 1)
  %179 = bitcast i32** %21 to i8**
  %180 = load i32, i32* @edge_list_size, align 4
  %181 = sext i32 %180 to i64
  %182 = mul i64 4, %181
  %183 = call i32 @cudaMalloc(i8** %179, i64 %182)
  %184 = load i32*, i32** %21, align 8
  %185 = bitcast i32* %184 to i8*
  %186 = load i32*, i32** %18, align 8
  %187 = bitcast i32* %186 to i8*
  %188 = load i32, i32* @edge_list_size, align 4
  %189 = sext i32 %188 to i64
  %190 = mul i64 4, %189
  %191 = call i32 @cudaMemcpy(i8* %185, i8* %187, i64 %190, i32 1)
  %192 = load i32, i32* @no_of_nodes, align 4
  %193 = sext i32 %192 to i64
  %194 = mul i64 1, %193
  %195 = call i32 @cudaMalloc(i8** %22, i64 %194)
  %196 = load i8*, i8** %22, align 8
  %197 = load i8*, i8** %10, align 8
  %198 = load i32, i32* @no_of_nodes, align 4
  %199 = sext i32 %198 to i64
  %200 = mul i64 1, %199
  %201 = call i32 @cudaMemcpy(i8* %196, i8* %197, i64 %200, i32 1)
  %202 = load i32, i32* @no_of_nodes, align 4
  %203 = sext i32 %202 to i64
  %204 = mul i64 1, %203
  %205 = call i32 @cudaMalloc(i8** %23, i64 %204)
  %206 = load i8*, i8** %23, align 8
  %207 = load i8*, i8** %11, align 8
  %208 = load i32, i32* @no_of_nodes, align 4
  %209 = sext i32 %208 to i64
  %210 = mul i64 1, %209
  %211 = call i32 @cudaMemcpy(i8* %206, i8* %207, i64 %210, i32 1)
  %212 = load i32, i32* @no_of_nodes, align 4
  %213 = sext i32 %212 to i64
  %214 = mul i64 1, %213
  %215 = call i32 @cudaMalloc(i8** %24, i64 %214)
  %216 = load i8*, i8** %24, align 8
  %217 = load i8*, i8** %12, align 8
  %218 = load i32, i32* @no_of_nodes, align 4
  %219 = sext i32 %218 to i64
  %220 = mul i64 1, %219
  %221 = call i32 @cudaMemcpy(i8* %216, i8* %217, i64 %220, i32 1)
  %222 = load i32, i32* @no_of_nodes, align 4
  %223 = sext i32 %222 to i64
  %224 = mul i64 4, %223
  %225 = call noalias i8* @malloc(i64 %224) #9
  %226 = bitcast i8* %225 to i32*
  store i32* %226, i32** %25, align 8
  store i32 0, i32* %26, align 4
  br label %227

227:                                              ; preds = %236, %164
  %228 = load i32, i32* %26, align 4
  %229 = load i32, i32* @no_of_nodes, align 4
  %230 = icmp slt i32 %228, %229
  br i1 %230, label %231, label %239

231:                                              ; preds = %227
  %232 = load i32*, i32** %25, align 8
  %233 = load i32, i32* %26, align 4
  %234 = sext i32 %233 to i64
  %235 = getelementptr inbounds i32, i32* %232, i64 %234
  store i32 -1, i32* %235, align 4
  br label %236

236:                                              ; preds = %231
  %237 = load i32, i32* %26, align 4
  %238 = add nsw i32 %237, 1
  store i32 %238, i32* %26, align 4
  br label %227

239:                                              ; preds = %227
  %240 = load i32*, i32** %25, align 8
  %241 = load i32, i32* %6, align 4
  %242 = sext i32 %241 to i64
  %243 = getelementptr inbounds i32, i32* %240, i64 %242
  store i32 0, i32* %243, align 4
  %244 = bitcast i32** %27 to i8**
  %245 = load i32, i32* @no_of_nodes, align 4
  %246 = sext i32 %245 to i64
  %247 = mul i64 4, %246
  %248 = call i32 @cudaMalloc(i8** %244, i64 %247)
  %249 = load i32*, i32** %27, align 8
  %250 = bitcast i32* %249 to i8*
  %251 = load i32*, i32** %25, align 8
  %252 = bitcast i32* %251 to i8*
  %253 = load i32, i32* @no_of_nodes, align 4
  %254 = sext i32 %253 to i64
  %255 = mul i64 4, %254
  %256 = call i32 @cudaMemcpy(i8* %250, i8* %252, i64 %255, i32 1)
  %257 = call i32 @cudaMalloc(i8** %28, i64 1)
  %258 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.7, i64 0, i64 0))
  %259 = load i32, i32* %7, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %29, i32 %259, i32 1, i32 1)
  %260 = load i32, i32* %8, align 4
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %30, i32 %260, i32 1, i32 1)
  store i32 0, i32* %31, align 4
  %261 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.8, i64 0, i64 0))
  br label %262

262:                                              ; preds = %321, %239
  store i8 0, i8* %32, align 1
  %263 = load i8*, i8** %28, align 8
  %264 = call i32 @cudaMemcpy(i8* %263, i8* %32, i64 1, i32 1)
  %265 = bitcast %struct.dim3* %33 to i8*
  %266 = bitcast %struct.dim3* %29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %265, i8* align 4 %266, i64 12, i1 false)
  %267 = bitcast %struct.dim3* %34 to i8*
  %268 = bitcast %struct.dim3* %30 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %267, i8* align 4 %268, i64 12, i1 false)
  %269 = bitcast { i64, i32 }* %35 to i8*
  %270 = bitcast %struct.dim3* %33 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %269, i8* align 4 %270, i64 12, i1 false)
  %271 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %35, i32 0, i32 0
  %272 = load i64, i64* %271, align 4
  %273 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %35, i32 0, i32 1
  %274 = load i32, i32* %273, align 4
  %275 = bitcast { i64, i32 }* %36 to i8*
  %276 = bitcast %struct.dim3* %34 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %275, i8* align 4 %276, i64 12, i1 false)
  %277 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %36, i32 0, i32 0
  %278 = load i64, i64* %277, align 4
  %279 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %36, i32 0, i32 1
  %280 = load i32, i32* %279, align 4
  %281 = call i32 @__cudaPushCallConfiguration(i64 %272, i32 %274, i64 %278, i32 %280, i64 0, i8* null)
  %282 = icmp ne i32 %281, 0
  br i1 %282, label %291, label %283

283:                                              ; preds = %262
  %284 = load %struct.Node*, %struct.Node** %20, align 8
  %285 = load i32*, i32** %21, align 8
  %286 = load i8*, i8** %22, align 8
  %287 = load i8*, i8** %23, align 8
  %288 = load i8*, i8** %24, align 8
  %289 = load i32*, i32** %27, align 8
  %290 = load i32, i32* @no_of_nodes, align 4
  call void @_Z6KernelP4NodePiPbS2_S2_S1_i(%struct.Node* %284, i32* %285, i8* %286, i8* %287, i8* %288, i32* %289, i32 %290)
  br label %291

291:                                              ; preds = %283, %262
  %292 = bitcast %struct.dim3* %37 to i8*
  %293 = bitcast %struct.dim3* %29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %292, i8* align 4 %293, i64 12, i1 false)
  %294 = bitcast %struct.dim3* %38 to i8*
  %295 = bitcast %struct.dim3* %30 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %294, i8* align 4 %295, i64 12, i1 false)
  %296 = bitcast { i64, i32 }* %39 to i8*
  %297 = bitcast %struct.dim3* %37 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %296, i8* align 4 %297, i64 12, i1 false)
  %298 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %39, i32 0, i32 0
  %299 = load i64, i64* %298, align 4
  %300 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %39, i32 0, i32 1
  %301 = load i32, i32* %300, align 4
  %302 = bitcast { i64, i32 }* %40 to i8*
  %303 = bitcast %struct.dim3* %38 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %302, i8* align 4 %303, i64 12, i1 false)
  %304 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %40, i32 0, i32 0
  %305 = load i64, i64* %304, align 4
  %306 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %40, i32 0, i32 1
  %307 = load i32, i32* %306, align 4
  %308 = call i32 @__cudaPushCallConfiguration(i64 %299, i32 %301, i64 %305, i32 %307, i64 0, i8* null)
  %309 = icmp ne i32 %308, 0
  br i1 %309, label %316, label %310

310:                                              ; preds = %291
  %311 = load i8*, i8** %22, align 8
  %312 = load i8*, i8** %23, align 8
  %313 = load i8*, i8** %24, align 8
  %314 = load i8*, i8** %28, align 8
  %315 = load i32, i32* @no_of_nodes, align 4
  call void @_Z7Kernel2PbS_S_S_i(i8* %311, i8* %312, i8* %313, i8* %314, i32 %315)
  br label %316

316:                                              ; preds = %310, %291
  %317 = load i8*, i8** %28, align 8
  %318 = call i32 @cudaMemcpy(i8* %32, i8* %317, i64 1, i32 2)
  %319 = load i32, i32* %31, align 4
  %320 = add nsw i32 %319, 1
  store i32 %320, i32* %31, align 4
  br label %321

321:                                              ; preds = %316
  %322 = load i8, i8* %32, align 1
  %323 = trunc i8 %322 to i1
  br i1 %323, label %262, label %324

324:                                              ; preds = %321
  %325 = load i32, i32* %31, align 4
  %326 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.9, i64 0, i64 0), i32 %325)
  %327 = load i32*, i32** %25, align 8
  %328 = bitcast i32* %327 to i8*
  %329 = load i32*, i32** %27, align 8
  %330 = bitcast i32* %329 to i8*
  %331 = load i32, i32* @no_of_nodes, align 4
  %332 = sext i32 %331 to i64
  %333 = mul i64 4, %332
  %334 = call i32 @cudaMemcpy(i8* %328, i8* %330, i64 %333, i32 2)
  %335 = call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.10, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.11, i64 0, i64 0))
  store %struct._IO_FILE* %335, %struct._IO_FILE** %41, align 8
  store i32 0, i32* %42, align 4
  br label %336

336:                                              ; preds = %349, %324
  %337 = load i32, i32* %42, align 4
  %338 = load i32, i32* @no_of_nodes, align 4
  %339 = icmp slt i32 %337, %338
  br i1 %339, label %340, label %352

340:                                              ; preds = %336
  %341 = load %struct._IO_FILE*, %struct._IO_FILE** %41, align 8
  %342 = load i32, i32* %42, align 4
  %343 = load i32*, i32** %25, align 8
  %344 = load i32, i32* %42, align 4
  %345 = sext i32 %344 to i64
  %346 = getelementptr inbounds i32, i32* %343, i64 %345
  %347 = load i32, i32* %346, align 4
  %348 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %341, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.12, i64 0, i64 0), i32 %342, i32 %347)
  br label %349

349:                                              ; preds = %340
  %350 = load i32, i32* %42, align 4
  %351 = add nsw i32 %350, 1
  store i32 %351, i32* %42, align 4
  br label %336

352:                                              ; preds = %336
  %353 = load %struct._IO_FILE*, %struct._IO_FILE** %41, align 8
  %354 = call i32 @fclose(%struct._IO_FILE* %353)
  %355 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.13, i64 0, i64 0))
  %356 = load %struct.Node*, %struct.Node** %9, align 8
  %357 = bitcast %struct.Node* %356 to i8*
  call void @free(i8* %357) #9
  %358 = load i32*, i32** %18, align 8
  %359 = bitcast i32* %358 to i8*
  call void @free(i8* %359) #9
  %360 = load i8*, i8** %10, align 8
  call void @free(i8* %360) #9
  %361 = load i8*, i8** %11, align 8
  call void @free(i8* %361) #9
  %362 = load i8*, i8** %12, align 8
  call void @free(i8* %362) #9
  %363 = load i32*, i32** %25, align 8
  %364 = bitcast i32* %363 to i8*
  call void @free(i8* %364) #9
  %365 = load %struct.Node*, %struct.Node** %20, align 8
  %366 = bitcast %struct.Node* %365 to i8*
  %367 = call i32 @cudaFree(i8* %366)
  %368 = load i32*, i32** %21, align 8
  %369 = bitcast i32* %368 to i8*
  %370 = call i32 @cudaFree(i8* %369)
  %371 = load i8*, i8** %22, align 8
  %372 = call i32 @cudaFree(i8* %371)
  %373 = load i8*, i8** %23, align 8
  %374 = call i32 @cudaFree(i8* %373)
  %375 = load i8*, i8** %24, align 8
  %376 = call i32 @cudaFree(i8* %375)
  %377 = load i32*, i32** %27, align 8
  %378 = bitcast i32* %377 to i8*
  %379 = call i32 @cudaFree(i8* %378)
  br label %380

380:                                              ; preds = %352, %57
  ret void
}

; Function Attrs: noinline optnone uwtable
define dso_local void @_Z5UsageiPPc(i32 %0, i8** %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  %6 = load i8**, i8*** %4, align 8
  %7 = getelementptr inbounds i8*, i8** %6, i64 0
  %8 = load i8*, i8** %7, align 8
  %9 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %5, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str, i64 0, i64 0), i8* %8)
  ret void
}

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #3

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #4

declare dso_local i32 @printf(i8*, ...) #3

declare dso_local %struct._IO_FILE* @fopen(i8*, i8*) #3

declare dso_local i32 @fscanf(%struct._IO_FILE*, i8*, ...) #3

; Function Attrs: nounwind readnone speculatable willreturn
declare double @llvm.ceil.f64(double) #5

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #6

declare dso_local i32 @fclose(%struct._IO_FILE*) #3

declare dso_local i32 @cudaMalloc(i8**, i64) #3

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

; Function Attrs: noinline nounwind optnone uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %0, i32 %1, i32 %2, i32 %3) unnamed_addr #7 comdat align 2 {
  %5 = alloca %struct.dim3*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store %struct.dim3* %0, %struct.dim3** %5, align 8
  store i32 %1, i32* %6, align 4
  store i32 %2, i32* %7, align 4
  store i32 %3, i32* %8, align 4
  %9 = load %struct.dim3*, %struct.dim3** %5, align 8
  %10 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 0
  %11 = load i32, i32* %6, align 4
  store i32 %11, i32* %10, align 4
  %12 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 1
  %13 = load i32, i32* %7, align 4
  store i32 %13, i32* %12, align 4
  %14 = getelementptr inbounds %struct.dim3, %struct.dim3* %9, i32 0, i32 2
  %15 = load i32, i32* %8, align 4
  store i32 %15, i32* %14, align 4
  ret void
}

declare dso_local i32 @__cudaPushCallConfiguration(i64, i32, i64, i32, i64, i8*) #3

; Function Attrs: nounwind
declare dso_local void @free(i8*) #6

declare dso_local i32 @cudaFree(i8*) #3

attributes #0 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { noinline norecurse optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable willreturn }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)"}

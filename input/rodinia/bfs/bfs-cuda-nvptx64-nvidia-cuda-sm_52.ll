; ModuleID = 'bfs.cu'
source_filename = "bfs.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }
%struct.Node = type { i32, i32 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaMalloc(i8** %0, i64 %1) #0 {
  %3 = alloca i8**, align 8
  %4 = alloca i64, align 8
  store i8** %0, i8*** %3, align 8
  store i64 %1, i64* %4, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes* %0, i8* %1) #0 {
  %3 = alloca %struct.cudaFuncAttributes*, align 8
  %4 = alloca i8*, align 8
  store %struct.cudaFuncAttributes* %0, %struct.cudaFuncAttributes** %3, align 8
  store i8* %1, i8** %4, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaDeviceGetAttribute(i32* %0, i32 %1, i32 %2) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32* %0, i32** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaGetDevice(i32* %0) #0 {
  %2 = alloca i32*, align 8
  store i32* %0, i32** %2, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32* %0, i8* %1, i32 %2, i64 %3) #0 {
  %5 = alloca i32*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i64, align 8
  store i32* %0, i32** %5, align 8
  store i8* %1, i8** %6, align 8
  store i32 %2, i32* %7, align 4
  store i64 %3, i64* %8, align 8
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32* %0, i8* %1, i32 %2, i64 %3, i32 %4) #0 {
  %6 = alloca i32*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i64, align 8
  %10 = alloca i32, align 4
  store i32* %0, i32** %6, align 8
  store i8* %1, i8** %7, align 8
  store i32 %2, i32* %8, align 4
  store i64 %3, i64* %9, align 8
  store i32 %4, i32* %10, align 4
  ret i32 999
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z6KernelP4NodePiPbS2_S2_S1_i(%struct.Node* %0, i32* %1, i8* %2, i8* %3, i8* %4, i32* %5, i32 %6) #0 {
  %8 = alloca %struct.Node*, align 8
  %9 = alloca i32*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  %12 = alloca i8*, align 8
  %13 = alloca i32*, align 8
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  store %struct.Node* %0, %struct.Node** %8, align 8
  store i32* %1, i32** %9, align 8
  store i8* %2, i8** %10, align 8
  store i8* %3, i8** %11, align 8
  store i8* %4, i8** %12, align 8
  store i32* %5, i32** %13, align 8
  store i32 %6, i32* %14, align 4
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !11
  %19 = mul i32 %18, 512
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !12
  %21 = add i32 %19, %20
  store i32 %21, i32* %15, align 4
  %22 = load i32, i32* %15, align 4
  %23 = load i32, i32* %14, align 4
  %24 = icmp slt i32 %22, %23
  br i1 %24, label %25, label %91

25:                                               ; preds = %7
  %26 = load i8*, i8** %10, align 8
  %27 = load i32, i32* %15, align 4
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds i8, i8* %26, i64 %28
  %30 = load i8, i8* %29, align 1
  %31 = trunc i8 %30 to i1
  br i1 %31, label %32, label %91

32:                                               ; preds = %25
  %33 = load i8*, i8** %10, align 8
  %34 = load i32, i32* %15, align 4
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds i8, i8* %33, i64 %35
  store i8 0, i8* %36, align 1
  %37 = load %struct.Node*, %struct.Node** %8, align 8
  %38 = load i32, i32* %15, align 4
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds %struct.Node, %struct.Node* %37, i64 %39
  %41 = getelementptr inbounds %struct.Node, %struct.Node* %40, i32 0, i32 0
  %42 = load i32, i32* %41, align 4
  store i32 %42, i32* %16, align 4
  br label %43

43:                                               ; preds = %87, %32
  %44 = load i32, i32* %16, align 4
  %45 = load %struct.Node*, %struct.Node** %8, align 8
  %46 = load i32, i32* %15, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds %struct.Node, %struct.Node* %45, i64 %47
  %49 = getelementptr inbounds %struct.Node, %struct.Node* %48, i32 0, i32 1
  %50 = load i32, i32* %49, align 4
  %51 = load %struct.Node*, %struct.Node** %8, align 8
  %52 = load i32, i32* %15, align 4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds %struct.Node, %struct.Node* %51, i64 %53
  %55 = getelementptr inbounds %struct.Node, %struct.Node* %54, i32 0, i32 0
  %56 = load i32, i32* %55, align 4
  %57 = add nsw i32 %50, %56
  %58 = icmp slt i32 %44, %57
  br i1 %58, label %59, label %90

59:                                               ; preds = %43
  %60 = load i32*, i32** %9, align 8
  %61 = load i32, i32* %16, align 4
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds i32, i32* %60, i64 %62
  %64 = load i32, i32* %63, align 4
  store i32 %64, i32* %17, align 4
  %65 = load i8*, i8** %12, align 8
  %66 = load i32, i32* %17, align 4
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds i8, i8* %65, i64 %67
  %69 = load i8, i8* %68, align 1
  %70 = trunc i8 %69 to i1
  br i1 %70, label %86, label %71

71:                                               ; preds = %59
  %72 = load i32*, i32** %13, align 8
  %73 = load i32, i32* %15, align 4
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds i32, i32* %72, i64 %74
  %76 = load i32, i32* %75, align 4
  %77 = add nsw i32 %76, 1
  %78 = load i32*, i32** %13, align 8
  %79 = load i32, i32* %17, align 4
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds i32, i32* %78, i64 %80
  store i32 %77, i32* %81, align 4
  %82 = load i8*, i8** %11, align 8
  %83 = load i32, i32* %17, align 4
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds i8, i8* %82, i64 %84
  store i8 1, i8* %85, align 1
  br label %86

86:                                               ; preds = %71, %59
  br label %87

87:                                               ; preds = %86
  %88 = load i32, i32* %16, align 4
  %89 = add nsw i32 %88, 1
  store i32 %89, i32* %16, align 4
  br label %43

90:                                               ; preds = %43
  br label %91

91:                                               ; preds = %90, %25, %7
  ret void
}

; Function Attrs: convergent noinline nounwind optnone
define dso_local void @_Z7Kernel2PbS_S_S_i(i8* %0, i8* %1, i8* %2, i8* %3, i32 %4) #0 {
  %6 = alloca i8*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i8*, align 8
  %9 = alloca i8*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i8* %0, i8** %6, align 8
  store i8* %1, i8** %7, align 8
  store i8* %2, i8** %8, align 8
  store i8* %3, i8** %9, align 8
  store i32 %4, i32* %10, align 4
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !11
  %13 = mul i32 %12, 512
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !12
  %15 = add i32 %13, %14
  store i32 %15, i32* %11, align 4
  %16 = load i32, i32* %11, align 4
  %17 = load i32, i32* %10, align 4
  %18 = icmp slt i32 %16, %17
  br i1 %18, label %19, label %40

19:                                               ; preds = %5
  %20 = load i8*, i8** %7, align 8
  %21 = load i32, i32* %11, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i8, i8* %20, i64 %22
  %24 = load i8, i8* %23, align 1
  %25 = trunc i8 %24 to i1
  br i1 %25, label %26, label %40

26:                                               ; preds = %19
  %27 = load i8*, i8** %6, align 8
  %28 = load i32, i32* %11, align 4
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds i8, i8* %27, i64 %29
  store i8 1, i8* %30, align 1
  %31 = load i8*, i8** %8, align 8
  %32 = load i32, i32* %11, align 4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds i8, i8* %31, i64 %33
  store i8 1, i8* %34, align 1
  %35 = load i8*, i8** %9, align 8
  store i8 1, i8* %35, align 1
  %36 = load i8*, i8** %7, align 8
  %37 = load i32, i32* %11, align 4
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds i8, i8* %36, i64 %38
  store i8 0, i8* %39, align 1
  br label %40

40:                                               ; preds = %26, %19, %5
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_52" "target-features"="+ptx64,+sm_52" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!nvvm.annotations = !{!3, !4, !5, !6, !5, !7, !7, !7, !7, !8, !8, !7}
!llvm.ident = !{!9}
!nvvmir.version = !{!10}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{void (%struct.Node*, i32*, i8*, i8*, i8*, i32*, i32)* @_Z6KernelP4NodePiPbS2_S2_S1_i, !"kernel", i32 1}
!4 = !{void (i8*, i8*, i8*, i8*, i32)* @_Z7Kernel2PbS_S_S_i, !"kernel", i32 1}
!5 = !{null, !"align", i32 8}
!6 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!7 = !{null, !"align", i32 16}
!8 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!9 = !{!"clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)"}
!10 = !{i32 1, i32 4}
!11 = !{i32 0, i32 2147483647}
!12 = !{i32 0, i32 1024}

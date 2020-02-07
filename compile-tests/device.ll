; ModuleID = 'device.bc'
source_filename = "input.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.cudaFuncAttributes = type { i64, i64, i64, i32, i32, i32, i32, i32, i32, i32 }
%printf_args = type { i32 }

@.str = private unnamed_addr constant [15 x i8] c"Hello from %d\0A\00", align 1

; Function Attrs: nounwind
define weak dso_local i32 @cudaMalloc(i8** %0, i64 %1) local_unnamed_addr #0 !dbg !16 {
  ret i32 999, !dbg !19
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaFuncGetAttributes(%struct.cudaFuncAttributes* %0, i8* %1) local_unnamed_addr #0 !dbg !20 {
  ret i32 999, !dbg !21
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaDeviceGetAttribute(i32* %0, i32 %1, i32 %2) local_unnamed_addr #0 !dbg !22 {
  ret i32 999, !dbg !23
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaGetDevice(i32* %0) local_unnamed_addr #0 !dbg !24 {
  ret i32 999, !dbg !25
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessor(i32* %0, i8* %1, i32 %2, i64 %3) local_unnamed_addr #0 !dbg !26 {
  ret i32 999, !dbg !27
}

; Function Attrs: nounwind
define weak dso_local i32 @cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(i32* %0, i8* %1, i32 %2, i64 %3, i32 %4) local_unnamed_addr #0 !dbg !28 {
  ret i32 999, !dbg !29
}

; Function Attrs: norecurse nounwind readnone
define dso_local void @_Z14another_kernelv() local_unnamed_addr #1 !dbg !30 {
  ret void, !dbg !32
}

; Function Attrs: norecurse nounwind readnone
define dso_local void @_Z9devicefunv() local_unnamed_addr #1 !dbg !33 {
  ret void, !dbg !34
}

; Function Attrs: nounwind
define dso_local void @testKernel() local_unnamed_addr #0 !dbg !35 {
  %1 = alloca %printf_args, align 8
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !dbg !36, !range !40
  %3 = getelementptr inbounds %printf_args, %printf_args* %1, i64 0, i32 0, !dbg !41
  store i32 %2, i32* %3, align 8, !dbg !41
  %4 = bitcast %printf_args* %1 to i8*, !dbg !41
  %5 = call i32 @vprintf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), i8* nonnull %4) #3, !dbg !41
  ret void, !dbg !42
}

declare i32 @vprintf(i8*, i8*) local_unnamed_addr

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_52" "target-features"="+ptx64,+sm_52" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_52" "target-features"="+ptx64,+sm_52" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!nvvm.annotations = !{!8, !9, !10, !11, !10, !12, !12, !12, !12, !13, !13, !12}
!llvm.ident = !{!14}
!nvvmir.version = !{!15}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 1]}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !6, producer: "clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !7, nameTableKind: None)
!6 = !DIFile(filename: "input.cu", directory: "/home/gkarlos/Projects/msc-project/kerma-static-analysis/compile-tests/mk")
!7 = !{}
!8 = !{void ()* @_Z14another_kernelv, !"kernel", i32 1}
!9 = !{void ()* @testKernel, !"kernel", i32 1}
!10 = !{null, !"align", i32 8}
!11 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!12 = !{null, !"align", i32 16}
!13 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!14 = !{!"clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)"}
!15 = !{i32 1, i32 4}
!16 = distinct !DISubprogram(name: "cudaMalloc", scope: !17, file: !17, line: 75, type: !18, scopeLine: 76, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!17 = !DIFile(filename: "/usr/local/cuda/include/cuda_device_runtime_api.h", directory: "")
!18 = !DISubroutineType(types: !7)
!19 = !DILocation(line: 77, column: 3, scope: !16)
!20 = distinct !DISubprogram(name: "cudaFuncGetAttributes", scope: !17, file: !17, line: 80, type: !18, scopeLine: 81, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!21 = !DILocation(line: 82, column: 3, scope: !20)
!22 = distinct !DISubprogram(name: "cudaDeviceGetAttribute", scope: !17, file: !17, line: 85, type: !18, scopeLine: 86, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!23 = !DILocation(line: 87, column: 3, scope: !22)
!24 = distinct !DISubprogram(name: "cudaGetDevice", scope: !17, file: !17, line: 90, type: !18, scopeLine: 91, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!25 = !DILocation(line: 92, column: 3, scope: !24)
!26 = distinct !DISubprogram(name: "cudaOccupancyMaxActiveBlocksPerMultiprocessor", scope: !17, file: !17, line: 95, type: !18, scopeLine: 96, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!27 = !DILocation(line: 97, column: 3, scope: !26)
!28 = distinct !DISubprogram(name: "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", scope: !17, file: !17, line: 100, type: !18, scopeLine: 101, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!29 = !DILocation(line: 102, column: 3, scope: !28)
!30 = distinct !DISubprogram(name: "another_kernel", scope: !31, file: !31, line: 1, type: !18, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!31 = !DIFile(filename: "./test_include.cuh", directory: "/home/gkarlos/Projects/msc-project/kerma-static-analysis/compile-tests/mk")
!32 = !DILocation(line: 3, column: 1, scope: !30)
!33 = distinct !DISubprogram(name: "devicefun", scope: !6, file: !6, line: 6, type: !18, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!34 = !DILocation(line: 8, column: 1, scope: !33)
!35 = distinct !DISubprogram(name: "testKernel", scope: !6, file: !6, line: 10, type: !18, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!36 = !DILocation(line: 53, column: 3, scope: !37, inlinedAt: !39)
!37 = distinct !DISubprogram(name: "__fetch_builtin_x", scope: !38, file: !38, line: 53, type: !18, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !5, retainedNodes: !7)
!38 = !DIFile(filename: "s/llvm/10/lib/clang/10.0.0/include/__clang_cuda_builtin_vars.h", directory: "/home/gkarlos")
!39 = distinct !DILocation(line: 11, column: 29, scope: !35)
!40 = !{i32 0, i32 1024}
!41 = !DILocation(line: 11, column: 3, scope: !35)
!42 = !DILocation(line: 12, column: 1, scope: !35)

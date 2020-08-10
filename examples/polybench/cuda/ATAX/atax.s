	.text
	.file	"atax.cu"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z7rtclockv
.LCPI0_0:
	.quad	4517329193108106637     # double 9.9999999999999995E-7
	.text
	.globl	_Z7rtclockv
	.p2align	4, 0x90
	.type	_Z7rtclockv,@function
_Z7rtclockv:                            # @_Z7rtclockv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	leaq	-24(%rbp), %rdi
	leaq	-8(%rbp), %rsi
	callq	gettimeofday
	movl	%eax, -28(%rbp)
	cmpl	$0, -28(%rbp)
	je	.LBB0_2
# %bb.1:
	movl	-28(%rbp), %esi
	movabsq	$.L.str, %rdi
	movb	$0, %al
	callq	printf
.LBB0_2:
	movsd	.LCPI0_0(%rip), %xmm0   # xmm0 = mem[0],zero
	cvtsi2sdq	-24(%rbp), %xmm1
	cvtsi2sdq	-16(%rbp), %xmm2
	mulsd	%xmm0, %xmm2
	addsd	%xmm2, %xmm1
	movaps	%xmm1, %xmm0
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	_Z7rtclockv, .Lfunc_end0-_Z7rtclockv
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               # -- Begin function _Z6absValf
.LCPI1_0:
	.long	3212836864              # float -1
	.text
	.globl	_Z6absValf
	.p2align	4, 0x90
	.type	_Z6absValf,@function
_Z6absValf:                             # @_Z6absValf
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movss	%xmm0, -8(%rbp)
	xorps	%xmm0, %xmm0
	ucomiss	-8(%rbp), %xmm0
	jbe	.LBB1_2
# %bb.1:
	movss	.LCPI1_0(%rip), %xmm0   # xmm0 = mem[0],zero,zero,zero
	mulss	-8(%rbp), %xmm0
	movss	%xmm0, -4(%rbp)
	jmp	.LBB1_3
.LBB1_2:
	movss	-8(%rbp), %xmm0         # xmm0 = mem[0],zero,zero,zero
	movss	%xmm0, -4(%rbp)
.LBB1_3:
	movss	-4(%rbp), %xmm0         # xmm0 = mem[0],zero,zero,zero
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	_Z6absValf, .Lfunc_end1-_Z6absValf
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z11percentDiffdd
.LCPI2_0:
	.quad	4576918229304087675     # double 0.01
.LCPI2_2:
	.quad	4487126258294980608     # double 9.9999999392252903E-9
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2
.LCPI2_1:
	.long	1120403456              # float 100
	.text
	.globl	_Z11percentDiffdd
	.p2align	4, 0x90
	.type	_Z11percentDiffdd,@function
_Z11percentDiffdd:                      # @_Z11percentDiffdd
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movsd	%xmm0, -16(%rbp)
	movsd	%xmm1, -24(%rbp)
	movsd	-16(%rbp), %xmm0        # xmm0 = mem[0],zero
	cvtsd2ss	%xmm0, %xmm0
	callq	_Z6absValf
	movsd	.LCPI2_0(%rip), %xmm1   # xmm1 = mem[0],zero
	cvtss2sd	%xmm0, %xmm0
	ucomisd	%xmm0, %xmm1
	jbe	.LBB2_3
# %bb.1:
	movsd	-24(%rbp), %xmm0        # xmm0 = mem[0],zero
	cvtsd2ss	%xmm0, %xmm0
	callq	_Z6absValf
	movsd	.LCPI2_0(%rip), %xmm1   # xmm1 = mem[0],zero
	cvtss2sd	%xmm0, %xmm0
	ucomisd	%xmm0, %xmm1
	jbe	.LBB2_3
# %bb.2:
	xorps	%xmm0, %xmm0
	movss	%xmm0, -4(%rbp)
	jmp	.LBB2_4
.LBB2_3:
	movsd	-16(%rbp), %xmm0        # xmm0 = mem[0],zero
	subsd	-24(%rbp), %xmm0
	cvtsd2ss	%xmm0, %xmm0
	callq	_Z6absValf
	movsd	.LCPI2_2(%rip), %xmm1   # xmm1 = mem[0],zero
	addsd	-16(%rbp), %xmm1
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm0, -28(%rbp)        # 4-byte Spill
	movaps	%xmm1, %xmm0
	callq	_Z6absValf
	movss	-28(%rbp), %xmm1        # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
	callq	_Z6absValf
	movss	.LCPI2_1(%rip), %xmm1   # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm0, %xmm1
	movss	%xmm1, -4(%rbp)
.LBB2_4:
	movss	-4(%rbp), %xmm0         # xmm0 = mem[0],zero,zero,zero
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	_Z11percentDiffdd, .Lfunc_end2-_Z11percentDiffdd
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z10init_arrayPfS_
.LCPI3_0:
	.quad	4614256656552045848     # double 3.1415926535897931
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2
.LCPI3_1:
	.long	1166016512              # float 4096
	.text
	.globl	_Z10init_arrayPfS_
	.p2align	4, 0x90
	.type	_Z10init_arrayPfS_,@function
_Z10init_arrayPfS_:                     # @_Z10init_arrayPfS_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$0, -20(%rbp)
.LBB3_1:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_3 Depth 2
	cmpl	$4096, -20(%rbp)        # imm = 0x1000
	jge	.LBB3_8
# %bb.2:                                #   in Loop: Header=BB3_1 Depth=1
	movsd	.LCPI3_0(%rip), %xmm0   # xmm0 = mem[0],zero
	cvtsi2sdl	-20(%rbp), %xmm1
	mulsd	%xmm0, %xmm1
	cvtsd2ss	%xmm1, %xmm0
	movq	-8(%rbp), %rax
	movslq	-20(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
	movl	$0, -24(%rbp)
.LBB3_3:                                #   Parent Loop BB3_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpl	$4096, -24(%rbp)        # imm = 0x1000
	jge	.LBB3_6
# %bb.4:                                #   in Loop: Header=BB3_3 Depth=2
	movss	.LCPI3_1(%rip), %xmm0   # xmm0 = mem[0],zero,zero,zero
	cvtsi2ssl	-20(%rbp), %xmm1
	cvtsi2ssl	-24(%rbp), %xmm2
	mulss	%xmm2, %xmm1
	divss	%xmm0, %xmm1
	movq	-16(%rbp), %rax
	movl	-20(%rbp), %ecx
	shll	$12, %ecx
	addl	-24(%rbp), %ecx
	movslq	%ecx, %rdx
	movss	%xmm1, (%rax,%rdx,4)
# %bb.5:                                #   in Loop: Header=BB3_3 Depth=2
	movl	-24(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -24(%rbp)
	jmp	.LBB3_3
.LBB3_6:                                #   in Loop: Header=BB3_1 Depth=1
	jmp	.LBB3_7
.LBB3_7:                                #   in Loop: Header=BB3_1 Depth=1
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	jmp	.LBB3_1
.LBB3_8:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end3:
	.size	_Z10init_arrayPfS_, .Lfunc_end3-_Z10init_arrayPfS_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z14compareResultsPfS_
.LCPI4_0:
	.quad	4602678819172646912     # double 0.5
	.text
	.globl	_Z14compareResultsPfS_
	.p2align	4, 0x90
	.type	_Z14compareResultsPfS_,@function
_Z14compareResultsPfS_:                 # @_Z14compareResultsPfS_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$0, -24(%rbp)
	movl	$0, -20(%rbp)
.LBB4_1:                                # =>This Inner Loop Header: Depth=1
	cmpl	$4096, -20(%rbp)        # imm = 0x1000
	jge	.LBB4_6
# %bb.2:                                #   in Loop: Header=BB4_1 Depth=1
	movq	-8(%rbp), %rax
	movslq	-20(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movq	-16(%rbp), %rax
	movslq	-20(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm1    # xmm1 = mem[0],zero,zero,zero
	cvtss2sd	%xmm1, %xmm1
	callq	_Z11percentDiffdd
	movsd	.LCPI4_0(%rip), %xmm1   # xmm1 = mem[0],zero
	cvtss2sd	%xmm0, %xmm0
	ucomisd	%xmm1, %xmm0
	jbe	.LBB4_4
# %bb.3:                                #   in Loop: Header=BB4_1 Depth=1
	movl	-24(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -24(%rbp)
.LBB4_4:                                #   in Loop: Header=BB4_1 Depth=1
	jmp	.LBB4_5
.LBB4_5:                                #   in Loop: Header=BB4_1 Depth=1
	movl	-20(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -20(%rbp)
	jmp	.LBB4_1
.LBB4_6:
	movsd	.LCPI4_0(%rip), %xmm0   # xmm0 = mem[0],zero
	movl	-24(%rbp), %esi
	movabsq	$.L.str.1, %rdi
	movb	$1, %al
	callq	printf
	addq	$32, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end4:
	.size	_Z14compareResultsPfS_, .Lfunc_end4-_Z14compareResultsPfS_
	.cfi_endproc
                                        # -- End function
	.globl	_Z13GPU_argv_initv      # -- Begin function _Z13GPU_argv_initv
	.p2align	4, 0x90
	.type	_Z13GPU_argv_initv,@function
_Z13GPU_argv_initv:                     # @_Z13GPU_argv_initv
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$720, %rsp              # imm = 0x2D0
	xorl	%esi, %esi
	leaq	-712(%rbp), %rdi
	callq	cudaGetDeviceProperties
	xorl	%esi, %esi
	leaq	-712(%rbp), %rdx
	movabsq	$.L.str.2, %rdi
	movl	%eax, -716(%rbp)        # 4-byte Spill
	movb	$0, %al
	callq	printf
	xorl	%edi, %edi
	movl	%eax, -720(%rbp)        # 4-byte Spill
	callq	cudaSetDevice
	addq	$720, %rsp              # imm = 0x2D0
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end5:
	.size	_Z13GPU_argv_initv, .Lfunc_end5-_Z13GPU_argv_initv
	.cfi_endproc
                                        # -- End function
	.globl	_Z12atax_kernel1PfS_S_  # -- Begin function _Z12atax_kernel1PfS_S_
	.p2align	4, 0x90
	.type	_Z12atax_kernel1PfS_S_,@function
_Z12atax_kernel1PfS_S_:                 # @_Z12atax_kernel1PfS_S_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$176, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -120(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -112(%rbp)
	leaq	-40(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	leaq	-64(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	callq	__cudaPopCallConfiguration
	movabsq	$_Z12atax_kernel1PfS_S_, %rcx
	movq	-64(%rbp), %rdx
	movq	-72(%rbp), %rsi
	movq	-40(%rbp), %rdi
	movq	%rdi, -88(%rbp)
	movl	-32(%rbp), %r8d
	movl	%r8d, -80(%rbp)
	movq	-88(%rbp), %rdi
	movl	-80(%rbp), %r8d
	movq	-56(%rbp), %r9
	movq	%r9, -104(%rbp)
	movl	-48(%rbp), %r10d
	movl	%r10d, -96(%rbp)
	movq	-104(%rbp), %r9
	movl	-96(%rbp), %r10d
	movq	%rdi, -136(%rbp)        # 8-byte Spill
	movq	%rcx, %rdi
	movq	-136(%rbp), %rcx        # 8-byte Reload
	movq	%rsi, -144(%rbp)        # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -152(%rbp)        # 8-byte Spill
	movl	%r8d, %edx
	movq	%r9, %rcx
	movl	%r10d, %r8d
	leaq	-128(%rbp), %r9
	movq	-152(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movq	-144(%rbp), %r11        # 8-byte Reload
	movq	%r11, 8(%rsp)
	movl	%eax, -156(%rbp)        # 4-byte Spill
	callq	cudaLaunchKernel
# %bb.1:
	addq	$176, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end6:
	.size	_Z12atax_kernel1PfS_S_, .Lfunc_end6-_Z12atax_kernel1PfS_S_
	.cfi_endproc
                                        # -- End function
	.globl	_Z12atax_kernel2PfS_S_  # -- Begin function _Z12atax_kernel2PfS_S_
	.p2align	4, 0x90
	.type	_Z12atax_kernel2PfS_S_,@function
_Z12atax_kernel2PfS_S_:                 # @_Z12atax_kernel2PfS_S_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$176, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, -128(%rbp)
	leaq	-16(%rbp), %rax
	movq	%rax, -120(%rbp)
	leaq	-24(%rbp), %rax
	movq	%rax, -112(%rbp)
	leaq	-40(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	leaq	-64(%rbp), %rdx
	leaq	-72(%rbp), %rcx
	callq	__cudaPopCallConfiguration
	movabsq	$_Z12atax_kernel2PfS_S_, %rcx
	movq	-64(%rbp), %rdx
	movq	-72(%rbp), %rsi
	movq	-40(%rbp), %rdi
	movq	%rdi, -88(%rbp)
	movl	-32(%rbp), %r8d
	movl	%r8d, -80(%rbp)
	movq	-88(%rbp), %rdi
	movl	-80(%rbp), %r8d
	movq	-56(%rbp), %r9
	movq	%r9, -104(%rbp)
	movl	-48(%rbp), %r10d
	movl	%r10d, -96(%rbp)
	movq	-104(%rbp), %r9
	movl	-96(%rbp), %r10d
	movq	%rdi, -136(%rbp)        # 8-byte Spill
	movq	%rcx, %rdi
	movq	-136(%rbp), %rcx        # 8-byte Reload
	movq	%rsi, -144(%rbp)        # 8-byte Spill
	movq	%rcx, %rsi
	movq	%rdx, -152(%rbp)        # 8-byte Spill
	movl	%r8d, %edx
	movq	%r9, %rcx
	movl	%r10d, %r8d
	leaq	-128(%rbp), %r9
	movq	-152(%rbp), %r11        # 8-byte Reload
	movq	%r11, (%rsp)
	movq	-144(%rbp), %r11        # 8-byte Reload
	movq	%r11, 8(%rsp)
	movl	%eax, -156(%rbp)        # 4-byte Spill
	callq	cudaLaunchKernel
# %bb.1:
	addq	$176, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end7:
	.size	_Z12atax_kernel2PfS_S_, .Lfunc_end7-_Z12atax_kernel2PfS_S_
	.cfi_endproc
                                        # -- End function
	.globl	_Z8atax_cpuPfS_S_S_     # -- Begin function _Z8atax_cpuPfS_S_S_
	.p2align	4, 0x90
	.type	_Z8atax_cpuPfS_S_S_,@function
_Z8atax_cpuPfS_S_S_:                    # @_Z8atax_cpuPfS_S_S_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movl	$0, -36(%rbp)
.LBB8_1:                                # =>This Inner Loop Header: Depth=1
	cmpl	$4096, -36(%rbp)        # imm = 0x1000
	jge	.LBB8_4
# %bb.2:                                #   in Loop: Header=BB8_1 Depth=1
	movq	-24(%rbp), %rax
	movslq	-36(%rbp), %rcx
	xorps	%xmm0, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
# %bb.3:                                #   in Loop: Header=BB8_1 Depth=1
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	jmp	.LBB8_1
.LBB8_4:
	movl	$0, -36(%rbp)
.LBB8_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB8_7 Depth 2
                                        #     Child Loop BB8_11 Depth 2
	cmpl	$4096, -36(%rbp)        # imm = 0x1000
	jge	.LBB8_16
# %bb.6:                                #   in Loop: Header=BB8_5 Depth=1
	movq	-32(%rbp), %rax
	movslq	-36(%rbp), %rcx
	xorps	%xmm0, %xmm0
	movss	%xmm0, (%rax,%rcx,4)
	movl	$0, -40(%rbp)
.LBB8_7:                                #   Parent Loop BB8_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpl	$4096, -40(%rbp)        # imm = 0x1000
	jge	.LBB8_10
# %bb.8:                                #   in Loop: Header=BB8_7 Depth=2
	movq	-32(%rbp), %rax
	movslq	-36(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	movq	-8(%rbp), %rax
	movl	-36(%rbp), %edx
	shll	$12, %edx
	addl	-40(%rbp), %edx
	movslq	%edx, %rcx
	movss	(%rax,%rcx,4), %xmm1    # xmm1 = mem[0],zero,zero,zero
	movq	-16(%rbp), %rax
	movslq	-40(%rbp), %rcx
	mulss	(%rax,%rcx,4), %xmm1
	addss	%xmm1, %xmm0
	movq	-32(%rbp), %rax
	movslq	-36(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
# %bb.9:                                #   in Loop: Header=BB8_7 Depth=2
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
	jmp	.LBB8_7
.LBB8_10:                               #   in Loop: Header=BB8_5 Depth=1
	movl	$0, -40(%rbp)
.LBB8_11:                               #   Parent Loop BB8_5 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpl	$4096, -40(%rbp)        # imm = 0x1000
	jge	.LBB8_14
# %bb.12:                               #   in Loop: Header=BB8_11 Depth=2
	movq	-24(%rbp), %rax
	movslq	-40(%rbp), %rcx
	movss	(%rax,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	movq	-8(%rbp), %rax
	movl	-36(%rbp), %edx
	shll	$12, %edx
	addl	-40(%rbp), %edx
	movslq	%edx, %rcx
	movss	(%rax,%rcx,4), %xmm1    # xmm1 = mem[0],zero,zero,zero
	movq	-32(%rbp), %rax
	movslq	-36(%rbp), %rcx
	mulss	(%rax,%rcx,4), %xmm1
	addss	%xmm1, %xmm0
	movq	-24(%rbp), %rax
	movslq	-40(%rbp), %rcx
	movss	%xmm0, (%rax,%rcx,4)
# %bb.13:                               #   in Loop: Header=BB8_11 Depth=2
	movl	-40(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -40(%rbp)
	jmp	.LBB8_11
.LBB8_14:                               #   in Loop: Header=BB8_5 Depth=1
	jmp	.LBB8_15
.LBB8_15:                               #   in Loop: Header=BB8_5 Depth=1
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	jmp	.LBB8_5
.LBB8_16:
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end8:
	.size	_Z8atax_cpuPfS_S_S_, .Lfunc_end8-_Z8atax_cpuPfS_S_S_
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2               # -- Begin function _Z7ataxGpuPfS_S_S_S_
.LCPI9_0:
	.long	1166016512              # float 4096
.LCPI9_1:
	.long	1593835520              # float 9.22337203E+18
	.text
	.globl	_Z7ataxGpuPfS_S_S_S_
	.p2align	4, 0x90
	.type	_Z7ataxGpuPfS_S_S_S_,@function
_Z7ataxGpuPfS_S_S_S_:                   # @_Z7ataxGpuPfS_S_S_S_
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$400, %rsp              # imm = 0x190
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	%r8, -40(%rbp)
	leaq	-64(%rbp), %rdi
	movl	$67108864, %eax         # imm = 0x4000000
	movq	%rax, %rsi
	movq	%rax, -272(%rbp)        # 8-byte Spill
	callq	cudaMalloc
	leaq	-72(%rbp), %rdi
	movl	$16384, %ecx            # imm = 0x4000
	movq	%rcx, %rsi
	movl	%eax, -276(%rbp)        # 4-byte Spill
	movq	%rcx, -288(%rbp)        # 8-byte Spill
	callq	cudaMalloc
	leaq	-80(%rbp), %rdi
	movq	-288(%rbp), %rsi        # 8-byte Reload
	movl	%eax, -292(%rbp)        # 4-byte Spill
	callq	cudaMalloc
	leaq	-88(%rbp), %rdi
	movq	-288(%rbp), %rsi        # 8-byte Reload
	movl	%eax, -296(%rbp)        # 4-byte Spill
	callq	cudaMalloc
	movq	-64(%rbp), %rdi
	movq	-8(%rbp), %rsi
	movl	$1, %r9d
	movq	-272(%rbp), %rdx        # 8-byte Reload
	movl	%r9d, %ecx
	movl	%eax, -300(%rbp)        # 4-byte Spill
	movl	%r9d, -304(%rbp)        # 4-byte Spill
	callq	cudaMemcpy
	movq	-72(%rbp), %rdi
	movq	-16(%rbp), %rsi
	movq	-288(%rbp), %rdx        # 8-byte Reload
	movl	-304(%rbp), %ecx        # 4-byte Reload
	movl	%eax, -308(%rbp)        # 4-byte Spill
	callq	cudaMemcpy
	movq	-80(%rbp), %rdi
	movq	-24(%rbp), %rsi
	movq	-288(%rbp), %rdx        # 8-byte Reload
	movl	-304(%rbp), %ecx        # 4-byte Reload
	movl	%eax, -312(%rbp)        # 4-byte Spill
	callq	cudaMemcpy
	movq	-88(%rbp), %rdi
	movq	-32(%rbp), %rsi
	movq	-288(%rbp), %rdx        # 8-byte Reload
	movl	-304(%rbp), %ecx        # 4-byte Reload
	movl	%eax, -316(%rbp)        # 4-byte Spill
	callq	cudaMemcpy
	leaq	-104(%rbp), %rdi
	movl	$256, %esi              # imm = 0x100
	movl	-304(%rbp), %edx        # 4-byte Reload
	movl	-304(%rbp), %ecx        # 4-byte Reload
	movl	%eax, -320(%rbp)        # 4-byte Spill
	callq	_ZN4dim3C2Ejjj
	movl	-104(%rbp), %eax
	movl	%eax, %edi
	cvtsi2ss	%rdi, %xmm0
	movss	.LCPI9_0(%rip), %xmm1   # xmm1 = mem[0],zero,zero,zero
	movaps	%xmm1, %xmm2
	divss	%xmm0, %xmm2
	movaps	%xmm2, %xmm0
	movss	%xmm1, -324(%rbp)       # 4-byte Spill
	callq	_ZSt4ceilf
	movss	.LCPI9_1(%rip), %xmm1   # xmm1 = mem[0],zero,zero,zero
	movaps	%xmm0, %xmm2
	subss	%xmm1, %xmm2
	cvttss2si	%xmm2, %rdi
	movabsq	$-9223372036854775808, %r8 # imm = 0x8000000000000000
	xorq	%r8, %rdi
	cvttss2si	%xmm0, %r10
	ucomiss	%xmm1, %xmm0
	cmovbq	%r10, %rdi
                                        # kill: def $edi killed $edi killed $rdi
	leaq	-120(%rbp), %r10
	movl	%edi, -328(%rbp)        # 4-byte Spill
	movq	%r10, %rdi
	movl	-328(%rbp), %esi        # 4-byte Reload
	movl	-304(%rbp), %edx        # 4-byte Reload
	movl	-304(%rbp), %ecx        # 4-byte Reload
	movss	%xmm1, -332(%rbp)       # 4-byte Spill
	movq	%r8, -344(%rbp)         # 8-byte Spill
	callq	_ZN4dim3C2Ejjj
	movl	-104(%rbp), %eax
	movl	%eax, %edi
	cvtsi2ss	%rdi, %xmm0
	movss	-324(%rbp), %xmm1       # 4-byte Reload
                                        # xmm1 = mem[0],zero,zero,zero
	divss	%xmm0, %xmm1
	movaps	%xmm1, %xmm0
	callq	_ZSt4ceilf
	movaps	%xmm0, %xmm1
	movss	-332(%rbp), %xmm2       # 4-byte Reload
                                        # xmm2 = mem[0],zero,zero,zero
	subss	%xmm2, %xmm1
	cvttss2si	%xmm1, %rdi
	movq	-344(%rbp), %r8         # 8-byte Reload
	xorq	%r8, %rdi
	cvttss2si	%xmm0, %r10
	ucomiss	%xmm2, %xmm0
	cmovbq	%r10, %rdi
                                        # kill: def $edi killed $edi killed $rdi
	leaq	-136(%rbp), %r10
	movl	%edi, -348(%rbp)        # 4-byte Spill
	movq	%r10, %rdi
	movl	-348(%rbp), %esi        # 4-byte Reload
	movl	$1, %eax
	movl	%eax, %edx
	movl	%eax, %ecx
	callq	_ZN4dim3C2Ejjj
	callq	_Z7rtclockv
	xorl	%eax, %eax
	movl	%eax, %edi
	movsd	%xmm0, -48(%rbp)
	movq	-120(%rbp), %r8
	movq	%r8, -152(%rbp)
	movl	-112(%rbp), %eax
	movl	%eax, -144(%rbp)
	movq	-104(%rbp), %r8
	movq	%r8, -168(%rbp)
	movl	-96(%rbp), %eax
	movl	%eax, -160(%rbp)
	movq	-152(%rbp), %r8
	movq	%r8, -184(%rbp)
	movl	-144(%rbp), %eax
	movl	%eax, -176(%rbp)
	movq	-184(%rbp), %r8
	movl	-176(%rbp), %esi
	movq	-168(%rbp), %r10
	movq	%r10, -200(%rbp)
	movl	-160(%rbp), %eax
	movl	%eax, -192(%rbp)
	movq	-200(%rbp), %rdx
	movl	-192(%rbp), %ecx
	movq	%rdi, -360(%rbp)        # 8-byte Spill
	movq	%r8, %rdi
	movq	-360(%rbp), %r8         # 8-byte Reload
	movq	-360(%rbp), %r9         # 8-byte Reload
	callq	__cudaPushCallConfiguration
	cmpl	$0, %eax
	jne	.LBB9_2
# %bb.1:
	movq	-64(%rbp), %rdi
	movq	-72(%rbp), %rsi
	movq	-88(%rbp), %rdx
	callq	_Z12atax_kernel1PfS_S_
.LBB9_2:
	callq	cudaThreadSynchronize
	xorl	%ecx, %ecx
	movl	%ecx, %edx
	movq	-136(%rbp), %rsi
	movq	%rsi, -216(%rbp)
	movl	-128(%rbp), %ecx
	movl	%ecx, -208(%rbp)
	movq	-104(%rbp), %rsi
	movq	%rsi, -232(%rbp)
	movl	-96(%rbp), %ecx
	movl	%ecx, -224(%rbp)
	movq	-216(%rbp), %rsi
	movq	%rsi, -248(%rbp)
	movl	-208(%rbp), %ecx
	movl	%ecx, -240(%rbp)
	movq	-248(%rbp), %rdi
	movl	-240(%rbp), %esi
	movq	-232(%rbp), %r8
	movq	%r8, -264(%rbp)
	movl	-224(%rbp), %ecx
	movl	%ecx, -256(%rbp)
	movq	-264(%rbp), %r8
	movl	-256(%rbp), %ecx
	movq	%rdx, -368(%rbp)        # 8-byte Spill
	movq	%r8, %rdx
	movq	-368(%rbp), %r8         # 8-byte Reload
	movq	-368(%rbp), %r9         # 8-byte Reload
	movl	%eax, -372(%rbp)        # 4-byte Spill
	callq	__cudaPushCallConfiguration
	cmpl	$0, %eax
	jne	.LBB9_4
# %bb.3:
	movq	-64(%rbp), %rdi
	movq	-80(%rbp), %rsi
	movq	-88(%rbp), %rdx
	callq	_Z12atax_kernel2PfS_S_
.LBB9_4:
	callq	cudaThreadSynchronize
	movl	%eax, -376(%rbp)        # 4-byte Spill
	callq	_Z7rtclockv
	movsd	%xmm0, -56(%rbp)
	movq	stdout, %rdi
	movsd	-56(%rbp), %xmm0        # xmm0 = mem[0],zero
	subsd	-48(%rbp), %xmm0
	movabsq	$.L.str.3, %rsi
	movb	$1, %al
	callq	fprintf
	movq	-40(%rbp), %rcx
	movq	-80(%rbp), %rdx
	movq	%rcx, %rdi
	movq	%rdx, %rsi
	movl	$16384, %edx            # imm = 0x4000
	movl	$2, %ecx
	movl	%eax, -380(%rbp)        # 4-byte Spill
	callq	cudaMemcpy
	movq	-64(%rbp), %rdx
	movq	%rdx, %rdi
	movl	%eax, -384(%rbp)        # 4-byte Spill
	callq	cudaFree
	movq	-72(%rbp), %rdx
	movq	%rdx, %rdi
	movl	%eax, -388(%rbp)        # 4-byte Spill
	callq	cudaFree
	movq	-80(%rbp), %rdx
	movq	%rdx, %rdi
	movl	%eax, -392(%rbp)        # 4-byte Spill
	callq	cudaFree
	movq	-88(%rbp), %rdx
	movq	%rdx, %rdi
	movl	%eax, -396(%rbp)        # 4-byte Spill
	callq	cudaFree
	addq	$400, %rsp              # imm = 0x190
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end9:
	.size	_Z7ataxGpuPfS_S_S_S_, .Lfunc_end9-_Z7ataxGpuPfS_S_S_S_
	.cfi_endproc
                                        # -- End function
	.section	.text._ZN4dim3C2Ejjj,"axG",@progbits,_ZN4dim3C2Ejjj,comdat
	.weak	_ZN4dim3C2Ejjj          # -- Begin function _ZN4dim3C2Ejjj
	.p2align	4, 0x90
	.type	_ZN4dim3C2Ejjj,@function
_ZN4dim3C2Ejjj:                         # @_ZN4dim3C2Ejjj
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movl	%esi, -12(%rbp)
	movl	%edx, -16(%rbp)
	movl	%ecx, -20(%rbp)
	movq	-8(%rbp), %rax
	movl	-12(%rbp), %ecx
	movl	%ecx, (%rax)
	movl	-16(%rbp), %ecx
	movl	%ecx, 4(%rax)
	movl	-20(%rbp), %ecx
	movl	%ecx, 8(%rax)
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end10:
	.size	_ZN4dim3C2Ejjj, .Lfunc_end10-_ZN4dim3C2Ejjj
	.cfi_endproc
                                        # -- End function
	.section	.text._ZSt4ceilf,"axG",@progbits,_ZSt4ceilf,comdat
	.weak	_ZSt4ceilf              # -- Begin function _ZSt4ceilf
	.p2align	4, 0x90
	.type	_ZSt4ceilf,@function
_ZSt4ceilf:                             # @_ZSt4ceilf
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movss	%xmm0, -4(%rbp)
	movss	-4(%rbp), %xmm0         # xmm0 = mem[0],zero,zero,zero
	callq	ceilf
	addq	$16, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end11:
	.size	_ZSt4ceilf, .Lfunc_end11-_ZSt4ceilf
	.cfi_endproc
                                        # -- End function
	.text
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$80, %rsp
	movl	$0, -4(%rbp)
	movl	%edi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$67108864, %edi         # imm = 0x4000000
	callq	malloc
	movq	%rax, -40(%rbp)
	movl	$16384, %edi            # imm = 0x4000
	callq	malloc
	movq	%rax, -48(%rbp)
	movl	$16384, %edi            # imm = 0x4000
	callq	malloc
	movq	%rax, -56(%rbp)
	movl	$16384, %edi            # imm = 0x4000
	callq	malloc
	movq	%rax, -64(%rbp)
	movl	$16384, %edi            # imm = 0x4000
	callq	malloc
	movq	%rax, -72(%rbp)
	movq	-48(%rbp), %rdi
	movq	-40(%rbp), %rsi
	callq	_Z10init_arrayPfS_
	callq	_Z13GPU_argv_initv
	movq	-40(%rbp), %rdi
	movq	-48(%rbp), %rsi
	movq	-56(%rbp), %rdx
	movq	-72(%rbp), %rcx
	movq	-64(%rbp), %r8
	callq	_Z7ataxGpuPfS_S_S_S_
	callq	_Z7rtclockv
	movsd	%xmm0, -24(%rbp)
	movq	-40(%rbp), %rdi
	movq	-48(%rbp), %rsi
	movq	-56(%rbp), %rdx
	movq	-72(%rbp), %rcx
	callq	_Z8atax_cpuPfS_S_S_
	callq	_Z7rtclockv
	movsd	%xmm0, -32(%rbp)
	movq	stdout, %rdi
	movsd	-32(%rbp), %xmm0        # xmm0 = mem[0],zero
	subsd	-24(%rbp), %xmm0
	movabsq	$.L.str.4, %rsi
	movb	$1, %al
	callq	fprintf
	movq	-56(%rbp), %rdi
	movq	-64(%rbp), %rsi
	movl	%eax, -76(%rbp)         # 4-byte Spill
	callq	_Z14compareResultsPfS_
	movq	-40(%rbp), %rcx
	movq	%rcx, %rdi
	callq	free
	movq	-48(%rbp), %rcx
	movq	%rcx, %rdi
	callq	free
	movq	-56(%rbp), %rcx
	movq	%rcx, %rdi
	callq	free
	movq	-64(%rbp), %rcx
	movq	%rcx, %rdi
	callq	free
	movq	-72(%rbp), %rcx
	movq	%rcx, %rdi
	callq	free
	xorl	%eax, %eax
	addq	$80, %rsp
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end12:
	.size	main, .Lfunc_end12-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Error return from gettimeofday: %d"
	.size	.L.str, 35

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n"
	.size	.L.str.1, 74

	.type	.L.str.2,@object        # @.str.2
.L.str.2:
	.asciz	"setting device %d with name %s\n"
	.size	.L.str.2, 32

	.type	.L.str.3,@object        # @.str.3
.L.str.3:
	.asciz	"GPU Runtime: %0.6lfs\n"
	.size	.L.str.3, 22

	.type	.L.str.4,@object        # @.str.4
.L.str.4:
	.asciz	"CPU Runtime: %0.6lfs\n"
	.size	.L.str.4, 22


	.ident	"clang version 10.0.0 (https://llvm.org/git/clang.git 65acf43270ea2894dffa0d0b292b92402f80c8cb) (https://llvm.org/git/llvm.git 2c4ca6832fa6b306ee6a7010bfb80a3f2596f824)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z7rtclockv
	.addrsig_sym gettimeofday
	.addrsig_sym printf
	.addrsig_sym _Z6absValf
	.addrsig_sym _Z11percentDiffdd
	.addrsig_sym _Z10init_arrayPfS_
	.addrsig_sym _Z14compareResultsPfS_
	.addrsig_sym _Z13GPU_argv_initv
	.addrsig_sym cudaGetDeviceProperties
	.addrsig_sym cudaSetDevice
	.addrsig_sym _Z12atax_kernel1PfS_S_
	.addrsig_sym __cudaPopCallConfiguration
	.addrsig_sym cudaLaunchKernel
	.addrsig_sym _Z12atax_kernel2PfS_S_
	.addrsig_sym _Z8atax_cpuPfS_S_S_
	.addrsig_sym _Z7ataxGpuPfS_S_S_S_
	.addrsig_sym cudaMalloc
	.addrsig_sym cudaMemcpy
	.addrsig_sym _ZSt4ceilf
	.addrsig_sym __cudaPushCallConfiguration
	.addrsig_sym cudaThreadSynchronize
	.addrsig_sym fprintf
	.addrsig_sym cudaFree
	.addrsig_sym malloc
	.addrsig_sym free
	.addrsig_sym stdout

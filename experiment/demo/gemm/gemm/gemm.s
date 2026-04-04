	.text
	.file	"gemm.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function gemm
.LCPI0_0:
	.quad	0x3ff3333333333333              # double 1.2
.LCPI0_1:
	.quad	0x3ff8000000000000              # double 1.5
	.text
	.globl	gemm
	.p2align	4, 0x90
	.type	gemm,@function
gemm:                                   # @gemm
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movsd	.LCPI0_1(%rip), %xmm0           # xmm0 = mem[0],zero
	movsd	%xmm0, -48(%rbp)
	movsd	.LCPI0_0(%rip), %xmm0           # xmm0 = mem[0],zero
	movsd	%xmm0, -56(%rbp)
	movl	$0, -28(%rbp)
.LBB0_1:                                # %for.cond
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_9 Depth 3
	cmpl	$1000, -28(%rbp)                # imm = 0x3E8
	jge	.LBB0_16
# %bb.2:                                # %for.body
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	$0, -32(%rbp)
.LBB0_3:                                # %for.cond1
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	cmpl	$1100, -32(%rbp)                # imm = 0x44C
	jge	.LBB0_6
# %bb.4:                                # %for.body3
                                        #   in Loop: Header=BB0_3 Depth=2
	movsd	-56(%rbp), %xmm0                # xmm0 = mem[0],zero
	movq	-8(%rbp), %rax
	movslq	-28(%rbp), %rcx
	imulq	$8800, %rcx, %rcx               # imm = 0x2260
	addq	%rcx, %rax
	movslq	-32(%rbp), %rcx
	mulsd	(%rax,%rcx,8), %xmm0
	movsd	%xmm0, (%rax,%rcx,8)
# %bb.5:                                # %for.inc
                                        #   in Loop: Header=BB0_3 Depth=2
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -32(%rbp)
	jmp	.LBB0_3
.LBB0_6:                                # %for.end
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	$0, -36(%rbp)
.LBB0_7:                                # %for.cond6
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_9 Depth 3
	cmpl	$1200, -36(%rbp)                # imm = 0x4B0
	jge	.LBB0_14
# %bb.8:                                # %for.body8
                                        #   in Loop: Header=BB0_7 Depth=2
	movl	$0, -32(%rbp)
.LBB0_9:                                # %for.cond9
                                        #   Parent Loop BB0_1 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	cmpl	$1100, -32(%rbp)                # imm = 0x44C
	jge	.LBB0_12
# %bb.10:                               # %for.body11
                                        #   in Loop: Header=BB0_9 Depth=3
	movsd	-48(%rbp), %xmm0                # xmm0 = mem[0],zero
	movq	-16(%rbp), %rax
	movslq	-28(%rbp), %rcx
	imulq	$9600, %rcx, %rcx               # imm = 0x2580
	addq	%rcx, %rax
	movslq	-36(%rbp), %rcx
	mulsd	(%rax,%rcx,8), %xmm0
	movq	-24(%rbp), %rax
	movslq	-36(%rbp), %rcx
	imulq	$8800, %rcx, %rcx               # imm = 0x2260
	addq	%rcx, %rax
	movslq	-32(%rbp), %rcx
	movsd	(%rax,%rcx,8), %xmm2            # xmm2 = mem[0],zero
	movq	-8(%rbp), %rax
	movslq	-28(%rbp), %rcx
	imulq	$8800, %rcx, %rcx               # imm = 0x2260
	addq	%rcx, %rax
	movslq	-32(%rbp), %rcx
	movsd	(%rax,%rcx,8), %xmm1            # xmm1 = mem[0],zero
	mulsd	%xmm2, %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, (%rax,%rcx,8)
# %bb.11:                               # %for.inc26
                                        #   in Loop: Header=BB0_9 Depth=3
	movl	-32(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -32(%rbp)
	jmp	.LBB0_9
.LBB0_12:                               # %for.end28
                                        #   in Loop: Header=BB0_7 Depth=2
	jmp	.LBB0_13
.LBB0_13:                               # %for.inc29
                                        #   in Loop: Header=BB0_7 Depth=2
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -36(%rbp)
	jmp	.LBB0_7
.LBB0_14:                               # %for.end31
                                        #   in Loop: Header=BB0_1 Depth=1
	jmp	.LBB0_15
.LBB0_15:                               # %for.inc32
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	-28(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -28(%rbp)
	jmp	.LBB0_1
.LBB0_16:                               # %for.end34
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	gemm, .Lfunc_end0-gemm
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 18.0.0"
	.section	".note.GNU-stack","",@progbits
	.addrsig

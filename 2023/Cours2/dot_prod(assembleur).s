	.file	"dot_prod.c"
	.text
	.globl	dot_product_fma
	.type	dot_product_fma, @function
dot_product_fma:
.LFB4203:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	andq	$-32, %rsp
	subq	$288, %rsp
	movq	%rdi, 24(%rsp)
	movq	%rsi, 16(%rsp)
	movl	%edx, 12(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 280(%rsp)
	xorl	%eax, %eax
	vxorps	%xmm0, %xmm0, %xmm0
	vmovaps	%ymm0, 64(%rsp)
	movl	$0, 36(%rsp)
	jmp	.L3
.L7:
	movl	36(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	24(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	vmovups	(%rax), %ymm0
	vmovaps	%ymm0, 96(%rsp)
	movl	36(%rsp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	16(%rsp), %rax
	addq	%rdx, %rax
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %rax
	vmovups	(%rax), %ymm0
	vmovaps	%ymm0, 128(%rsp)
	vmovaps	64(%rsp), %ymm0
	vmovaps	96(%rsp), %ymm1
	vmovaps	%ymm1, 160(%rsp)
	vmovaps	128(%rsp), %ymm1
	vmovaps	%ymm1, 192(%rsp)
	vmovaps	%ymm0, 224(%rsp)
	vmovaps	192(%rsp), %ymm1
	vmovaps	224(%rsp), %ymm0
	vfmadd231ps	160(%rsp), %ymm1, %ymm0
	nop
	vmovaps	%ymm0, 64(%rsp)
	addl	$8, 36(%rsp)
.L3:
	movl	36(%rsp), %eax
	cmpl	12(%rsp), %eax
	jl	.L7
	vmovss	64(%rsp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmovsd	%xmm0, 40(%rsp)
	vmovss	68(%rsp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmovsd	40(%rsp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 40(%rsp)
	vmovss	72(%rsp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmovsd	40(%rsp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 40(%rsp)
	vmovss	76(%rsp), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	vmovsd	40(%rsp), %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 40(%rsp)
	vmovsd	40(%rsp), %xmm0
	vmovq	%xmm0, %rax
	movq	280(%rsp), %rdx
	subq	%fs:40, %rdx
	je	.L9
	call	__stack_chk_fail@PLT
.L9:
	vmovq	%rax, %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4203:
	.size	dot_product_fma, .-dot_product_fma
	.ident	"GCC: (Ubuntu 11.2.0-19ubuntu1) 11.2.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:

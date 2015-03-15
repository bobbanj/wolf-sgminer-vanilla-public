/*
 * Copyright 2014 sgminer developers
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.  See COPYING for more details.
 */

#include "algorithm.h"
#include "sha2.h"
#include "ocl.h"

#include "algorithm/whirlpoolx.h"

#include "compat.h"

#include <inttypes.h>
#include <string.h>

const char *algorithm_type_str[] = {
  "Unknown",
  "Scrypt",
  "NScrypt",
  "X11",
  "X13",
  "X14",
  "X15",
  "Keccak",
  "Quarkcoin",
  "Twecoin",
  "Fugue256",
  "NIST",
  "Fresh",
  "Whirlcoin",
  "Neoscrypt"
};

void gen_hash(const unsigned char *data, unsigned int len, unsigned char *hash)
{
	unsigned char hash1[32];
	sha256_ctx ctx;

	sha256_init(&ctx);
	sha256_update(&ctx, data, len);
	sha256_final(&ctx, hash1);
	sha256(hash1, 32, hash);
}

#define CL_SET_BLKARG(blkvar) status |= clSetKernelArg(*kernel, num++, sizeof(uint), (void *)&blk->blkvar)
#define CL_SET_VARG(args, var) status |= clSetKernelArg(*kernel, num++, args * sizeof(uint), (void *)var)
#define CL_SET_ARG_N(n, var) do { status |= clSetKernelArg(*kernel, n, sizeof(var), (void *)&var); } while (0)
#define CL_SET_ARG_0(var) CL_SET_ARG_N(0, var)
#define CL_SET_ARG(var) CL_SET_ARG_N(num++, var)
#define CL_NEXTKERNEL_SET_ARG_N(n, var) do { kernel++; CL_SET_ARG_N(n, var); } while (0)
#define CL_NEXTKERNEL_SET_ARG_0(var) CL_NEXTKERNEL_SET_ARG_N(0, var)
#define CL_NEXTKERNEL_SET_ARG(var) CL_NEXTKERNEL_SET_ARG_N(num++, var)

// Placeholder - I may use it later.
static void whirlpoolx_set_compile_options(build_kernel_data *build_data, struct cgpu_info *gpu, algorithm_settings_t *settings)
{
	return;
}

static cl_int queue_whirlpoolx_kernel(struct __clState *clState, struct _dev_blk_ctx *blk, __maybe_unused cl_uint threads)
{
	uint64_t midblock[8], key[8] = { 0 };
	cl_ulong le_target;
	cl_int status;
	
	le_target = *(cl_ulong *)(blk->work->device_target + 24);
	flip80(clState->cldata, blk->work->data);
	
	memcpy(midblock, clState->cldata, 64);
	
	// midblock = n, key = h
	for(int i = 0; i < 10; ++i)
	{
		whirlpool_round(key, (uint64_t [8]){ WHIRLPOOL_ROUND_CONSTANTS[i], 0, 0, 0, 0, 0, 0, 0 });		
		whirlpool_round(midblock, (uint64_t [8]){ 0 });
		
		for(int x = 0; x < 8; ++x) midblock[x] ^= key[x];
	}
	
	for(int i = 0; i < 8; ++i) midblock[i] ^= ((uint64_t *)(clState->cldata))[i];
	
	status = clSetKernelArg(clState->kernel, 0, sizeof(cl_ulong8), (cl_ulong8 *)&midblock);
	status |= clSetKernelArg(clState->kernel, 1, sizeof(cl_ulong), (void *)(((uint64_t *)clState->cldata) + 8));
	status |= clSetKernelArg(clState->kernel, 2, sizeof(cl_ulong), (void *)(((uint64_t *)clState->cldata) + 9));
	status |= clSetKernelArg(clState->kernel, 3, sizeof(cl_mem), (void *)&clState->outputBuffer);
	status |= clSetKernelArg(clState->kernel, 4, sizeof(cl_ulong), (void *)&le_target);

	return status;
}

static algorithm_settings_t algos[] = 
{  
	{ "whirlpoolx", ALGO_WHIRLPOOLX, "", 1, 1, 1, 0, 0, 0xFFU, 0xFFFFULL, 0x0000FFFFUL, 0, 0, 0, whirlpoolx_regenhash, queue_whirlpoolx_kernel, gen_hash, NULL },
	// Terminator (do not remove)
	{ NULL, ALGO_UNK, "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL}
};

// I really hope you've defined strcasecmp() if it didn't exist, because I saw it used
// in a macro, and it'd really be bad form if you used a non-standard function that
// doesn't exist on some popular compilers...

void set_algorithm(algorithm_t *algo, const char *newname_alias)
{
	if(!strcasecmp(newname_alias, "vanillacoin") || !strcasecmp(newname_alias, "whirlpoolx"))
	{
		strcpy(algo->name, algos[0].name);
		
		// If a custom kernel file name wasn't specified, use the default for WhirlpoolX.
		algo->kernelfile = (algo->kernelfile) ? algo->kernelfile : algos[0].kernelfile;
		algo->type = algos[0].type;

		algo->diff_multiplier1 = algos[0].diff_multiplier1;
		algo->diff_multiplier2 = algos[0].diff_multiplier2;
		algo->share_diff_multiplier = algos[0].share_diff_multiplier;
		algo->xintensity_shift = algos[0].xintensity_shift;
		algo->intensity_shift = algos[0].intensity_shift;
		algo->found_idx = algos[0].found_idx;
		algo->diff_numerator = algos[0].diff_numerator;
		algo->diff1targ = algos[0].diff1targ;
		algo->n_extra_kernels = algos[0].n_extra_kernels;
		algo->rw_buffer_size = algos[0].rw_buffer_size;
		algo->cq_properties = algos[0].cq_properties;
		algo->regenhash = algos[0].regenhash;
		algo->queue_kernel = algos[0].queue_kernel;
		algo->gen_hash = algos[0].gen_hash;
		algo->set_compile_options = algos[0].set_compile_options;
	}
}

void set_algorithm_nfactor(algorithm_t* algo, const uint8_t nfactor) { }
bool cmp_algorithm(const algorithm_t *algo1, const algorithm_t *algo2) { return(!safe_cmp(algo1->name, algo2->name) && !safe_cmp(algo1->kernelfile, algo2->kernelfile)); }

/*
 * Copyright 2011-2012 Con Kolivas
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <sys/types.h>

#ifdef WIN32
  #include <winsock2.h>
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netdb.h>
#endif

#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#include "findnonce.h"
#include "algorithm.h"
#include "ocl.h"
#include "ocl/build_kernel.h"
#include "ocl/binary_kernel.h"

/* FIXME: only here for global config vars, replace with configuration.h
 * or similar as soon as config is in a struct instead of littered all
 * over the global namespace.
 */
#include "miner.h"

int opt_platform_id = -1;

bool get_opencl_platform(int preferred_platform_id, cl_platform_id *platform) {
  cl_int status;
  cl_uint numPlatforms;
  cl_platform_id *platforms = NULL;
  unsigned int i;
  bool ret = false;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  /* If this fails, assume no GPUs. */
  if (status != CL_SUCCESS) {
    applog(LOG_ERR, "Error %d: clGetPlatformsIDs failed (no OpenCL SDK installed?)", status);
    goto out;
  }

  if (numPlatforms == 0) {
    applog(LOG_ERR, "clGetPlatformsIDs returned no platforms (no OpenCL SDK installed?)");
    goto out;
  }

  if (preferred_platform_id >= (int)numPlatforms) {
    applog(LOG_ERR, "Specified platform that does not exist");
    goto out;
  }

  platforms = (cl_platform_id *)malloc(numPlatforms*sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  if (status != CL_SUCCESS) {
    applog(LOG_ERR, "Error %d: Getting Platform Ids. (clGetPlatformsIDs)", status);
    goto out;
  }

  for (i = 0; i < numPlatforms; i++) {
    if (preferred_platform_id >= 0 && (int)i != preferred_platform_id)
      continue;

    *platform = platforms[i];
    ret = true;
    break;
  }
out:
  if (platforms) free(platforms);
  return ret;
}


int clDevicesNum(void) {
  cl_int status;
  char pbuff[256];
  cl_uint numDevices;
  cl_platform_id platform = NULL;
  int ret = -1;

  if (!get_opencl_platform(opt_platform_id, &platform)) {
    goto out;
  }

  status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
  if (status != CL_SUCCESS) {
    applog(LOG_ERR, "Error %d: Getting Platform Info. (clGetPlatformInfo)", status);
    goto out;
  }

  applog(LOG_INFO, "CL Platform vendor: %s", pbuff);
  status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pbuff), pbuff, NULL);
  if (status == CL_SUCCESS)
    applog(LOG_INFO, "CL Platform name: %s", pbuff);
  status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(pbuff), pbuff, NULL);
  if (status == CL_SUCCESS)
    applog(LOG_INFO, "CL Platform version: %s", pbuff);
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (status != CL_SUCCESS) {
    applog(LOG_INFO, "Error %d: Getting Device IDs (num)", status);
    goto out;
  }
  applog(LOG_INFO, "Platform devices: %d", numDevices);
  if (numDevices) {
    unsigned int j;
    cl_device_id *devices = (cl_device_id *)malloc(numDevices*sizeof(cl_device_id));

    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    for (j = 0; j < numDevices; j++) {
      clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
      applog(LOG_INFO, "\t%i\t%s", j, pbuff);
    }
    free(devices);
  }

  ret = numDevices;
out:
  return ret;
}

static cl_int create_opencl_context(cl_context *context, cl_platform_id *platform)
{
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)*platform, 0 };
	cl_int status;

	*context = clCreateContextFromType(cps, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
	return status;
}

static float get_opencl_version(cl_device_id device)
{
  /* Check for OpenCL >= 1.0 support, needed for global offset parameter usage. */
  char devoclver[1024];
  char *find;
  float version = 1.0;
  cl_int status;

  status = clGetDeviceInfo(device, CL_DEVICE_VERSION, 1024, (void *)devoclver, NULL);
  if (status != CL_SUCCESS) {
    quit(1, "Failed to clGetDeviceInfo when trying to get CL_DEVICE_VERSION");
  }
  find = strstr(devoclver, "OpenCL 1.0");
  if (!find) {
    version = 1.1;
    find = strstr(devoclver, "OpenCL 1.1");
    if (!find)
      version = 1.2;
  }
  return version;
}

// I should really pass the OpenCL version to this function, but fuck it...
static cl_int create_opencl_command_queue(cl_command_queue *command_queue, cl_context *context, cl_device_id *device, const void *cq_properties)
{
	cl_int status;
	
	#ifdef HAS_OCL2
	
	if(get_opencl_version(*device) < 2.0)
	{
		*command_queue = clCreateCommandQueue(*context, *device, *((const cl_command_queue_properties *)cq_properties), &status);
		
		// Didn't work, try again with no properties.
		if(status != CL_SUCCESS) *command_queue = clCreateCommandQueue(*context, *device, 0, &status);
	}
	else
	{
		*command_queue = clCreateCommandQueueWithProperties(*context, *device, (const cl_queue_properties *)cq_properties, &status);
		
		// Didn't work, same deal.
		if(status != CL_SUCCESS) *command_queue = clCreateCommandQueueWithProperties(*context, *device, 0, &status);
	}
	
	#else
	
	*command_queue = clCreateCommandQueue(*context, *device, *((const cl_command_queue_properties *)cq_properties), &status);
	
	// Didn't work, try again with no properties.
	if(status != CL_SUCCESS) *command_queue = clCreateCommandQueue(*context, *device, 0, &status);
	
	#endif
	
	return status;
}



_clState *initCl(unsigned int gpu, char *name, size_t nameSize, algorithm_t *algorithm)
{
	cl_int status = 0;
	size_t compute_units = 0;
	cl_platform_id platform = NULL;
	struct cgpu_info *cgpu = &gpus[gpu];
	_clState *clState = (_clState *)calloc(1, sizeof(_clState));
	cl_uint preferred_vwidth, slot = 0, cpnd = 0, numDevices = clDevicesNum();
	cl_device_id *devices = (cl_device_id *)alloca(numDevices * sizeof(cl_device_id));
	build_kernel_data *build_data = (build_kernel_data *)alloca(sizeof(struct _build_kernel_data));
	unsigned char **pbuff = (unsigned char **)alloca(sizeof(unsigned char *) * numDevices), filename[256];
	
	// pbuff and filename were originally char buffers, but I prefer to KNOW if my char buffers are unsigned or not...
	// Anyways, get the platform and sanity check some shit.

	if(!get_opencl_platform(opt_platform_id, &platform)) return NULL;
	if(numDevices <= 0) return NULL;
		
	// You don't need to use the status variable if you're just checking one call, inline the fucker!
	// It's slightly more ambiguous, but users will have no idea what the code means, anyway - would
	// probably be better for debugging to resolve it to a string.
	if(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL) != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error while attempting to get list of Device IDs for all GPUs present.");
		return NULL;
	}
	
	// IMO, this is cleaner, at the expense of a very slightly more ambiguous debugging statement.
	// NOW we can use the status variable.
	
	applog(LOG_INFO, "List of devices: ");
	
	/*
		You may think the original way of doing this was better - it certainly used less memory,
		as it didn't store all of the GPU names. However, it also assumed that the response to
		the CL_DEVICE_NAME query would always be less than 256 chars - something I am not sure
		will always be true.
	*/
	
	for(int i = 0; i < numDevices; ++i)
	{
		size_t tmpsize;
		if(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &tmpsize) != CL_SUCCESS)
		{
			applog(LOG_ERR, "Error while getting the length of the name for GPU #%d.", i);
			return NULL;
		}
		
		// Does the size include the NULL terminator? Who knows, just add one, it's faster than looking it up.
		pbuff[i] = (unsigned char *)alloca(sizeof(unsigned char) * (tmpsize + 1));
		status |= clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(unsigned char) * tmpsize, pbuff[i], NULL);
	}
	
	// I really wanna change CL_SUCCESS to zero and make it if(status), but this would be bad form.
	// Were OpenCL to change the definition of CL_SUCCESS (like that'll ever happen), it'd break the code.
	
	// Heh, actually... my little OR trick only works if CL_SUCCESS is defined to zero.
	// Note to self: Fix that shit.
	
	if(status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error while attempting to get device information.");
		return NULL;
	}
	
	// Print list of devices...
	for(int i = 0; i < numDevices; ++i) applog(LOG_INFO, "\t%i\t%s", i, pbuff[i]);
	
	applog(LOG_INFO, "Selected %d: %s", gpu, pbuff[gpu]);

	if(gpu >= numDevices)
	{
		applog(LOG_ERR, "Selected GPU doesn't seem to exist - the number of GPUs present is %d, but the index of the selected GPU is %d.", numDevices, gpu);
		return NULL;
	}
	
	// Again, far cleaner, only slightly more ambiguous errors.
	status = 0;
	
	status |= create_opencl_context(&clState->context, &platform);
	status |= create_opencl_command_queue(&clState->commandQueue, &clState->context, &devices[gpu], (const void *)&(cgpu->algorithm.cq_properties));
	
	if(status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error creating OpenCL context or command queue.");
		return NULL;
	}
	
	// If it doesn't have bitalign support, you shouldn't be mining with it, goddamnit.
	clState->hasBitAlign = true;
		
	status = 0;
	status |= clGetDeviceInfo(devices[gpu], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), (void *)&clState->max_work_size, NULL);
	status |= clGetDeviceInfo(devices[gpu], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), (void *)&compute_units, NULL);
	
	if(status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error obtaining device information for GPU #%d during query of the maximum workgroup size and/or the maximum amount of compute units. ", gpu);
		return NULL;
	}

	// AMD architechture got 64 compute shaders per compute unit.
	// Source: http://www.amd.com/us/Documents/GCN_Architecture_whitepaper.pdf
	
	// Congrats, you did do SOME reading. Now, just one little nitpick - multiplications by powers of two bother me.
	// Yes, I know, most modern compilers for CPUs are good, but I'm still changing it to a left shift by 6 bits.
	// Force of habit.
	
	clState->compute_shaders = compute_units << 6;
	
	// I really think there should be a log level between info and debug for this kind of stuff...
	
	applog(LOG_INFO, "Maximum work size for this GPU (%d) is %d.", gpu, clState->max_work_size);
	applog(LOG_INFO, "Your GPU (#%d) has %d compute units, and all AMD cards in the 7 series or newer (GCN cards) \
		have 64 shaders per compute unit - this means it has %d shaders.", gpu, compute_units, clState->compute_shaders);

	/*
		This was originally a comment describing how the binary kernel file name was constructed, which is normally
		complex and kind of ugly, but it makes sense. - it would be the same as before, but this is a WhirlpoolX-only 
		miner. As such lookup gap, thread concurrency, and Nfactor make zero sense with this algo, so it'll be a
		cleaner-looking filename, like kernel_name + g + worksize + sizeof(long) + .bin
		
		Oh, yeah - originally, that ternary operator used to check for the kernelfile option, and use the default
		kernel name were it an empty string, below - it had no spaces. Now, I won't bitch about the ugly-ass use
		of two spaces instead of tabs (preferably proper tabs, with a width equal to 4 spaces), but I WILL complain
		that syntax as fucked as this sprintf used to be is fucking hard to read.
	*/

	sprintf(filename, "%s.cl", (!empty_string(cgpu->algorithm.kernelfile) ? cgpu->algorithm.kernelfile : "whirlpoolx-wolf"));

	applog(LOG_INFO, "Using source file %s.", filename);

	// Doesn't make sense for Whirlpool, but may as well set it for completeness.
	cgpu->vwidth = clState->vwidth = 1;

	clState->goffset = true;
	
	// This originally was done with an if/else - I think a ternary operator does it better.
	// IMO, anything besides a worksize of 256 (or maybe greater, if any card supports it) is
	// suboptimal for WhirlpoolX, but hey, let them try.
	
	clState->wsize = (cgpu->work_size && cgpu->work_size <= clState->max_work_size) ? cgpu->work_size : 256;
	
	build_data->context = clState->context;
	build_data->device = devices + gpu;

	// Build information
	strcpy(build_data->source_filename, filename);
	strcpy(build_data->platform, name);
	strcpy(build_data->sgminer_path, sgminer_path);
	
	// If opt_kernel_path is NULL, instead of assigning kernel_path to opt_kernel_path (again, NULL) - you explicitly assign
	// it to NULL? Seriously, what the fuck? You should need a license to practice the C programming language, I swear...
	
	//if (opt_kernel_path && *opt_kernel_path) build_data->kernel_path = opt_kernel_path;
	//else build_data->kernel_path = NULL;
	
	build_data->kernel_path = (*opt_kernel_path) ? opt_kernel_path : NULL;

	build_data->work_size = clState->wsize;
	build_data->has_bit_align = true;				// See above.
	build_data->opencl_version = get_opencl_version(devices[gpu]);

	// BFI patching is fucking obsolete, guys. Really, let it go. It's okay to delete the code.
	// It was a cool performance-enhancing hack, but the AMD OpenCL compiler is (marginally)
	// smarter now.
	build_data->patch_bfi = false;

	// Care to explain why you do the same "check if user specified kernel filename" logic twice? Cause you could just copy
	// the filename variable, you know, the result that you got when you did this check the first time, minus the .cl bit...
	
	// Oh, and use the goddamned space bar. Again. Please.

	//strcpy(build_data->binary_filename, (!empty_string(cgpu->algorithm.kernelfile)?cgpu->algorithm.kernelfile:cgpu->algorithm.name));

	strcpy(build_data->binary_filename, filename);
	build_data->binary_filename[strlen(filename) - 3] = 0x00;		// And one NULL terminator, cutting off the .cl suffix.

	strcat(build_data->binary_filename, pbuff[gpu]);
	
	// clState->goffset is always set for WhirlpoolX, no if statement necessary.
	strcat(build_data->binary_filename, "g");
	
	set_base_compiler_options(build_data);
	if(algorithm->set_compile_options) algorithm->set_compile_options(build_data, cgpu, algorithm);

	strcat(build_data->binary_filename, ".bin");
	applog(LOG_DEBUG, "Using binary file %s", build_data->binary_filename);

	// Load program from file or build it if it doesn't exist...
	
	// I think it's more readable to remove this from the if statement, as well as doing the 
	// same for the build_opencl_kernel call, but it's really a style thing, so I don't fault
	// the original author(s) for it.
	
	clState->program = load_opencl_binary_kernel(build_data);
	
	// Couldn't find a bin, build the kernel from source.
	if(!clState->program)
	{
		applog(LOG_NOTICE, "Building binary %s", build_data->binary_filename);
		
		clState->program = build_opencl_kernel(build_data, filename);
		if(!clState->program) return NULL;
		
		save_opencl_kernel(build_data, clState->program);
	}

	// Load kernel - so much simpler without checking for BFI support (all cards that are
	// worth mining with have it), and removing the support for the very obsolete binary
	// patching for BFI_INT.
	
	applog(LOG_NOTICE, "Initialising kernel %s...", filename);

	// Get a handle to the kernel
	clState->kernel = clCreateKernel(clState->program, "WhirlpoolX", &status);
	if(status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error creating WhirlpoolX kernel with clCreateKernel.");
		return NULL;
	}
	
	applog(LOG_DEBUG, "Using output buffer sized %lu", BUFFERSIZE);
	
	clState->outputBuffer = clCreateBuffer(clState->context, CL_MEM_WRITE_ONLY, BUFFERSIZE, NULL, &status);
	
	if(status != CL_SUCCESS)
	{
		applog(LOG_ERR, "Error creating output buffer.");
		return NULL;
	}

	return clState;
}


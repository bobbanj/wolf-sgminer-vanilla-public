
ACLOCAL_AMFLAGS = -I m4

JANSSON_CPPFLAGS= -I$(top_builddir)/submodules/jansson/src -I$(top_srcdir)/submodules/jansson/src

EXTRA_DIST	= example.conf m4/gnulib-cache.m4 \
		  ADL_SDK/readme.txt api-example.php miner.php	\
		  API.class API.java api-example.c hexdump.c \
		  doc/API doc/FAQ doc/GPU doc/SCRYPT doc/windows-build.txt

SUBDIRS		= lib submodules ccan

bin_PROGRAMS     = sgminer

# Well, I need to support OpenCL 1.1, 1.2, and 2.0+, and only 2.0+ has
# clCreateCommandQueueWithProperties(), and it warns if you use the
# older, deprecated clCreateCommandQueue. There doesn't seem to be
# a way to cleanly get the OpenCL version on the host side during
# compile time, so the cleanest way I know of to remove this warning,
# as I have dealt with it by conditional calls using runtime OpenCL
# version detection, is to simply -Wno-deprecated-declarations.
# Still feels bad, though.

sgminer_CPPFLAGS = $(PTHREAD_FLAGS) -std=gnu11 -Wno-deprecated-declarations $(JANSSON_CPPFLAGS)
sgminer_LDFLAGS  = $(PTHREAD_FLAGS)
sgminer_LDADD    = $(DLOPEN_FLAGS) @LIBCURL_LIBS@ @JANSSON_LIBS@ @PTHREAD_LIBS@ \
		  @OPENCL_LIBS@ @NCURSES_LIBS@ @PDCURSES_LIBS@ @WS2_LIBS@ \
		  @MM_LIBS@ @RT_LIBS@ @MATH_LIBS@ lib/libgnu.a ccan/libccan.a

sgminer_CPPFLAGS += -I$(top_builddir)/lib -I$(top_srcdir)/lib @OPENCL_FLAGS@

if !HAVE_WINDOWS
sgminer_CPPFLAGS += @LIBCURL_CFLAGS@
endif

sgminer_CPPFLAGS += $(ADL_CPPFLAGS)

if USE_GIT_VERSION
GIT_VERSION      := $(shell sh -c 'git describe --abbrev=4 --dirty')
sgminer_CPPFLAGS += -DGIT_VERSION=\"$(GIT_VERSION)\"
endif

sgminer_SOURCES := sgminer.c
sgminer_SOURCES	+= api.c api.h
sgminer_SOURCES	+= elist.h miner.h compat.h bench_block.h
sgminer_SOURCES	+= util.c util.h uthash.h
sgminer_SOURCES	+= logging.c logging.h
sgminer_SOURCES += driver-opencl.c driver-opencl.h
sgminer_SOURCES += ocl.c ocl.h
sgminer_SOURCES += sha2.c sha2.h
sgminer_SOURCES += findnonce.c findnonce.h
sgminer_SOURCES += adl.c adl.h adl_functions.h
sgminer_SOURCES += pool.c pool.h
sgminer_SOURCES += algorithm.c algorithm.h
sgminer_SOURCES += config_parser.c config_parser.h
sgminer_SOURCES += events.c events.h
sgminer_SOURCES += ocl/patch_kernel.c ocl/patch_kernel.h
sgminer_SOURCES += ocl/build_kernel.c ocl/build_kernel.h
sgminer_SOURCES += ocl/binary_kernel.c ocl/binary_kernel.h

sgminer_SOURCES += kernel/*.cl
sgminer_SOURCES += algorithm/whirlpoolx.c algorithm/whirlpoolx.h

bin_SCRIPTS	= $(top_srcdir)/kernel/*.cl


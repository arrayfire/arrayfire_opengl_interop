## How to use this Makefile

## make:			Create a release  executable.
## make debug:		Create an executable in debug mode.
## make profile:	Create a version for profiling using nvvp

#include Makefile.local

# Paths
AF_PATH?=/opt/arrayfire-2.0
CUDA?=/usr/local/cuda
GLFW_LIB_PATH?=/usr/local/lib

#======Choose================
# Linux compile
CXX=g++ -DLINUX
#============================

LIB=lib
ifeq ($(shell uname), Linux)
  ifeq ($(shell uname -m), x86_64)
	LIB=lib64
  endif
endif
PWD?=$(shell pwd)

# Flags
DEBUG =
CUDA_DEBUG =

DEFINES +=
INCLUDES +=\
	-I$(AF_PATH)/include \
	-I$(CUDA)/include  \
	-I/usr/include  # callgrind
LIBRARIES +=\
	-L$(AF_PATH)/$(LIB) -lafcu \
	-L$(CUDA)/$(LIB) -lnvToolsExt \
	-L$(GLFW_LIB_PATH) -lglfw \
	-lGL -lGLEW -lGLU \
	-L$(CUDA)/$(LIB) -lcuda -lcudart

#=======================
NVCC=$(CUDA)/bin/nvcc
CUDA_OPTIMISE=-O3
NVCCFLAGS += -ccbin $(CXX) $(ARCH_FLAGS) $(CUDA_DEBUG) $(CUDA_OPTIMISE)\
	-gencode=arch=compute_20,code=sm_20 \
	-gencode=arch=compute_30,code=sm_30 \
	-gencode=arch=compute_35,code=sm_35 \
	--ptxas-options=-v --machine 64 \
	-Xcompiler -fPIC

# Files which require compiling
SOURCE_FILES=\
	cugl_interop.cpp

OUT=cugl_interop

all: $(OUT)

$(OUT): $(SOURCE_FILES) Makefile
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(INCLUDES) $(LIBRARIES) $(SOURCE_FILES) -o $@

debug: set_debug $(OUT)

profile: set_profile $(OUT)

set_profile:
	$(eval OPTIMISE = -pg -O3)
	$(eval CUDA_OPTIMISE = -pg -lineinfo -O3)

set_debug:
	$(eval DEBUG = -g -pg)
	$(eval CUDA_DEBUG = -G)
	$(eval OPTIMISE = -O0)
	$(eval CUDA_OPTIMISE = -pg -O0)

.PHONY: clean
clean:
	@echo "Cleaning"
	rm $(OUT)
	@echo done

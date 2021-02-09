# This is set up for the CORR containers

nvcc = /usr/local/cuda-8.0/bin/nvcc
arch = sm_61
sigproc_includes = /usr/local/sigproc/src

process: process.cu
	$(nvcc) -o $@ $^ -D CUDA -arch=$(arch) -ccbin=g++ -I$(sigproc_includes) -I/usr/local/include -L/usr/local/lib -lsigproc  -O3

.PHONY: clean all

clean:
	rm -f *.o *~ process

all: process



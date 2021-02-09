// -*- c++ -*-
#include <iostream>
#include <algorithm>
using std::cout;
using std::cerr;
using std::endl;
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <time.h>
#include <syslog.h>
#include <curand.h>
#include <curand_kernel.h>

#include <sigproc.h>
#include <header.h>
#include "spline.h"

// make sure these are right for you
#define GULP 1024	// # of time samples in gulp, also baseline length
#define ZeroChannels 64 // # of channels to zero out at band edges
#define NCHAN 1024 // # of channels in data set
#define NBLCHAN 16 // # of channels over which to baseline

#define NTHREADS_GPU 32
#define MN 64.0
#define SIG 8.0
#define RMAX 16384
#define nFiltSize 21
#define MAX_NT 86400

// global variables
int DEBUG = 0;

FILE *input, *output;


void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  fwrite(&len, sizeof(int), 1, output);
  fwrite(string, sizeof(char), len, output);
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  fwrite(&floating_point,sizeof(float),1,output);
}

void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  fwrite(&double_precision,sizeof(double),1,output);
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(int),1,output);
}

void send_char(char *name, char integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(char),1,output);
}


void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(long),1,output);
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}

// kernel to calculate mean spectrum
// launch with nchan blocks of NTHREADS_GPU threads 
__global__
void calc_spectrum(float *data, float * spectrum) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int ch = (int)(block_id);
  int tm0 = (int)(thread_id*(GULP/NTHREADS_GPU));
  
  // find sum of local times
  int idx0 = tm0*NCHAN + ch;
  for (int tm=0; tm<GULP/NTHREADS_GPU; tm++) {    
    csum[thread_id] += (data[idx0]);
    idx0 += NCHAN;
  }

  __syncthreads();
  
  // sum into shared memory
  if (thread_id<16) {
    csum[thread_id] += csum[thread_id+16];
    __syncthreads();
    csum[thread_id] += csum[thread_id+8];
      __syncthreads();
    csum[thread_id] += csum[thread_id+4];
      __syncthreads();
    csum[thread_id] += csum[thread_id+2];
      __syncthreads();
    csum[thread_id] += csum[thread_id+1];
      __syncthreads();
  }
  
  if (thread_id==0) {    
    spectrum[ch] = csum[thread_id] / (1.*GULP);
  }

}


// kernel to calculate variance spectrum
// launch with NCHAN blocks of NTHREADS_GPU threads 
__global__
void calc_varspec(float *data, float * spectrum, float * varspec) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int ch = (int)(block_id);
  //if (thread_id==0) printf("%d %f\n",ch,spectrum[ch]);
  int tm0 = (int)(thread_id*(GULP/NTHREADS_GPU));
  float val;
  
  // find sum of local times
  int idx0 = tm0*NCHAN + ch;
  for (int tm=0; tm<GULP/NTHREADS_GPU; tm++) {    
    val = (data[idx0]) - spectrum[ch];
    csum[thread_id] += val*val;
    idx0 += NCHAN;
  }
  
  __syncthreads();
  
  // sum into shared memory
  if (thread_id<16) {
    csum[thread_id] += csum[thread_id+16];
    __syncthreads();
    csum[thread_id] += csum[thread_id+8];
        __syncthreads();
    csum[thread_id] += csum[thread_id+4];
        __syncthreads();
    csum[thread_id] += csum[thread_id+2];
        __syncthreads();
    csum[thread_id] += csum[thread_id+1];
        __syncthreads();
  }

  if (thread_id==0) {    
    varspec[ch] = csum[thread_id] / (1.*GULP);
  }

}

// kernel to calculate maximum value
// launch with NCHAN blocks of NTHREADS_GPU threads 
__global__
void calc_maxspec(float *data, float * maxspec) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  __shared__ float csum[NTHREADS_GPU];
  csum[thread_id] = 0.;

  int ch = (int)(block_id);
  int tm0 = (int)(thread_id*(GULP/NTHREADS_GPU));
  float val=0.;
  
  // find max of local times
  int idx0 = tm0*NCHAN + ch;
  for (int i=idx0;i<idx0+NCHAN*(GULP/NTHREADS_GPU);i+=NCHAN) {
    if ((float)(data[i])>val) val = (float)(data[i]);
  }
  csum[thread_id] = val;
  
  __syncthreads();
  
  // sum into shared memory
  int maxn = NTHREADS_GPU/2;
  int act_maxn = maxn;
  if (thread_id<maxn) {
    while (act_maxn>0) {
      if (csum[thread_id]<csum[thread_id+act_maxn])
	csum[thread_id]=csum[thread_id+act_maxn];
      act_maxn = (int)(act_maxn/2);
    }
  }

  if (thread_id==0) {    
    maxspec[ch] = csum[thread_id];
  }

}

// kernel to scale data
// launch with GULP*NCHAN/NTHREADS_GPU blocks of NTHREADS_GPU threads
__global__
void scaley(float *data, float *spectrum, float *varspec) {

  int idx = blockIdx.x*NTHREADS_GPU + threadIdx.x;
  int ch = (int)(idx % NCHAN);

  float val = (float)(data[idx]);
  val = (val-spectrum[ch])*(SIG/sqrtf(varspec[ch])) + MN;

  if (ch<ZeroChannels || ch>NCHAN-ZeroChannels)
    data[idx] = 0.;
  else
    data[idx] = val;
  
}

// kernel to subtract baseline
// launch with GULP*NCHAN/NTHREADS_GPU blocks of NTHREADS_GPU threads
// baseline has NBLCHAN chans
__global__
void subty(float *data, float *baseline) {

  int idx = blockIdx.x*NTHREADS_GPU + threadIdx.x;
  int tm = (int)(idx / NCHAN);
  int ch = (int)(idx % NCHAN);
  int blch = (int)(ch / (NCHAN/NBLCHAN));

  if (ch<ZeroChannels || ch>NCHAN-ZeroChannels)
    data[idx] = 0.;
  else
    data[idx] /= baseline[tm*NBLCHAN + blch];
  
}


// kernel to do flagging
// launch with n_mask*NTIMES_P/NTHREADS_GPU blocks of NTHREADS_GPU threads 
__global__
void flag(float *data, int * midx, float *repval) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int midx_idx = (int)(block_id/(GULP/NTHREADS_GPU));
  
  int ch = (int)(midx[midx_idx]);
  int tm = ((int)(block_id % (GULP/NTHREADS_GPU)))*NTHREADS_GPU + thread_id;
  int idx = tm*NCHAN + ch;  

  // do replacement
  data[idx] = repval[ch*GULP+tm];
    
}

// kernel to make random numbers
// launch with GULP*NCHAN/NTHREADS_GPU blocks of NTHREADS_GPU threads 
__global__
void genrand(float *repval, unsigned int seed) {

  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  
  // for random number
  curandState_t state;
  float u1, u2, va;
  curand_init(seed, block_id*NTHREADS_GPU+thread_id, 1, &state);
  u1 = ((float)(curand(&state) % RMAX)+1.)/(1.*(RMAX+2.));
  u2 = ((float)(curand(&state) % RMAX)+1.)/(1.*(RMAX+2.));
  va = sqrtf(-2.*logf(u1))*cosf(2.*M_PI*u2);

  // do replacement
  repval[block_id*NTHREADS_GPU+thread_id] = va*SIG+MN;
  
}



// assumed spec has size NBEAMS_P*NCHAN_P
// ref is reference value
void genmask(float *spec, float thresh, float ref, int *mask) {

  for (int i=0;i<NCHAN;i++) {
    if (fabs(spec[i]-ref)>thresh) mask[i] = 1;
  }

}


void swap(float *p,float *q) {
   float t;
   
   t=*p; 
   *p=*q; 
   *q=t;
}

float medval(float *a,int n) { 
	int i,j;
	float tmp[n];
	for (i = 0;i < n;i++)
		tmp[i] = a[i];
	
	for(i = 0;i < n-1;i++) {
		for(j = 0;j < n-i-1;j++) {
			if(tmp[j] > tmp[j+1])
				swap(&tmp[j],&tmp[j+1]);
		}
	}
	return tmp[(n+1)/2-1];
}

void channflag(float* spec, float Thr, int * mask);

void channflag(float* spec, float Thr, int * mask) {
	
  int i, j;
  float* baselinecorrec;	// baseline correction
  float* CorrecSpec;			// corrected spectrum

  baselinecorrec = (float *)malloc(sizeof(float)*NCHAN);
  CorrecSpec = (float *)malloc(sizeof(float)*NCHAN);
  float medspec, madspec;
  float *normspec = (float *)malloc(sizeof(float)*NCHAN);
  
  
  // calculate median filtered spectrum and correct spectrum at the same time
  for (i = 0; i < NCHAN-nFiltSize; i++){
    baselinecorrec[i] = medval(&spec[i],nFiltSize);
    CorrecSpec[i] = spec[i] - baselinecorrec[i];
  }
	
  // calculate median value for each beam
  medspec = medval(CorrecSpec,NCHAN);
  
  // compute MAD for each beam
  for (j = ZeroChannels; j < NCHAN-ZeroChannels; j++){
    normspec[j-ZeroChannels] = abs(CorrecSpec[j]-medspec);
  }
  madspec = medval(normspec,NCHAN-2*ZeroChannels);
	
  // mask  
  for (j = ZeroChannels; j < NCHAN-ZeroChannels; j++){
    if (CorrecSpec[j] > Thr * madspec || CorrecSpec[j] < - Thr * madspec)
      mask[j] = 1;
  }

  free(baselinecorrec);
  free(CorrecSpec);
  free(normspec);
  
}

// to gather mask indices
void gather_mask(int *h_idx, int *h_mask, int *n_mask) {

  (*n_mask) = 0;
  for (int i=0;i<NCHAN;i++) {
    if (h_mask[i]==1) {      
      h_idx[(*n_mask)] = i;
      //if (DEBUG) syslog(LOG_INFO,"%d %d %d",i,h_mask[i],(*n_mask));
      (*n_mask) += 1;
    }
  }

}


void usage()
{
  fprintf (stdout,
	   "process [options]\n"
	   " -f input filename\n"
	   " -o output filename\n"
	   " -b baselined output filename\n"
	   " -d send debug messages to syslog\n"
	   " -t flagging threshold [default 5.0]\n"
	   " -h print usage\n");
}


int main(int argc, char **argv)
{

  // syslog start
  openlog ("process", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  // set cuda device
  cudaSetDevice(0);
  
  // read command line args
  
  // command line arguments
  int arg = 0;
  double thresh = 5.0;
  int basel = 0;
  char foutput[200], fboutput[200];
  
  while ((arg=getopt(argc,argv,"f:t:o:b:dh")) != -1)
    {
      switch (arg)
	{
	case 't':
	  if (optarg)
	    {
	      thresh = atof(optarg);
	      syslog(LOG_INFO,"modified THRESH to %g",thresh);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-t flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      strcpy(foutput,optarg);
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'b':
	  if (optarg)
	    {
	      strcpy(fboutput,optarg);
	      basel=1;
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-b flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'f':
	  if (optarg)
	    {
	      input = fopen(optarg,"rb");
	      if (input==NULL) {
		syslog(LOG_ERR,"error opening file %s\n",optarg);
		exit(-1);
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-f flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }

	case 'd':
	  DEBUG=1;
	  syslog (LOG_DEBUG, "Will excrete all debug messages");
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }  

  syslog(LOG_INFO,"read command-line arguments");

  // read header
  int sigproc = read_header(input);
  syslog(LOG_INFO,"Read header with NCHAN %d fch1 %f",nchans,fch1);
  if (nchans != NCHAN) {
    syslog(LOG_ERR,"wrong number of channels");
    exit(-1);
  }

  // make output file
  output=fopen(foutput,"wb");
  send_string("HEADER_START");
  send_string("source_name");
  send_string(source_name);
  send_int("machine_id",1);
  send_int("telescope_id",82);
  send_int("data_type",1); // filterbank data
  send_double("fch1",fch1); // THIS IS CHANNEL 0 :)
  send_double("foff",foff);
  send_int("nchans",nchans);
  send_int("nbits",nbits);
  send_double("tstart",tstart);
  send_double("tsamp",tsamp);
  send_int("nifs",1);
  send_string("HEADER_END");
  // TODO add RA and DEC
  
  // declare stuff for host and GPU
  float * d_data;
  cudaMalloc((void **)&d_data, GULP*NCHAN*sizeof(float));
  float * h_data = (float *)malloc(sizeof(float)*GULP*NCHAN);
  int * h_mask = (int *)malloc(sizeof(int)*NCHAN);
  int * d_mask;
  cudaMalloc((void **)&d_mask, NCHAN*sizeof(int));
  float * d_spec, * d_oldspec;
  cudaMalloc((void **)&d_spec, NCHAN*sizeof(float));
  cudaMalloc((void **)&d_oldspec, NCHAN*sizeof(float));
  float * h_spec = (float *)malloc(sizeof(float)*NCHAN);
  float * h_subspec = (float *)malloc(sizeof(float)*NCHAN);
  float * h_var = (float *)malloc(sizeof(float)*NCHAN);
  float * h_max = (float *)malloc(sizeof(float)*NCHAN);
  float * h_oldspec = (float *)malloc(sizeof(float)*NCHAN);
  float *d_spec0, *d_var0;
  cudaMalloc((void **)&d_spec0, NCHAN*sizeof(float));
  cudaMalloc((void **)&d_var0, NCHAN*sizeof(float));
  for (int i=0;i<NCHAN;i++) h_oldspec[i] = 0.;
  cudaMemcpy(d_oldspec, h_oldspec, NCHAN*sizeof(float), cudaMemcpyHostToDevice);
  float * d_var, * d_max;
  cudaMalloc((void **)&d_var, NCHAN*sizeof(float));
  cudaMalloc((void **)&d_max, NCHAN*sizeof(float));
  int * h_idx = (int *)malloc(sizeof(int)*NCHAN);
  int * d_idx;
  cudaMalloc((void **)&d_idx, NCHAN*sizeof(int));
  int n_mask = 0;
  float * ts = (float *)malloc(sizeof(float)*MAX_NT*NBLCHAN);
  float * h_baseline = (float *)malloc(sizeof(float)*GULP*NBLCHAN);
  float * d_baseline;
  cudaMalloc((void **)&d_baseline, GULP*NBLCHAN*sizeof(float));
  
  syslog(LOG_INFO,"allocated all memory");
  
  // random numbers
  float *d_repval;
  cudaMalloc((void **)&d_repval, GULP*NCHAN*sizeof(float));
  genrand<<<GULP*NCHAN/NTHREADS_GPU,NTHREADS_GPU>>>(d_repval,time(NULL));
  syslog(LOG_INFO,"done with repvals");

  // first reading loop

  int nmemb = GULP*NCHAN;
  int started=0;
  int ct = 0;
  
  while (nmemb==GULP*NCHAN) {

    nmemb = fread(h_data,sizeof(float),GULP*NCHAN,input);
    if (DEBUG) syslog(LOG_INFO,"read nmemb %d",nmemb);
    
    if (nmemb==GULP*NCHAN) {

      // copy data to device
      cudaMemcpy(d_data, h_data, GULP*NCHAN*sizeof(float), cudaMemcpyHostToDevice);
      if (DEBUG) syslog(LOG_INFO,"copied data");

      // if not first block, correct data
      if (started==1) 
	scaley<<<GULP*NCHAN/NTHREADS_GPU,NTHREADS_GPU>>>(d_data, d_spec0, d_var0);

      if (DEBUG) syslog(LOG_INFO,"scaled");
    
      // measure spectrum and varspec
      calc_spectrum<<<NCHAN, NTHREADS_GPU>>>(d_data, d_spec);
      cudaDeviceSynchronize();
      calc_varspec<<<NCHAN, NTHREADS_GPU>>>(d_data, d_spec, d_var);
      cudaDeviceSynchronize();
      cudaMemcpy(h_spec, d_spec, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_var, d_var, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);

      
      if (DEBUG) syslog(LOG_INFO,"done spec and var");
    
      // if not first block
      if (started==1) {

	// calc maxspec
	//calc_spectrum<<<NCHAN, NTHREADS_GPU>>>(d_data, d_max);

	// derive channel flags
	//cudaMemcpy(h_max, d_max, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i=0;i<NCHAN;i++) {
	  h_mask[i] = 0;
	  h_subspec[i] = h_spec[i]-h_oldspec[i];
	}
	channflag(h_subspec,thresh,h_mask);
	channflag(h_var,thresh,h_mask);
	//channflag(h_max,thresh,h_mask);      
	
	// apply mask
	gather_mask(h_idx, h_mask, &n_mask);
	if (DEBUG) syslog(LOG_INFO,"FLAG_COUNT %d",n_mask);   		
	cudaMemcpy(d_idx, h_idx, n_mask*sizeof(int), cudaMemcpyHostToDevice);
	flag<<<n_mask*GULP/NTHREADS_GPU, NTHREADS_GPU>>>(d_data, d_idx, d_repval);      
	cudaDeviceSynchronize();

	for (int i=0;i<NCHAN;i++) {
	  h_oldspec[i] = h_spec[i];	  
	}
	calc_spectrum<<<NCHAN, NTHREADS_GPU>>>(d_data, d_spec);
	cudaDeviceSynchronize();
	cudaMemcpy(h_spec, d_spec, NCHAN*sizeof(float), cudaMemcpyDeviceToHost);

	
	
      }

      // scale first block anyway
      if (started==0) 
	scaley<<<GULP*NCHAN/NTHREADS_GPU,NTHREADS_GPU>>>(d_data, d_spec, d_var);
      
      // copy data to host and write to buffer
      cudaMemcpy(h_data, d_data, GULP*NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      fwrite(h_data, sizeof(float), GULP*NCHAN, output);

      for (int i=0;i<NBLCHAN;i++) {
	ts[ct*NBLCHAN + i] = 0.;
	for (int j=i*(NCHAN/NBLCHAN);j<(i+1)*NCHAN/NBLCHAN;j++) {
	  ts[ct*NBLCHAN + i] += h_spec[j]/(1.*(NCHAN/NBLCHAN));
	}
      }
      ct++;

      //if (DEBUG && started==0) 
      //	for (int i=0;i<NCHAN;i++) printf("%f %f\n",h_spec[i],h_var[i]);
      
      // deal with started and oldspec
      if (started==0) {
	cudaMemcpy(d_spec0, d_spec, NCHAN*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_var0, d_var, NCHAN*sizeof(float), cudaMemcpyDeviceToDevice);
	started=1;
      }
      
      if (DEBUG) syslog(LOG_INFO,"done with round");
   
    }
   
  }

  fclose(input);
  fclose(output);

  if (basel==1) {

    syslog(LOG_INFO,"Removing baseline");

    // read header
    input = fopen(foutput,"rb");
    int sigproc = read_header(input);

    // make output file
    output=fopen(fboutput,"wb");
    send_string("HEADER_START");
    send_string("source_name");
    send_string(source_name);
    send_int("machine_id",1);
    send_int("telescope_id",82);
    send_int("data_type",1); // filterbank data
    send_double("fch1",fch1); // THIS IS CHANNEL 0 :)
    send_double("foff",foff);
    send_int("nchans",nchans);
    send_int("nbits",nbits);
    send_double("tstart",tstart);
    send_double("tsamp",tsamp);
    send_int("nifs",1);
    send_string("HEADER_END");
    // TODO add RA and DEC    
  
    // trial spline fit
    std::vector<double> x(ct);
    std::vector<double> y(ct);
    tk::spline s[NBLCHAN];

    for (int j=0;j<NBLCHAN;j++) {
      for (int i=0;i<ct;i++) {
	x[i] = (double)((i+0.5)*GULP*1.);
	y[i] = (double)(ts[i*NBLCHAN+j]);
      }
      y[0] = y[1]; // to look after unscaled first block
      s[j].set_points(x,y);
    }
    
    // do read of data
    nmemb = GULP*NCHAN;
    ct = 0.;
  
    while (nmemb==GULP*NCHAN) {

      nmemb = fread(h_data,sizeof(float),GULP*NCHAN,input);
      if (DEBUG) syslog(LOG_INFO,"read nmemb %d",nmemb);
      
      if (nmemb==GULP*NCHAN) {
	
	// copy data to device
	cudaMemcpy(d_data, h_data, GULP*NCHAN*sizeof(float), cudaMemcpyHostToDevice);
	if (DEBUG) syslog(LOG_INFO,"copied data");

	// make baseline
	for (int j=0;j<NBLCHAN;j++) {
	  for (int i=0;i<GULP;i++) 
	    h_baseline[i*NBLCHAN+j] = (float)(s[j]((double)(ct*GULP+i)));	  
	}
	cudaMemcpy(d_baseline, h_baseline, NBLCHAN*GULP*sizeof(float), cudaMemcpyHostToDevice);
	
	// correct data
	subty<<<GULP*NCHAN/NTHREADS_GPU,NTHREADS_GPU>>>(d_data, d_baseline);
	if (DEBUG) syslog(LOG_INFO,"removed baseline");

	cudaMemcpy(h_data, d_data, GULP*NCHAN*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	fwrite(h_data, sizeof(float), GULP*NCHAN, output);

	ct++;
	
      }

    }

    fclose(input);
    fclose(output);
    
  }
  

  free(h_data);
  free(h_mask);
  free(h_spec);
  free(h_oldspec);
  free(h_var);
  free(h_subspec);
  free(h_max);
  free(h_idx);
  free(ts);
  free(h_baseline);
  cudaFree(d_repval);
  cudaFree(d_data);
  cudaFree(d_spec);
  cudaFree(d_oldspec);
  cudaFree(d_var);
  cudaFree(d_mask);
  cudaFree(d_spec0);
  cudaFree(d_var0);
  cudaFree(d_max);
  cudaFree(d_idx);
  cudaFree(d_baseline);
  return 0;    
} 

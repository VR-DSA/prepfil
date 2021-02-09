# prepfil

Contains a single routine to pre-process X-band DSN filterbank files:
 - removes edge channels
 - scales by initial estimate of bandpass
 - flags based on variations in the spectrum, and on the variance in the spectrum
 - removes a channelized baseline
All operations are done on the timescale of a gulp. Important variables to edit are at the top of the code.

### To compile

Edit the makefile (paying particular attention to the CUDA version and the compute architecture of the GPU). Add other include paths if needed.

### To run

process [options]
 -  -f input filename [no default - essential]
 -  -o output filename [no default - essential]
 -  -b baselined output filename [optional]
 -  -d send debug messages to syslog 
 -  -t flagging threshold [default 5.0]
 -  -h print usage
 
 ### Outputs
 
 Will always output a file prior to baselining, and optionally a file after baselining. 

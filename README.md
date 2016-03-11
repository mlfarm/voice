#   Voice project

Set of functions that is to do with voice processing.

##  File structure
-   pending: raw wave file that need to be processed, deleted when processed
-   wave:    copy of pending file
-   float:   float file of wave file, normalized in (-1, 1)
-   power:   fft power spectrum of float file
-   mfcc:    mfcc using power spectrum
-   encode:  encoded data of power spectrum using fft-encoder
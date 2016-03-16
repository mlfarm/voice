#   Voice project

Set of functions that is to do with voice processing.

##  File structure
-   pending: raw wave file that need to be processed, deleted when processed
-   wave:    copy of pending file
-   float:   float file of wave file, normalized in (-1, 1)
-   power:   fft power spectrum of float file
-   mfcc:    mfcc using power spectrum
-   encode:  encoded data of power spectrum using fft-encoder

##  Process
### FFT

44100Hzでサンプリングされた2byte整数のwaveファイルを[-1,1]に正規化しfloatとして保存した後, 1024のサンプル数でパワースペクトラムを取る.

### RAE

1024次元のパワースペクトラムをRuccursive Auto-Encoderを使って256次元に落とす.
この時RAEはencode, decodeにそれぞれ1つの層を使い計2層で構成される.
x[t]からx[t+1]を再現しencode層にのみLSTMを使う.

### Discriminator

256次元の表現ベクトルから話者認識に特化した16次元のベクトルをDiscriminatorを使って得る.
層はすべてLinearで次元は256->64->32->16と落ちていく.
活性化関数にはReLuを使い最終層はidentityとする.

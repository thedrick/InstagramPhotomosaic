#ifndef __CUDA_MOSAIC_H__
#define __CUDA_MOSAIC_H__

class CudaMosaic {

private:
  int numImages;
  int numSlices;
  int cutSize;

  int* cudaDeviceImageAverages;
  int* cudaDeviceAllAverages;
  int* cudaDeviceImageIndex;

  int* imageAverages;
  int* allAverages;
  int* imageIndex;

public:
  CudaMosaic();
  ~CudaMosaic();
  // void setup(int ni, int ns, int cs);
  void setup(int ni, int ns, int cs, int* imgavg, int* allavg);

  void imageMatch();

  // void setImageAverages(int* averages);
  // void setAllAverages(int* averages);
  const int* getIndices();
};


#endif

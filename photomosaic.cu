#include <math.h>
#include <stdio.h>
#include <vector>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "photomosaic.h"

struct GlobalConstants {
  int numImages;
  int numSlices;
  int cutSize;

  int* imageAverages;
  int* allAverages;
  int* imageIndex;
};

__constant__ GlobalConstants cuConstMosaicParams;


__device__ __inline__ int square(int x) {
  return x * x;
}

__device__ __inline__ int RGBdistance(int3 rgb1, int3 rgb2) {
  int red = square(rgb1.x - rgb2.x);
  int green = square(rgb1.y - rgb2.y);
  int blue = square(rgb1.z - rgb2.z);
  return (int)sqrt((float)(red + green + blue));
}

__global__ void kernelMatchImages() {
  int numImages = cuConstMosaicParams.numImages;
  int numSlices = cuConstMosaicParams.numSlices;
  int cutSize = cuConstMosaicParams.cutSize;
  int cSizeSquared = square(cutSize);

  int width = numSlices * cutSize;
  int index = blockIdx.x * blockDim.x + (blockDim.y * blockIdx.y * width);
  index += (threadIdx.x + (width * threadIdx.y));

  int imageAverageStart = index * cSizeSquared * 3;

  int dist;
  int minIndex = 0;
  int minVal = INT_MAX;
  for (int i = 0; i < (numImages * square(cutSize)); i += (cSizeSquared * 3)) {
    dist = 0;
    for (int j = 0; j < (cSizeSquared * 3); j += 3) {
      int3 rgb1 = *(int3*)(&cuConstMosaicParams.imageAverages[imageAverageStart + j]);
      int3 rgb2 = *(int3*)(&cuConstMosaicParams.allAverages[i + j]);
      dist += RGBdistance(rgb1, rgb2);
    }
    if (dist < minVal) {
      minVal = dist;
      minIndex = (i / (cSizeSquared * 3));
    }
  }
  __syncthreads();
  cuConstMosaicParams.imageIndex[index] = minIndex;
  __syncthreads();
}

CudaMosaic::CudaMosaic() {
  printf("Constructor\n");
  cudaDeviceImageAverages = NULL;
  cudaDeviceAllAverages = NULL;
  cudaDeviceImageIndex = NULL;

  imageAverages = NULL;
  allAverages = NULL;
  imageIndex = NULL;
}

CudaMosaic::~CudaMosaic() {
  printf("Destructing!\n");
  if (imageAverages) {
    delete [] imageAverages;
    delete [] allAverages;
    delete [] imageIndex;
  }

  if (cudaDeviceImageAverages) {
    cudaFree(cudaDeviceImageAverages);
    cudaFree(cudaDeviceAllAverages);
    cudaFree(cudaDeviceImageIndex);
  }

  delete [] imageAverages;
  delete [] allAverages;
  delete [] imageIndex;
}

// void CudaMosaic::setAllAverages(int *averages) {
//   for (int i = 0; i < (numImages * cutSize * cutSize * 3); i++) {
//     allAverages[i] = averages[i]; 
//   }
// }

// void CudaMosaic::setImageAverages(int *averages) {
//   printf("Trying to set image averages\n");
//   for (int i = 0; i < (numSlices * numSlices * cutSize * cutSize * 3); i++) {
//     imageAverages[i] = averages[i];
//   }
// }

const int* CudaMosaic::getIndices() {
  printf("Copying index data from device\n");
  cudaMemcpy(imageIndex,
             cudaDeviceImageIndex,
             sizeof(int) * numSlices * numSlices,
             cudaMemcpyDeviceToHost);
  // for (int i = 0; i < numSlices * numSlices; i++) {
  //   printf("Index %d found match at %d\n", i, imageIndex[i]);
  // }
  cudaDeviceReset();
  return imageIndex;
}

void CudaMosaic::setup(int ni, int ns, int cs, int* imgavg, int* allavg) {
  printf("Calling setup with numImages %d, numSlices %d, cutSize %d\n", ni, ns, cs);
  numImages = ni;
  numSlices = ns;
  cutSize = cs;

  imageAverages = new int[numSlices * numSlices * cutSize * cutSize * 3];
  allAverages = new int[numImages * cutSize * cutSize * 3];
  imageIndex = new int[numSlices * numSlices];

  int i;
  for (i = 0; i < (numSlices * numSlices * cutSize * cutSize * 3); i++) {
    imageAverages[i] = imgavg[i];
    // if (i % 100 == 0) {
    //   printf("Setting image average %d for index %d\n", imageAverages[i], i);
    // }
  }
  for (i = 0; i < (numImages * cutSize * cutSize * 3); i++) {
    allAverages[i] = allavg[i]; 
    // if (i % 100 == 0) {
    //   printf("Setting all average %d for index %d\n", allAverages[i], i);
    // }
  }

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("Initializing CUDA for CudaMosaic\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }

  cudaMalloc(&cudaDeviceImageAverages, sizeof(int) * cutSize * cutSize * 3 * (numSlices * numSlices));
  cudaMalloc(&cudaDeviceAllAverages, sizeof(int) * cutSize * cutSize * 3 * numImages);
  cudaMalloc(&cudaDeviceImageIndex, sizeof(int) * numSlices * numSlices);

  cudaMemcpy(cudaDeviceImageAverages, imageAverages, sizeof(int) * cutSize * cutSize * 3 * (numSlices * numSlices), cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceAllAverages, allAverages, sizeof(int) * cutSize * cutSize * 3 * numImages, cudaMemcpyHostToDevice);
  // cudaMemcpy(cudaDeviceImageIndex, imageIndex, sizeof(int) * numImages, cudaMemcpyHostToDevice);

  printf("Successfully transferred to device\n");

  GlobalConstants params;
  params.numImages = numImages;
  params.numSlices = numSlices;
  params.cutSize = cutSize;
  params.imageAverages = cudaDeviceImageAverages;
  params.allAverages = cudaDeviceAllAverages;
  params.imageIndex = cudaDeviceImageIndex;

  cudaMemcpyToSymbol(cuConstMosaicParams, &params, sizeof(GlobalConstants));
}

void CudaMosaic::imageMatch() {
  dim3 threadsPerBock(cutSize, cutSize, 1);
  dim3 numBlocks(numSlices, numSlices, 1);

  printf("About to launch kernel block size %d num blocks %d\n", threadsPerBock.x, numBlocks.x);
  kernelMatchImages<<<numBlocks, threadsPerBock>>>();
  cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf( "cudaCheckError() failed at %s", cudaGetErrorString( err ) );
        exit( -1 );
    }
  cudaThreadSynchronize();
}









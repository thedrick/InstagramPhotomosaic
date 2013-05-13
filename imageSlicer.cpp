#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <math.h>
#include <climits>
#include <list>
#include "mongo/client/dbclient.h"
#include "CycleTimer.h"

#include "imageSlicer.h"

using namespace std;
using namespace Magick;
using namespace mongo;

ImageSlicer::ImageSlicer (string src, int n, int cSize) {
  imgsrc = src;
  numSlices = n;
  cutSize = cSize;
  slices = vector< vector<Image> > (n);
  for (int x = 0; x < n; x++) {
    slices[x] = vector<Image> (n);
  }
  sourceImage = Image(src);
  if (sourceImage.columns() != 612 || sourceImage.rows() != 612) {
    printf("Trying to slice an image that is not 612 x 612 at %s\n", src.c_str());
    exit(1);
  }
}

// slice the input image into a grid of numSlices x numSlices
// and puts these subimages into an array for use later.
void ImageSlicer::slice() {
  Image img = sourceImage;
  int width = img.columns();
  int height = img.rows(); // size of instagram photo

  if (numSlices == 1) {
    Image piece = img;
    slices[0][0] = img;
    return;
  }

  int subwidth = width / numSlices;
  int subheight = height / numSlices;
  int count = 0;
  for (int x = 0; x < numSlices; x++) {
    for (int y = 0; y < numSlices; y++) {
      Image piece = img;
      piece.crop(Geometry(subwidth, subheight, x * subwidth, y * subheight));
      // piece.write(sstr.str());
      slices[y][x] = piece;
      count++;
    }
  }
  return;
}

// get a list of RGB values from the input image. This takes each slice
// and cuts it into a 3x3 grid and finds the average RGB value in each
// of the grid sections. Thus each sub image is represented by 9 RGB values
void ImageSlicer::calculateRGBValues() {
  int subwidth = 612 / numSlices;
  int subheight = 612 / numSlices;

  int average_subwidth = subwidth / cutSize;
  int average_subheight = subheight / cutSize;

  for (int x = 0; x < numSlices; x++) {
    for (int y = 0; y < numSlices; y++) {
      Image currentSlice = slices[x][y];
      Pixels view(currentSlice);
      PixelPacket *pixels = view.get(0, 0, currentSlice.columns(), currentSlice.rows());
      for (int i = 0; i < cutSize; i++) {
        for (int j = 0; j < cutSize; j++) {
          int red = 0;
          int green = 0;
          int blue = 0;
          for (int k = 0; k < average_subwidth; k++) {
            for (int l = 0; l < average_subheight; l++) {
              int curX = (i * average_subwidth) + k;
              int curY = (j * average_subheight) + l;
              int pixelLoc = curX + (curY * subwidth);
              ColorRGB pixel = ColorRGB(pixels[pixelLoc]);
              red += pixel.red() * 255;
              green += pixel.green() * 255;
              blue += pixel.blue() * 255;
            }
          }
          int numIters = average_subwidth * average_subheight;
          red /= numIters;
          blue /= numIters;
          green /= numIters;
          RGB rgb;
          rgb.red = red;
          rgb.green = green;
          rgb.blue = blue;
          // printf("Red: %d, Green: %d, Blue: %d\n", red, green, blue);
          averages.push_back(rgb);
        }
      }
    }
  }
}

vector< vector< Image > > ImageSlicer::getSlices() {
  slice();
  return slices;
}

vector<RGB> ImageSlicer::getAverages() {
  slice();
  calculateRGBValues();
  return averages;
}
#include <iostream>
#include <vector>
#include <string>
#include <Magick++.h>

using namespace std;
using namespace Magick;

struct RGB {
  int red;
  int green;
  int blue;
};

// object which holds an image and slices it into pieces to
// extract average RGB values for each subpiece.
class ImageSlicer {
  string imgsrc; // source of the input image
  int numSlices; // number of slices in x and y direction
  int cutSize; // number of sub slices to make from each image 
  vector< vector< RGB > > rgbs; // vector to store RGB values of pieces
  vector<RGB> averages; // vector to store average rgbs
  vector< vector< Image > > slices; // image slices.
  Image sourceImage;

public:
  ImageSlicer(string imgsrc, int n, int cSize);
  vector<RGB> getAverages();
  vector< vector< Image > > getSlices();

private:
  void slice();
  void calculateRGBValues();
};
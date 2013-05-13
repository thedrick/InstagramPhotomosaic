/*
ImageSlicer.cpp designed to reproduce the results of
ImageSlicer.py and slice up a target image into sections
used in image matching for creating photo mosaics
*/

#include <math.h>
#include <climits>
#include <list>
#include <pthread.h>
#include <iterator>
#include "mongo/client/dbclient.h"
#include "CycleTimer.h"
#include "imageSlicer.h"

static vector<int> blahblah;
static vector< vector <int> > blah;
static vector<int> indices;
vector<int>& finalRef = indices; 
vector<int>& averageRef = blahblah;
vector< vector< int > > dbImageColorRef = blah;
pthread_mutex_t mutex;

struct ThreadArgs {
  int cutSize;
  vector<int> indices;
};

int square(int x) {
  return x * x;
}

void printRGB(RGB rgb) {
  printf("RGB is Red: %d, Green: %d, Blue: %d\n", rgb.red, rgb.green, rgb.blue);
}

int RGBdistance(RGB t1, RGB t2) {
  printRGB(t2);
  int red = square(t1.red - t2.red);
  int green = square(t1.green - t2.green);
  int blue = square(t1.blue - t2.blue);
  return (int)sqrt(red + green + blue);
}

int totalDistance(vector<RGB> a1, vector<RGB> a2) {
  if (a2.size() == 0) {
    return INT_MAX;
  }
  int dist = 0;
  for (size_t i = 0; i < a1.size(); i++) {
    dist += RGBdistance(a1[i], a2[i]);
  }
  return dist;
}

void* handleThread(void *data) {
  ThreadArgs* args = (ThreadArgs*)data;
  // current value of the minimum distance and it's index.
  int cutSize = args->cutSize;
  int minIndex;
  int minVal;
  for (size_t i = 0; i < args->indices.size(); i++) {
    int index = args->indices[i];
    if (index % 50 == 0) {
      // printf("Thread working on index %d\n", index);
    }
    minIndex = 0;
    minVal = INT_MAX;
    vector<RGB> current(cutSize * cutSize);
    // grab the next 9 values (corresponds to a subimage)
    for (int j = 0; j < (cutSize * cutSize); j+=3) {
      RGB n;
      n.red = averageRef[index + j];
      n.green = averageRef[index + j + 1];
      n.blue = averageRef[index + j + 2];
      current.push_back(n);
    }
    //printf("Made the current vector\n");
    double subimagestart = CycleTimer::currentSeconds();
    for (size_t k = 0; k < dbImageColorRef.size(); k+=3) {
      // get distance from current subimage to current image from db.
      vector<RGB> icrVec;
      for (int poop = 0; poop < (int)dbImageColorRef[k].size(); poop+=3) {
        RGB icr;
        icr.red = dbImageColorRef[k][poop];
        icr.green = dbImageColorRef[k][poop + 1];
        icr.blue = dbImageColorRef[k][poop + 2]; 
        icrVec.push_back(icr); 
      }
      int dist = totalDistance(current, icrVec);
      if (dist < minVal) {
        minIndex = k;
        minVal = dist;
      }
    }
    printf("Final min distance was %d\n", minVal);
    double subimageend = CycleTimer::currentSeconds();
    if (index % 50 == 0) {
      printf("Time to find one image match: %f\n", (subimageend - subimagestart));
    }
    pthread_mutex_lock(&mutex);
    int actual = index / 9;
    finalRef[actual] = minIndex;
    // printf("Updated index %d with min index %d\n", actual, finalRef[actual]);
    pthread_mutex_unlock(&mutex);
  }
  // printf("This thread just finished and this is the state of finals\n");
  // for (size_t i = 0; i < finalRef.size(); i++) {
  //   printf("Index: %zu value: %d\n", i, finalRef[i]);
  // }
  return NULL;
  // replace minIndex with empty vector to remove values and avoid duplicates.
}

int main(int argc, char **argv) {
  InitializeMagick(0);
  if (argc < 4) {
    cout << "usage: " << argv[0] << " <image path> <save path> <cutSize>\n";
    return 1;
  }

  int cutSize = atoi(argv[3]);
  ImageSlicer slicer(argv[1], 51, cutSize);
  string savepath = argv[2];

  mongo::DBClientConnection c;
  c.connect("localhost");

  pthread_mutex_init(&mutex, NULL);

  vector <vector <int> > dbImageColors;
  vector <string> dbImageSources;
  vector <int> imageIndices;
  auto_ptr<mongo::DBClientCursor> cursor = c.query("instagram_photomosaic.image_pool_cpp", mongo::BSONObj());
  // load all the stuff from the database to check against.
  double dbstart = CycleTimer::currentSeconds();
  while (cursor->more()) {
    mongo::BSONObj obj = cursor->next();
    dbImageSources.push_back(obj.getStringField("srcsmall"));
    mongo::BSONObjIterator fields (obj.getObjectField("averages"));
    vector <int> curRGBs;
    while (fields.more()) {
      vector<mongo::BSONElement> elems = fields.next().Array();
      int red = elems[0].Int();
      int green = elems[2].Int();
      int blue = elems[1].Int();
      curRGBs.push_back(red);
      curRGBs.push_back(green);
      curRGBs.push_back(blue);
    }
    // add vector of 9 rgbs to large vector

    dbImageColors.push_back(curRGBs);
  }
  double dbend = CycleTimer::currentSeconds();
  printf("Time to read in DB %f\n", (dbend - dbstart));

  dbImageColorRef = dbImageColors;
  // average values of the input image. This is an array of a bunch of RGBs
  // where they are grouped in 9s in order.
  double avgstart = CycleTimer::currentSeconds();
  vector<RGB> avgRGB = slicer.getAverages();
  vector<int> averages;
  for (size_t i = 0; i < avgRGB.size(); i++) {
    RGB cur = avgRGB[i];
    averages.push_back(cur.red);
    averages.push_back(cur.green);
    averages.push_back(cur.blue);
  }
  // averages = slicer.getAverages();
  averageRef = averages;
  double avgend = CycleTimer::currentSeconds();
  printf("Time to find averages of input image %f\n", (avgend - avgstart));
  
  vector<string> finalImages;
  vector<ThreadArgs> vectorargs(4);
  ThreadArgs one;
  one.cutSize = cutSize;
  ThreadArgs two;
  two.cutSize = cutSize;
  ThreadArgs three;
  three.cutSize = cutSize;
  ThreadArgs four;
  four.cutSize = cutSize;
  vectorargs[0] = one;
  vectorargs[1] = two;
  vectorargs[2] = three;
  vectorargs[3] = four;

  vector<int> finalIndices(51*51);

  double imgstart = CycleTimer::currentSeconds();
  for (size_t i = 0; i < averages.size(); i += (cutSize * cutSize)) {
    vectorargs[i % 4].indices.push_back(i);
  }
  finalRef = finalIndices;
  vector<pthread_t> threads;
  for (int j = 0; j < (int)vectorargs.size(); j++) {
    pthread_t thread;
    threads.push_back(thread);
    pthread_create(&thread, NULL, &handleThread, static_cast<void*>(&vectorargs[j]));
  }
  for (int k = 0; k < (int)threads.size(); k++) {
    pthread_join(threads[k], NULL);
  }
  for (size_t i = 0; i < finalIndices.size(); i++) {
    int minindex = finalRef[i];
    printf("minindex is %d\n", minindex);
    finalImages.push_back(dbImageSources[minindex]);
  }
  dbImageColors.resize(0);
  averages.resize(0);
  double imgend = CycleTimer::currentSeconds();
  printf("Time to compute image matches %f\n", (imgend - imgstart));
  list <Image> finalMontage;
  Montage montage;
  montage.tile("51x51");
  montage.geometry("48x48");
  vector<Image> images;
  double montagestart = CycleTimer::currentSeconds();
  for (int n = 0; n < (int)finalImages.size(); n++) {
    string filename = finalImages[n];
    if (n % 100 == 0) {
      printf("Opening image %d at path %s\n", n, filename.c_str());
    }
    Image mosaicImage(filename);
    images.push_back(mosaicImage);
  }
  montageImages(&finalMontage, images.begin(), images.end(), montage);
  writeImages(finalMontage.begin(), finalMontage.end(), savepath);
  double montageend = CycleTimer::currentSeconds();
  printf("Time to create and write montage to file %f\n", (montageend - montagestart));
}




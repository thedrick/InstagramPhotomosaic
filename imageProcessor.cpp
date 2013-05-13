#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <string>
#include <Magick++.h>
#include "mongo/client/dbclient.h"
#include "imageSlicer.h"
#include "CycleTimer.h"

using namespace std;
using namespace Magick;
using namespace mongo;

// get back a vector with all of the files names.
vector<string> listFiles() {
  DIR *pDIR;
  vector<string> imgs;
  struct dirent *entry;
  if ((pDIR = opendir("./images"))) {
    while((entry = readdir(pDIR))) {
      if (strcmp(entry->d_name, ".") != 0 && 
          strcmp(entry->d_name, "..") != 0 &&
          strcmp(entry->d_name, ".DS_Store") != 0) {

        imgs.push_back(string(entry->d_name));
      }
    }
    closedir(pDIR);
  }
  return imgs;
}

int main(int argc, char** argv) {
  vector<string> imgs = listFiles();
  DBClientConnection c;
  c.connect("localhost");
  string db = "instagram_photomosaic.image_pool_cpp";
  // printf("connected\n");
  
  // printf("created the total array thing\n");
  int totalcount = 0;
  for (int i = 0; i < (int)imgs.size(); i++) {
    
    string imagePath = imgs[i];
    string largePath = string("images/") + imagePath;
    string smallPath = string("smallImages/") + imagePath;
    auto_ptr<DBClientCursor> cursor = c.query(db, QUERY("imgsrc" << largePath));
    int count = cursor->itcount();
    if (count != 0) {
      printf("Skipping image %s\n", largePath.c_str());
      continue;
    }
    ImageSlicer slicer(largePath, 1, 3); // cut each image into a 3x3 grid and save it to the DB
    // printf("made the slicer\n");
    vector<RGB> averages = slicer.getAverages();
    // printf("got the averages\n");
    BSONObjBuilder b;
    b.append("imgsrc", largePath);
    b.append("srcsmall", smallPath);
    // printf("just appended shit\n");
    BSONArrayBuilder a;
    for (int j = 0; j < (int)averages.size(); j++) {
      RGB rgb = averages[j];
      BSONArrayBuilder r;
      BSONArray ra = r.append(rgb.red).append(rgb.blue).append(rgb.green).arr();
      // printf("Just made an array\n");
      a.append(ra);
    }
    b.append("averages", a.arr());
    c.insert(db, b.obj());
    totalcount++;
    if (totalcount % 50 == 0) {
      printf("Inserted the %dth image\n", totalcount);
    }
  }
  return 0;
}

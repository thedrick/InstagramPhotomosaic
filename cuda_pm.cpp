#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <math.h>
#include <climits>
#include <list>
#include <jpeglib.h>
#include <GLUT/glut.h>

#include "photomosaic.h"
#include "mongo/client/dbclient.h"
#include "CycleTimer.h"
#include "imageSlicer.h"

using namespace mongo;
void add_images_to_raw(unsigned char* img, int index);

/* pointer to new image */
unsigned char *raw_image = NULL;

/* mosaic tile image dimensions */
int iwidth = 48;
int iheight = 48;

/* mosaic dimensions and component info */
int width = 2448;
int height = 2448;
int bytes_per_pixel = 3;
J_COLOR_SPACE color_space = JCS_RGB;

/*** change in c++ to iwidth/width ***/
int dim = width / iwidth;

int read_jpeg_to_array(char *filename, int idx) {
  /* standard libjpeg structures for reading */
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  /* structure to store scanline of image */
  JSAMPROW row_pointer[1];
  /* array to hold current image */
  unsigned char *image_lines;
  FILE *infile = fopen(filename, "rb");
  unsigned long loc = 0;
  int i = 0;

  if (!infile) {
    printf("Error opening file %s.\n", filename);
    return -1;
  }
  /* set up all decompress and reading image */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  image_lines = (unsigned char*)malloc(cinfo.output_width*cinfo.output_height*cinfo.num_components);
  row_pointer[0] = (unsigned char*)malloc(cinfo.output_width*cinfo.num_components);

  /* writes image scanline info to image_lines */
  while(cinfo.output_scanline < cinfo.image_height) {
    jpeg_read_scanlines(&cinfo, row_pointer, 1);
    for(i=0; i<(int)cinfo.image_width*cinfo.num_components; i++) {
      image_lines[loc++] = row_pointer[0][i];
    }
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  free(row_pointer[0]);
  fclose(infile);

  /* write image to respective idx in mosaic array */
  add_images_to_raw(image_lines, idx);
  free(image_lines);
  return 1;
}

void add_images_to_raw(unsigned char* img, int index) {
  int j;
  //int raw_loc = index*iwidth*iheight*bytes_per_pixel;
  int start_loc = (index*iwidth*bytes_per_pixel) + ((width*(index/dim))*iheight*bytes_per_pixel);
  int offset = (dim-1)*bytes_per_pixel;
  int count = 0;
  //printf("Image index %i\n", index);
  for (j=0; j < iheight*iwidth*bytes_per_pixel; j++) {
    if (count == iwidth*bytes_per_pixel) {
      start_loc += iwidth*offset;
      count = 0;
    }
    raw_image[start_loc++] = img[j];
    count++;;
  }
}

int write_jpeg_to_file(char *filename) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer[1];
  FILE *outfile = fopen(filename, "wb");

  if (!outfile) {
    printf("Error opening output file %s.\n", filename);
    return -1;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = bytes_per_pixel;
  cinfo.in_color_space = color_space;

  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);

  while(cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &raw_image[cinfo.next_scanline*cinfo.image_width*cinfo.input_components];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  fclose(outfile);
  return 1;
}


int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Please indicate the input image and save path\n");
    return 1;
  }

  // mosaic = new CudaMosaic(108560, 51, 3);

  raw_image = (unsigned char*)malloc(width*height*bytes_per_pixel*sizeof(char*));

  ImageSlicer slicer(argv[1], 51, 3);
  string savepath = argv[2];

  DBClientConnection c;
  c.connect("localhost");

  vector <int> dbImageColors;
  vector <string> dbImageSources;
  auto_ptr<DBClientCursor> cursor = c.query("instagram_photomosaic.image_pool_cpp", BSONObj());
  
  // load all the stuff from the database to check against.
  double dbstart = CycleTimer::currentSeconds();
  while (cursor->more()) {
    BSONObj obj = cursor->next();
    dbImageSources.push_back(obj.getStringField("srcsmall"));
    BSONObjIterator fields (obj.getObjectField("averages"));

    while (fields.more()) {
      vector<BSONElement> elems = fields.next().Array();
      int red = elems[0].Int();
      int green = elems[2].Int();
      int blue = elems[1].Int();
      dbImageColors.push_back(red);
      dbImageColors.push_back(green);
      dbImageColors.push_back(blue);
    }
  }
  double dbend = CycleTimer::currentSeconds();
  printf("Time to read in DB %f of totalsize %zu\n", (dbend - dbstart), dbImageColors.size());

  vector<RGB> imgavgs = slicer.getAverages();
  vector<int> cudaImgAverages;
  for (size_t i = 0; i < imgavgs.size(); i++) {
    RGB cur = imgavgs[i];
    cudaImgAverages.push_back(cur.red);
    cudaImgAverages.push_back(cur.green);
    cudaImgAverages.push_back(cur.blue);
  }
  double cudaStart = CycleTimer::currentSeconds();
  // mosaic->setup(108560, 51, 3);

  CudaMosaic* mosaic = new CudaMosaic();
  mosaic->setup(108560, 51, 3, cudaImgAverages.data(), dbImageColors.data());

  // mosaic->setImageAverages(cudaImgAverages.data());
  // mosaic->setAllAverages(dbImageColors.data());
  // mosaic->setImageAverages();
  // mosaic->setAllAverages();

  mosaic->imageMatch();
  double cudaEnd = CycleTimer::currentSeconds();
  printf("Time taken for CUDA is %f\n", (cudaEnd - cudaStart));
  const int *result = mosaic->getIndices();
  // int *result = NULL;

  vector<string> finalImages(51 * 51);
  for (int i = 0; i < (51 * 51); i++) {
    int idx = result[i];
    // printf("Final index was %d for %d\n", idx, i);
    if (idx > (int)dbImageSources.size()) {
      idx = 0;
    }
    finalImages[i] = dbImageSources[idx];
  }

  list <Image> finalMontage;
  vector<Image> images;
  double montagestart = CycleTimer::currentSeconds();
  for (int n = 0; n < (int)finalImages.size(); n++) {
    string filename = finalImages[n];
    if (n % 100 == 0) {
      printf("Opening image %d at path %s\n", n, filename.c_str());
    }
    // Image mosaicImage(filename);
    // images.push_back(mosaicImage);
    char* writable = new char[filename.size() + 1];
    copy(filename.begin(), filename.end(), writable);
    writable[filename.size()] = '\0';
    read_jpeg_to_array(writable, n);
    delete [] writable;
  }
  // montageImages(&finalMontage, images.begin(), images.end(), montage);
  // writeImages(finalMontage.begin(), finalMontage.end(), savepath);
  char *writable = new char[savepath.size() + 1];
  copy(savepath.begin(), savepath.end(), writable);
  writable[savepath.size()] = '\0';
  write_jpeg_to_file(writable);
  delete [] writable;
  // delete mosaic;
  free(raw_image);
  double montageend = CycleTimer::currentSeconds();
  printf("Time to create and write montage to file %f\n", (montageend - montagestart));
  return 0;
}







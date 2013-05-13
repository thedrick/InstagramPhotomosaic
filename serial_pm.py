import os
import sys 
import math
import time
import Image
from pymongo import MongoClient
from argparse import ArgumentParser
from imageSlicer import ImageSlicer
from operator import itemgetter


client = MongoClient()

def distance(t1, t2):
  t1 = t1[0]
  rs = (t1[0] - t2[0]) ** 2
  gs = (t1[1] - t2[1]) ** 2
  bs = (t1[2] - t2[2]) ** 2
  return math.sqrt(rs + gs + bs)

def totalDistance(arr1, arr2):
  if (arr2 == []):
    return 255**2
  mapped = map(lambda x: distance(arr1[x], arr2[x]), range(len(arr1)))
  return sum(mapped)

if __name__ == '__main__':
  desc = """Serial version of a photo mosaic creation program
  using Instagram photos as inputs and outputs."""
  parser = ArgumentParser(description=desc)
  parser.add_argument("image_path", metavar="PATH", type=str, nargs=1,
            help="path to the input image")
  parser.add_argument("-o", dest='output_path', type=str, nargs=1,
                      help="path to output image")
  parser.add_argument("--reuse", "-r", dest="reuse", action="store_true")
  args = parser.parse_args()
  db = client.instagram_photomosaic
  image_pool = db.image_pool_cpp
  pool = image_pool.find()
  pool_items = [x for x in pool]
  all_images = [x["averages"] for x in pool_items]
  image_paths = [x["imgsrc"] for x in pool_items]

  reuse = args.reuse
  # BEGIN TIMER
  start = time.clock()
  slicer = ImageSlicer(args.image_path[0], 51)
  averages = slicer.get_averages()
  subimages = [averages[i:i+9] for i in range(0, len(averages), 9)]
  final_images = []
  for img in subimages:
    start_one = time.clock()
    distances = map(lambda x: totalDistance(img, x), all_images)
    minIndex = min(enumerate(distances), key=itemgetter(1))[0]
    end_one = time.clock()
    print "Time taken to find one image is %f" % (end_one - start_one)
    final_images.append(image_paths[minIndex])
    if not reuse:
      all_images[minIndex] = []
  
  imageGrid = [final_images[i:i+51] for i in range(0, len(final_images), 51)]
  output = Image.new('RGB', (2448, 2448))
  for y in range(len(imageGrid)):
    for x in range(len(imageGrid)):
      im = Image.open(imageGrid[y][x]).resize((48, 48))
      output.paste(im, (x * 48, y * 48))
  end = time.clock()
  print "It took %f seconds to compute" % (end - start)
  if (args.output_path):
    print "Saving result to %s" % args.output_path[0]
    out = args.output_path[0]
    output.save(out)
  else:
    print "Saving result to testbig.jpg"
    output.save("testbig.jpg")









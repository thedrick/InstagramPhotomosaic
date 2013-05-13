import os
import sys 
import math
import time
import Image
from pymongo import MongoClient
from argparse import ArgumentParser
from imageSlicer import ImageSlicer
from operator import itemgetter
from multiprocessing import Pool, Queue


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


def process_handler(img):
  distances = map(lambda x: totalDistance(img, x), all_images)
  minIndex = min(enumerate(distances), key=itemgetter(1))[0]
  all_images[minIndex] = []
  return image_paths[minIndex]

if __name__ == '__main__':
  desc = """Serial version of a photo mosaic creation program
  using Instagram photos as inputs and outputs."""
  parser = ArgumentParser(description=desc)
  parser.add_argument("image_path", metavar="PATH", type=str, nargs=1,
            help="path to the input image")
  parser.add_argument("-o", dest='output_path', type=str, nargs=1,
                      help="path to output image")
  parser.add_argument("--reuse", "-r", dest="reuse", action="store_true")
  parser.add_argument("--num-threads", "-t", dest="num_threads", type=int, nargs=1,
                      help="specify the number of threads for the worker pool")
  args = parser.parse_args()
  db = client.instagram_photomosaic
  image_pool = db.image_pool
  pool = image_pool.find()
  pool_items = [x for x in pool]
  global all_images 
  all_images = [x["averages"] for x in pool_items]
  image_paths = [x["imgsrc"] for x in pool_items]
  if (args.num_threads):
    p = Pool(processes=args.num_threads[0])
  else:
    p = Pool(processes=4)

  reuse = args.reuse
  # BEGIN TIMER
  start = time.clock()
  slicer = ImageSlicer(args.image_path[0], 51)
  averages = slicer.get_averages()
  subimages = [averages[i:i+9] for i in range(0, len(averages), 9)]
  final_images = p.map(process_handler, subimages)
  
  imageGrid = [final_images[i:i+51] for i in range(0, len(final_images), 51)]
  output = Image.new('RGB', (2448, 2448))
  for y in range(len(imageGrid)):
    for x in range(len(imageGrid)):
      im = Image.open(imageGrid[y][x])
      resized = im.resize((48, 48))
      output.paste(resized, (x * 48, y * 48))
  end = time.clock()
  print "It took %f seconds to compute" % (end - start)
  if (args.output_path):
    print "Saving result to %s" % args.output_path[0]
    out = args.output_path[0]
    output.save(out)
  else:
    print "Saving result to testbig.jpg"
    output.save("testbig.jpg")



import os
import sys
import math
import time
import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pymongo import MongoClient
from argparse import ArgumentParser
from imageSlicer import ImageSlicer

client = MongoClient()

mod = SourceModule("""
  __global__ void find_matches(float *src, float *output)
  {
    int idx = threadIdx.x + threadIdx.y;
    output[idx] = idx;
  } 
  """)

def make_mosaic():
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
  image_pool = db.image_pool
  pool = image_pool.find()
  pool_items = [x for x in pool]
  all_images = [x["averages"] for x in pool_items]
  image_paths = [x["imgsrc"] for x in pool_items]

  slicer = ImageSlicer(args.image_path[0], 51)
  averages = slicer.get_averages()
  flat_averages = [item for sublist in averages for item in sublist]
  flat_averages = [item for sublist in flat_averages for item in sublist]

  flat_all = [item for sublist in all_images for item in sublist]
  flat_all = [item for sublist in flat_all for item in sublist]

  cuda_src = np.array(flat_averages, dtype=np.float32)
  cuda_db = np.array(flat_all, dtype=np.float32)
  cuda_out = np.array(np.empty_like(cuda_src), dtype=np.int32)

  a_gpu = cuda.mem_alloc(cuda_src.nbytes)
  cuda.memcpy_htod(a_gpu, cuda_src)
  #cuda.memcpy_htod(a_gpu, cuda_db)
  #cuda.memcpy_htod(a_gpu, cuda_out)

  # a_gpu = cuda.mem_alloc(cuda_src.nbytes + cuda_db.nbytes)
  # cuda.memcopy_htod(a_gpu, cuda_averages)
  # cuda.memcopy_htod(a_gpu, cuda_db)
  grid = (51,51)

  func = mod.get_function("find_matches")
  func(a_gpu, block=(1,1,1), grid=(51,51))
  result = numpy.empty_like(cuda_src)
  cuda.memcpy_dtoh(result, a_gpu)
  print result
  # func.prepare("PPP")
  # block = (128, 1, 1)
  # func.prepared_call(grid, block, cuda_src, cuda_db, cuda_out)
  #print cuda_out.get()

make_mosaic()

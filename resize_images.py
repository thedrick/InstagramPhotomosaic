# resisze images
import os
import sys
import glob
from PIL import Image
from argparse import ArgumentParser
from pymongo import MongoClient


def resize_images():
  count = 0
  imgs = glob.glob("images/*.jpg")
  smalls = glob.glob("smallImages/*.jpg")
  for img in imgs:
    smallimg = img.replace("images/", "smallImages/")
    if (smallimg in smalls):
      print "Skipping image at %s" % img
      continue
    large = Image.open(img)
    small = large.resize((48, 48))
    small.save(smallimg)
    count += 1
    if (count % 50 == 0):
      print "Converted and saved the %dth image" % count

resize_images()

def add_smallpath_to_db():
  client = MongoClient();
  db = client.instagram_photomosaic
  image_pool = db.image_pool
  pool = image_pool.find()
  count = 0
  for img in pool:
    #img["smallsrc"] = img["imgsrc"].replace("images", "smallImages").replace(".jpg", "small.jpg")
    print img
    count += 1
  print count

# add_smallpath_to_db()

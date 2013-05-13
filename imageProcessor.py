import os
import sys 
import Image
from imageSlicer import ImageSlicer
from pymongo import MongoClient
import glob

def processImages():
  client = MongoClient()
  db = client.instagram_photomosaic
  image_pool = db.image_pool_new
  imgs = glob.glob("images/*.jpg")
  for img in imgs:
    if (len(list(image_pool.find({"imgsrc" : img}))) != 0):
      continue
    print "Adding img %s to db" % img
    slicer = ImageSlicer(img, 1)
    averages = slicer.get_averages()
    mapper = lambda x: (x[0][0], x[0][1], x[0][2])
    averages = map(mapper, averages)
    imgsmall = img.replace("images", "smallImages").replace(".jpg", "small.jpg")
    dbitem = {
      "imgsrc" : img,
      "srcsmall" : imgsmall,
      "averages" : averages
    }
    image_pool.insert(dbitem)

processImages()

def processImages12():
  client = MongoClient()
  db = client.instagram_photomosaic
  image_pool = db.image_pool_12
  imgs = glob.glob("images/*.jpg")
  count = 0
  for img in imgs:
    if (len(list(image_pool.find({"imgsrc" : img}))) != 0):
      continue
    print "Trying to add img %s " % img
    slicer = ImageSlicer(img, 4)
    averages = slicer.get_averages()
    mapper = lambda x: (x[0][0], x[0][1], x[0][2])
    averages = map(mapper, averages)
    imgsmall = img.replace("images", "smallImages").replace(".jpg", "small.jpg")
    dbitem = {
      "imgsrc" : img,
      "srcsmall" : imgsmall,
      "averages" : averages
    }
    image_pool.insert(dbitem)
    if (count % 100 == 0):
      print "Inserted %d photo" % count
    count += 1

#processImages12()

def processImagesFull48():
  client = MongoClient()
  db = client.instagram_photomosaic
  image_pool = db.image_pool_48
  imgs = glob.glob("smallImages/*.jpg")
  count = 0
  for img in imgs:
    if (len(list(image_pool.find({"srcsmall" : img}))) != 0):
      continue
    loaded = Image.open(img)
    averages = list(loaded.getdata())
    largesrc = img.replace("smallImages", "images").replace("small.jpg", ".jpg")
    dbitem = {
      "imgsrc" : largesrc,
      "srcsmall" : img,
      "averages" : averages
    }
    image_pool.insert(dbitem)
    if (count % 100 == 0):
      print "Inserted %d photo" % count
    count += 1

#processImagesFull48()

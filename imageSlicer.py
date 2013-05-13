import os
import sys
from PIL import Image
from pymongo import MongoClient

# raised when num_slices is not a divisor of instagram_wh
class InstagramSizeException(Exception):
  pass

class ImageSlicer:
  """
  Image slicer

  Attributes:
    imgsrc     -- string specifiying the location of the image to be sliced
    num_slices -- number of slices to cut the image into. Ex: 12 = (12 x 12)
    saveloc    -- location to save the resulting slices 
  """

  # default width and height of instagram photos.
  instagram_wh = 612

  def __init__(self, imgsrc, num_slices=12):
    self.imgsrc = imgsrc
    self.num_slices = num_slices
    self.slices = []
    self.rgbs = []
    self.averages = []
    if ImageSlicer.instagram_wh % num_slices != 0:
      raise InstagramSizeException("num_slices is not a divisor of 612")
    self.client = MongoClient()

  def get_averages(self):
    if (self.averages == []):
      self.average_RGB()
    return self.averages

  # slice image into pieces of equal size defined by num_slices
  def slice(self):
    if (self.slices != []):
      return self.slices
    img = Image.open(self.imgsrc)
    width, height = img.size
    slices = [[None for w in range(self.num_slices)] for h in range(self.num_slices)]
    if (width != ImageSlicer.instagram_wh or 
       height != ImageSlicer.instagram_wh):
      raise InstagramSizeException("Photo is not 612 x 612")

    subwidth = ImageSlicer.instagram_wh / self.num_slices
    subheight = ImageSlicer.instagram_wh / self.num_slices
    for x in xrange(self.num_slices):
      for y in xrange(self.num_slices):
        box = [
          x * subwidth,
          y * subheight,
          (x + 1) * subwidth,
          (y + 1) * subheight
        ]
        piece = img.crop(box)
        slices[y][x] = (piece)
    self.slices = slices
    return slices

  def get_RGB_values(self):
    if self.slices is None or self.slices == []:
      self.slice()
    self.rgbs = map(lambda y: map(lambda x: list(x.getdata()), y), self.slices)
    return self.rgbs
    
  def average_RGB(self):

    cut_size = 3
    # def reduce_average((r1,g1,b1), (r2,g2,b2)):
    #   return ((r1 + r2) / 2, (g1 + g2) / 2, (b1 + b2) / 2)

    def reduce_average(l):
      rf = 0
      gf = 0
      bf = 0
      for (r,g,b) in l:
        rf += r
        gf += g
        bf += b
      num = len(l)
      return (rf / num, gf / num, bf / num)

    subpieces = []
    if (self.rgbs == []):
      self.get_RGB_values()
    subwidth = ImageSlicer.instagram_wh / self.num_slices
    subheight = ImageSlicer.instagram_wh / self.num_slices

    average_subwidth = subwidth / cut_size
    average_subheight = subheight / cut_size

    averages = []
    for x in xrange(self.num_slices):
      for y in xrange(self.num_slices):
        current_slice = self.slices[x][y]
        for i in xrange(cut_size):
          for j in xrange(cut_size):
            sub_averages = []
            cropped = current_slice.crop((average_subwidth * j,
                                          average_subheight * i,
                                          average_subwidth * j + average_subwidth,
                                          average_subheight * i + average_subheight))
            reduced = reduce_average(list(cropped.getdata()))
            # print reduced
            sub_averages.append(reduced)
            averages.append(sub_averages)
    self.averages = averages
    
    flattened = []
    block_size = self.num_slices*cut_size**2
    h = 0
    while (h != self.num_slices):
      for k in range(block_size*h, block_size*(h+1)):
        if (k % (cut_size**2) == 0 or k % (cut_size**2) == 1 or k % (cut_size**2) == 2):
          flattened.append(averages[k])
      
      for m in range(block_size*h, block_size*(h+1)):
        if (m % (cut_size**2) == 3 or m % (cut_size**2) == 4 or m % (cut_size**2) == 5) :
          flattened.append(averages[m])
      
      for n in range(block_size*h, block_size*(h+1)):
        if (n % (cut_size**2) == 6 or n % (cut_size**2) == 7 or n % (cut_size**2) == 8) :
          flattened.append(averages[n])          
      h+=1
    #print flattened
    return [item for sublist in flattened for item in sublist]


# slicer = ImageSlicer("images/318960010740589992_182714056.jpg", 1)

# # # slicer.average_RGB()
# # # exit(0)

# img.putdata(averages)
# img.save("./test3.jpg")
# # Image manip
# img = Image.open(path)
# # get a list of all rgb values in the image
# list(img.getdata())














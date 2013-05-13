import os, sys
import Image
import urllib
from StringIO import StringIO
from instagram.client import InstagramAPI
from time import sleep

words = ['me', 'cute', 'tbt', 'follow', 'eyes', 'happy', 'girl', 'statigram', 'l4l', 'instacollage', 'instadaily', 'throwbackthursday', 'christmas', 'easter', 'halloween', 'all_shots', 'selfie', 'like', 'nice', 'instago', 'smile', 'niallhoran', 'fashion', 'boyfriend', 'bestoftheday', 'shoutout', 'throwback', 'snow', 'iphonesia', 'home', 'shoes', 'loveit', 'webstagram', 'pretty', 'a', 'instagramhub', 'tweegram', 'in', 'my', 'swag', 'hair', 'nike', 'bored', 'old', 'life', 'model', 'picstitch', 'tree', 'tattoo', 'fitness', 'heart', 'cool', 'jj', 'like4like', 'amazing', 'doubletap', 'face', 'goodtimes', 'igdaily', 'repost', 'sea', 'friday', 'loveyou', 'drunk', 'cake']

api = InstagramAPI(client_id="4aa0a8ef77b34a0d8e3ddf2d97523f22", client_secret="da88550123c74ddb9de9d7a3e0bb088d")
total = 0
maxid = 0
try:
  for tag in words:
    maxid = 0
    for x in xrange(1):
      sleep(5)
      if (maxid == 0):
        popular_media, pagin = api.tag_recent_media(count=100, tag_name=tag)
        print pagin
      else:
        print "trying page %d for tag %s" % (x,tag)
        api.tag_recent_media(count=200, max_id=maxid, tag_name=tag)
      for media in popular_media:
        curimg = media.images['standard_resolution']
        # dont save images that aren't 612 x 612
        if (curimg.width != 612 or curimg.height != 612):
          continue
        url = curimg.url
        maxid = media.id
        try:
          Image.open(StringIO(urllib.urlopen(url).read())).save("images/" + str(media.id) + ".jpg")
          # Image.open(StringIO(urllib.urlopen(url).read())).resize((48,48)).save("smallImages/" + str(media.id) + "small.jpg")
          total += 1
        except Exception as e:
          "Handling exception while saving photos ",
          print e
          continue
except Exception as e:
  print "Handling error and stopping",
  print e

print "number of images %d" % total
  
  
  
  

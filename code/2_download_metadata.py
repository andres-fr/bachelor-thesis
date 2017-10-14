"""This script manages the downloading of the labels from the Dunya platform
   (http://dunya.compmusic.upf.edu). Note that it requires an authorized key to
   work.
"""
from __future__ import print_function
import pickle, urllib2, os.path, time
from optparse import OptionParser
from compmusic import dunya

token = "?"

dunya.set_token(token)
c = dunya.carnatic
with open("recordings_FINAL.pickle", "rb") as f:
      rec = pickle.load(f)


rec_complete = []
for x in rec:
     try:
          rec_complete.append(c.get_recording(x["mbid"]))
          print(len(rec_complete))
     except:
          pass


# # from the list returned by get_recordings(), the download_mp3(recID, path) function returns a 404 error for the following 17 "mbid"s.
# # get_recordings() returns 2472 elements that look like this:
# # {u'mbid': u'ffe2f301-1793-4e07-8256-72c8000a0eca', u'title': u'Agaramumaagi'}
# # from them, the download_mp3(recID, path) returns 2455, and a 404 error for the following 17;
# # [u'0b7fe96c-da9f-45fb-b03e-227c0fbc0768',
# #  u'23549a5a-b16b-4418-8844-3c64bb0dcbcb',
# #  u'29026436-b829-40e2-8ff6-381655d04e53',
# #  u'34997f07-2ddc-4694-a2af-3251cd6765d7',
# #  u'3c5ff8a7-3fd5-4096-a654-dccdd87473ee',
# #  u'4cf1afc0-0ac9-4a3d-8fe3-720a4d8a27fd',
# #  u'523ca9ae-d18b-4f81-8148-743a33e0c671',
# #  u'53863e6e-d7c3-46dd-ba8e-0208032fee93',
# #  u'6e779c65-eb8d-4eba-90c6-13e66b403183',
# #  u'745af700-9a66-430d-903b-136f85a3c452',
# #  u'7bf13f4b-66de-402a-aa53-0178dcbbb3d0',
# #  u'94ba70a3-7f41-4a9a-bd0d-874375e667ed',
# #  u'9d5cd306-ba80-476d-b9b3-ce1d451748ab',
# #  u'd726139e-5177-44c1-a95c-909cb86a50d2',
# #  u'da92128d-6ab6-4534-8368-1d273eeacc44',
# #  u'e90004c8-36aa-42b4-9b52-cdfe2d2249e5',
# #  u'f5b5788e-918c-46f1-9ad3-cee66e367223']
# # therefore this list isn't used, the "recording_LABELS.pickle" are enough
# with open("recordings_FINAL.pickle", "rb") as f:
#      yyyy = pickle.load(f)


# this file contains the 2455 labels that look like this:
# {u'raaga': [], u'form': [{u'name': u'Thiruppugazh'}], u'title': u'Agaramumaagi', u'work': [], u'album_artists': [{u'mbid': u'd13ab6b6-ee9c-4ad5-9c9a-65b4be9fa062', u'name': u'Vijay Siva'}], u'taala': [], u'length': 437332, u'artists': [], u'mbid': u'ffe2f301-1793-4e07-8256-72c8000a0eca', u'concert': [{u'mbid': u'df8f05b4-ffcd-4b72-bc58-eb0497350f72', u'title': u'Pazhani Vijayam'}]}
with open("recording_LABELS.pickle", "rb") as f:
     labels = pickle.load(f)

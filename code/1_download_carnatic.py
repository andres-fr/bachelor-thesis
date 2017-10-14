"""This script manages the downloading of the mp3 data from the Dunya platform
   (http://dunya.compmusic.upf.edu). Note that it requires an authorized key to
   work.

   LINUX: To convert all mp3 in folder from mp3 to wav, mono, sample rate 22050:
   for i in *.mp3; do ffmpeg -i "$i" -acodec pcm_u8 -ac 1 -ar 22050 "${i%.mp3}.wav"; done
   convert one single file:
   ffmpeg -i <FILENAME>.mp3 -acodec pcm_u8 -ac 1 -ar 22050 <FILENAME>.wav
"""


from __future__ import print_function
import pickle, urllib2, os.path, time
from optparse import OptionParser
from compmusic import dunya


def download_and_pickle_recording_list(filename, token):
     """downloads the list of all carnatic music samples in dunya, and
        pickles them to PICKLE_FILE as a python list. Needs to import
        compmusic, so it has to be installed (see )
     """
     dunya.set_token(token)
     recordings = dunya.carnatic.get_recordings()
     with open(filename,'w') as f:
          pickle.dump(recordings, f)


# Test function
def print_n_random_urls(n=10):
     """return the URLs of n random recordings (must be already unpickled)
     """
     from random import randrange
     examples = [recordings[randrange(0,len(recordings))] for _ in range(n)]
     for x in examples: print("http://dunya.compmusic.upf.edu/document/by-id/"+
                              x["mbid"]+ "/mp3")


def save_file_from_url(url, destiny, num, retry=True, max_time=30):
     """saves to the given destiny the mp3 file expected at the given url.
        WARNING: if the retry is set to true (default), this function will loop
        until success or max_time (in seconds) is reached.
     """
     timeout = time.time() + max_time
     doWhile = True # this variable ensures that while is done at least once
     while doWhile and time.time() <= timeout:
          doWhile = retry # after the first time, the 'retry' flag jumps in
          print("-->downloading no.",num,"\n\tfrom "+url+"\n\tto "+destiny)
          try:
               response = urllib2.urlopen(url)
               with open(destiny, "wb") as f:
                    html = response.read()
                    htmlSource = response.read()
                    if response.getcode() == 200:
                         f.write(html)
                         print("download succesful")
                         break
                    else:
                         print("retrying in 0.5s...")
                         time.sleep(0.5)
          except Exception as e:
               print(e)
               print("retrying in 0.5s...")
               time.sleep(0.5)

def download_carnatic_samples(n, subdir, begin_at):
     """the main function of the script. Downloads the mp3 files whose IDs are
        stored in the pickled list, starting from begin_at until begin_at+n,
        and saves them to a directory whose absolute address is given by subdir.
     """
     beg = begin_at%len(recordings)
     end = beg+n%len(recordings)
     end = end if end>=beg else len(recordings)-1
     try:
          os.mkdir(subdir)
     except Exception:
          pass
     print("DOWNLOADING ",beg,"-",end," SAMPLES TO ",subdir,\
          " (may take some time):", sep="")
     for i, x in enumerate(recordings[beg:end]):
          x_id = x["mbid"]
          x_url = "http://dunya.compmusic.upf.edu/document/by-id/"+x_id+"/mp3"
          x_address = os.path.join(subdir, x_id+".mp3")
          save_file_from_url(x_url, x_address, i+beg, True, 30)
     print("finished downloading")



def main(b, N, subdir):
     """The main routine of this script. Parses the command line arguments,
        loads the pickle file and downloads the samples.
     """
     ### SET NUMBER OF RECORDINGS TO DOWNLOAD
     if not N:
          print("Warning: only one sample will be downloaded. Set the flag "+
               "-N=n to download n samples, N=-1 downloads them all. Example:",
               "\n\tpython download_carnatic.py -N=20\nuse -h for more info")
          N = 1
     download_carnatic_samples(N, subdir,b)



def list_missing_files(subdir, pprint=True, try_recover=False):
     """once finished the download, some files may be missing. This function
        compares the filenames that were downloaded in subdir with the UUIDs at
        the 'recordings' list, and returns the UUIDs of the list that weren't
        found in subdir.
     """
     ids = {x["mbid"] : False for x in recordings}
     for f in os.listdir(subdir):
          ids[os.path.splitext(f)[0]] = True
     missing = [x for x in ids if not ids[x]]
     if pprint:
          print(missing)
          print("%d (unique) IDs listed\n%d files downloaded\
                \nmissing %d files" %(len(ids), len(ids)-len(missing),
                                       len(missing)))
     if try_recover:
          print("TRYING TO RECOVER MISSING FILES (may take some time):")
          for i, dup in enumerate(missing):
               url = "http://dunya.compmusic.upf.edu/document/by-id/"+dup+"/mp3"
               address = os.path.join(subdir, dup+".mp3")
               save_file_from_url(url, address, i, False, 30)
          print("finished recovey")
     return missing





def check_duplicates(pprint=True):
     """the list of recordigs may have some duplicate entries. This function
        returns them
     """
     d = {x["mbid"] : [] for x in recordings}
     for x in recordings:
          d[x["mbid"]].append(x["title"])
     dups = {k:d[k] for k in d.keys() if len(d[k])>1}
     if pprint: print(dups)
     return dups




if __name__== "__main__":
     ### PARSE COMMAND LINE ARGUMENTS (use the flag -h for more info)
     parser = OptionParser()
     parser.add_option("-f", "--pickle_file", dest="PICKLE_FILE",
                  help="the address to look for the pickled recording list",
                  metavar="FILE", default="recording_list.pickle")
     parser.add_option("-N", "--download_number", dest="N",
                  help="set the number of samples to be downloaded."+\
                  " N=-1 downloads them all, it may take very long",
                       metavar="NUMBER", default=None, type="int")
     parser.add_option("-d", "--subdirectory", dest="SUBDIR_NAME",
                       help="set the name of the folder where to download the"+\
                       " mp3 samples, in the same directory as this script."+\
                       " Default is 'recordings_mp3'.",
                       metavar="DIR", default="recordings_mp3")
     parser.add_option("-b", "--begin_at", dest="BEGIN_AT",
                       help="index where the script starts the download",
                       metavar="BEG", default="0", type="int")
     parser.add_option("-m", "--print_missing", dest="PRINT_MISSING",
                       help="if this flag is activated, the script ends "+\
                       "printing the missing IDs",
                       metavar="BOOL", default="False")
     (options, args) = parser.parse_args()

     ### GLOBALS
     SUBDIR = os.path.join(os.getcwd(), options.SUBDIR_NAME)

     ### TO BE DONE ONLY ONCE, IF NO PICKLE FILE IS PRESENT
     # download_and_pickle_recording_list(PICKLE_FILE)

     ### LOAD THE LIST OF RECORDINGS FROM THE PICKLE FILE
     with open(options.PICKLE_FILE,'rb') as f:
          recordings = pickle.load(f)

     ### test function
     # print_n_random_urls(10)

     ## print duplicate entries in the recordings list
     check_duplicates(True)


     ### main routine (uncomment to download)
     # main(options.BEGIN_AT, options.N, SUBDIR)

     ### uncomment to check if any files are missing
     if not os.path.exists(SUBDIR):
          os.makedirs(SUBDIR)
     if options.PRINT_MISSING:
          list_missing_files(SUBDIR, pprint=True, try_recover=True)
          list_missing_files(SUBDIR, pprint=True, try_recover=False)

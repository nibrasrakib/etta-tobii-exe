import gzip
import os
import shutil

# change to your chosen directory that hosts all the zipped files
basedir = '/Users/xinzhaoli/Documents/Research/pubmed_data'
unzipdir = os.path.join(basedir, 'unzipped')
os.mkdir(unzipdir)

for filename in os.listdir(basedir): 
    if filename.endswith(".gz"):
        print(filename)
        outname = os.path.splitext(os.path.basename(filename))[0]
        print(outname)
        with gzip.open(os.path.join(basedir, filename),'rb') as f_in:
            with open(os.path.join(unzipdir, outname),'wb') as f_out:
                shutil.copyfileobj(f_in,f_out)


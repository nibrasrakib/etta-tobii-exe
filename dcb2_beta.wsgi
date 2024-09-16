import sys
#import logging
#logging.basicConfig(stream=sys.stderr)
if sys.version_info[0]<3:       # require python3
 raise Exception("Python3 required! Current (wrong) version: '%s'" % sys.version_info)

sys.path.append('/var/www/PATTIE_Digital_Square')

from plos_exp import app as application
#application.secret_key = 'Add your secret key'

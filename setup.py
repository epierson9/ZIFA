from distutils.core import setup

DESCRIPTION = "Implements Zero-Inflated Factor Analysis"
LONG_DESCRIPTION = DESCRIPTION
NAME = "ZIFA"
AUTHOR = "Emma Pierson"
AUTHOR_EMAIL = "emmap1@cs.stanford.edu"
MAINTAINER = "Emma Pierson"
MAINTAINER_EMAIL = "emmap1@cs.stanford.edu"
DOWNLOAD_URL = 'https://github.com/epierson9/ZIFA'
LICENSE = 'MIT'

VERSION = '0.1'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['ZIFA'],
      package_data={}
     )

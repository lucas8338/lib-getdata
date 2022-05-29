from setuptools import setup,find_packages

setup(name='lib_getdata',
      version='3.0.2',
      description='general purpose library to manipulate data',
      author='Lucas Monteiro',
      author_email='lucas.ma8338@gmail.com',
      url='https://github.com/lucas8338/lib-getdata',
      packages=find_packages(),
      requires=['pandas','numpy','MetaTrader5','sklearn'],
     )
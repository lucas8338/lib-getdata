from setuptools import setup, find_packages

setup(name='lib_getdata',
      version='5.0.1',
      description='general purpose library to manipulate data',
      author='Lucas Monteiro',
      author_email='lucas.ma8338@gmail.com',
      url='https://github.com/lucas8338/lib-getdata',
      packages=find_packages(),
      install_requires=['pandas', 'numpy', 'MetaTrader5', 'sklearn', 'scipy', 'tqdm', 'statsmodels', 'pyod', 'requests_futures', 'holidays',
                'pytrends'],
      )

setup()

from setuptools import setup, find_packages

setup(name='bgmodelbuilder',
      version='0.2',
      description='Tools for building radioactive background models',
      url='http://github.com/bloer/bgmodelbuilder',
      author='Ben Loer',
      author_email='ben.loer@pnnl.gov',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          'pint==0.9',
          'uncertainties',
      ],
)

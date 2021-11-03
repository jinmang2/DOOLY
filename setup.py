from setuptools import setup, find_packages


VERSION = {}  # type: ignore
with open("dooly/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)
    

setup(name='dooly', 
      version=VERSION["version"],
      url='https://github.com/jinmang2/dooly', 
      author='jinmang2', 
      author_email='jinmang2@gmail.com', 
      description='', 
      packages=find_packages(exclude=['tests']), 
      long_description=open('README.md').read(), 
      long_description_content_type='text/markdown', 
      install_requires=['torch'],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
      ],
)
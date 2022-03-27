from setuptools import setup, find_packages


VERSION = {}  # type: ignore
with open("dooly/__version__.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setup(name='dooly',
      version=VERSION["version"],
      url='https://github.com/jinmang2/DOOLY',
      author='jinmang2',
      author_email='jinmang2@gmail.com',
      description='A library that handles everything with 🤗 and supports batching to models in PORORO',
      packages=find_packages(exclude=['tests', 'convert_scripts']),
      long_description=open('README.md', encoding="utf-8").read(),
      long_description_content_type='text/markdown',
      install_requires=['torch'],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
      ],
)

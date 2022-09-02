from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='NeuroPredPLM',
      version='0.1.0',
      description='an interpretable and robust model for neuropeptide prediction by protein language model',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='neuropeptide prediction',
      classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
      url='http://github.com/isyslab-hust/NeuroPred-PLM',
      author='Lei Wang',
      author_email='wanglei94@hust.edu.cn',
      license='MIT',
      packages=['NeuroPredPLM'],
      install_requires=[
        'einops',
        'fair-esm',
        'torch',
        'numpy'
    ],
      include_package_data=True,
      zip_safe=False)
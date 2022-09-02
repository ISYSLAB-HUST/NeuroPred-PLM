from setuptools import setup, find_packages


setup(name='NeuroPredPLM',
      version='0.1',
      description='an interpretable and robust model for neuropeptide prediction by protein language model',
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
      install_requires=find_packages("NeuroPredPLM"),
      include_package_data=True,
      zip_safe=False)
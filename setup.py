#! /usr/bin/env python
 # -*- coding: utf-8 -*_
 # Author: rentao<rentao@amss.ac.cn>
from distutils.core import setup
import setuptools
setup(
    name='pencil', # 包的名字
    version='1.0.0',  # 版本号
    description='PENCIL is a novel tool for single cell data analysis to identify phenotype enriched subpopulations and key dominated genes simutaneously.',  # 描述
    author='Tao Ren', # 作者
    author_email='rentao@amss.ac.cn', 
    url='https://github.com/Cliffthinker/PENCIL',
    packages=setuptools.find_packages(exclude=['*bak', 'example', 'pics', 'libs']),  # 包内不需要引⽤的⽂件夹
    # 依赖包
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'seaborn',
        'mlflow',
    ],
    zip_safe=False,
    license='GPL-3.0'
)

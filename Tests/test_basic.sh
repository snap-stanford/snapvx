#!/bin/bash

for dir in tests_installation tests_functionality;
do
    cd $dir
    python2.7 -m unittest discover -v
    cd ..
done

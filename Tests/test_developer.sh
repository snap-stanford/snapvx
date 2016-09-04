#!/bin/bash

for dir in tests_installation tests_functionality tests_scalability unit_tests;
do
    cd $dir
    python2.7 -m unittest discover -v
    cd ..
done

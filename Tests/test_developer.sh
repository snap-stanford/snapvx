#!/bin/bash

for dir in tests_installation tests_functionality tests_scalability unit_tests;
do
    cd $dir
    for i in `ls`;do
        python2.7 $i;
    done
    cd ..
done

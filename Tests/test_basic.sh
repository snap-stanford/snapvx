#!/bin/bash

for dir in tests_installation tests_functionality unit_tests;
do
    cd $dir
    for i in `ls`;do
        python $i
    done
    cd ..
done

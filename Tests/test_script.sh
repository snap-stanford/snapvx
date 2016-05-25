#!/bin/bash

for dir in tests_installation tests_functionality tests_scalability;
do
    cd $dir
    for entry in `ls`;
    do
        #echo "$dir""/""$entry";
        if [ -f "$entry" ];
        then
            echo `python2.7 "$entry"`;
        fi
    done
    cd ..
done

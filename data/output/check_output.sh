#!/bin/bash

# compare two files
compare_files() {
    diff $1 $2 > /dev/null
    status=$?
    if [ $status -ne 0 ]; then
        echo "difference in $1 and $2"
        exit 1 # exit if any difference is found
    fi
}

# check outputs to ensure they match the targets
compare_files chunk_output.txt chunk_target.txt
compare_files train_output.txt train_target.txt
compare_files encode_output.txt encode_target.txt

# no differences found
echo "SUCCESS"
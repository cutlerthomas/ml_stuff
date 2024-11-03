#!/bin/bash

# Define an array of arguments to pass to your program
alpha=(
    0.05
    0.075
    0.1
    0.125
    0.15
)

gamma=(
    0
    0.05
    0.1
    0.15
    0.2
    0.3
)

epsilon=(
    0.1
    0.3
    0.5
    0.7
    0.9
)

for a in "${alpha[@]}"; do
    for g in "${gamma[@]}"; do
        for e in "${epsilon[@]}"; do

            echo "Running program with args: $a, $g, and $e" >> data.txt
            ./td-q-learning.py -a "$a" -g "$g" -c "$e" -n 20000 >> data.txt
            ./td-q-learning.py score -a "$a" -g "$g" -c "$e" -n 5000 >> data.txt
            echo "                                           " >> data.txt
        done
    done
done 

#!/bin/bash

# Define an array of arguments to pass to your program
alpha=(
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
)

gamma=(
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
)

epsilon=(
    0.1
    0.2
    0.3
)

for a in "${alpha[@]}"; do
    for g in "${gamma[@]}"; do
        for e in "${epsilon[@]}"; do

            echo "Running program with args: $a, $g, and $e"
            ./td-q-learning-ai.py -a "$a" -g "$g" -c "$e" -n 5000
            ./td-q-learning-ai.py score -a "$a" -g "$g" -c "$e" -n 100
            echo "                                           "
        done
    done
done

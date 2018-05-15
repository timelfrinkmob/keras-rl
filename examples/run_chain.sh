#!/bin/bash

for i in 10 20 30 40 50 60 70 80 90 100;
do
    python examples/chain_env.py --n "$i" --exp bs --episodes 2000
done

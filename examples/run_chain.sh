#!/bin/bash

for i in `seq 5 100`;
do
    python examples/chain_env.py --n "$i" --exp bs --episodes 2000
done

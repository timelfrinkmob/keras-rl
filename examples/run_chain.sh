#!/bin/bash
for seed in 1 2 3
do
    for i in 10 20 30 40 50 60 70 80 90 100;
    do
	    for exp in bs;
	    do
            python examples/chain_env.py --n "$i" --exp "$exp" --episodes 2000 --seed "$seed"
        done
    done
done
aws s3 cp ./models/ s3://thesis-tim/models/ --recursive
aws s3 cp ./logs/ s3://thesis-tim/logs/ --recursive
sudo shutdown -P now

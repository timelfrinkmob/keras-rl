#!/bin/bash
for seed in 1 2 3
do
    for envname in BreakoutDeterministic-v4 PongDeterministic-v4;
    do
	    for exp in bs eps noisy leps;
	    do
            python examples/gym_atari.py --envname "$envname" --exp "$exp" --seed "$seed"
        done
    done
done
aws s3 cp ./models/ s3://thesis-tim/models/gym/ --recursive
aws s3 cp ./logs/ s3://thesis-tim/logs/gym/ --recursive
sudo shutdown -P now

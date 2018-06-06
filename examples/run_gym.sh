#!/bin/bash
for seed in 1 2 3
do
    for envname in MountainCar-v0 CartPole-v0;
    do
	    for exp in bs eps noisy leps;
	    do
            python examples/gym_env.py --envname "$envname" --exp "$exp" --seed "$seed" --nbsteps "200000" --mem "50000"
        done
    done
done
aws s3 cp ./models/ s3://thesis-tim/models/gym/ --recursive
aws s3 cp ./logs/ s3://thesis-tim/logs/gym/ --recursive
sudo shutdown -P now

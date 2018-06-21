for seed in 1 2 3
do
    for envname in BoxingDeterministic-v4;
    do
	    for exp in bs;
	    do
            python examples/test_atari.py --envname "$envname" --exp "$exp" --seed "$seed" --bsheads 10
        done
    done
done
aws s3 cp ./models/ s3://thesis-tim-files/models/atari/ --recursive
aws s3 cp ./logs/ s3://thesis-tim-files/logs/atari/ --recursive
sudo shutdown -P now

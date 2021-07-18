"""Main operations

Created by Kunhong Yu
Date: 2021/07/16
"""

from train import train
from test import test
import fire


def main(**kwargs):
    if 'only_test' in kwargs and kwargs['only_test']:
        test(**kwargs)
    else:
        train(**kwargs)
        test(**kwargs)


if __name__ == '__main__':
    fire.Fire()
    print('\nDone!\n')

    """
    Usage:
    1. Run 
        python main.py main \
            --data_root='./data/' \
            --dataset='mnist' \
            --distill=True \
            --baseline=False \
            --temperature=1. \
            --lambd=0.3 \
            --alpha=0.01 \
            --distill_type='kd' \
            --batch_size=32 \
            --epochs=160 \
            --optimizer='adam' \
            --init_lr=1e-5 \
            --gamma=0.2 \
            --milestones=[30,60,90,120,150] \
            --weight_decay=1e-5 \
            --fgsm=False \
            --epsilon=0.01 \
            --repeat_error_ver=False \
            --only_test=False

    2. Simply run
        python main.py main --dataset='cifar10' --distill=False --baseline=False --temperature=1.5 --distill_type='kd' --batch_size=32 --epochs=20 --fgsm=True --epsilon=1e-3 --repeat_error_ver=False --only_test=False
    
    3. Run scripts
        ./run.sh
    """
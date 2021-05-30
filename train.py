import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
def main(args):
    wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    args.model = 'lstm'
    args.asset_dir = 'asset_test/'

    #0529 feature들을 merge하여 embedding할지 정합니다.
    args.merge_feature = True

    #0529 train_data와 validation_data를 나눠 선언합니다.
    preprocess_train = Preprocess(args)
    preprocess_valid = Preprocess(args)
    preprocess_train.load_train_data('train_data.csv')
    preprocess_valid.load_test_data('validation_data.csv')


    train_data = preprocess_train.get_train_data()
    valid_data = preprocess_valid.get_test_data()

    wandb.init(project='dkt_test', config=vars(args))

    trainer.run(args, train_data, valid_data)



if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
import argparse
import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter
from data import *
from model import *
from torch import nn
from data import MyDataset
import warnings

warnings.filterwarnings('ignore')



def parse_arguments():
    parser = argparse.ArgumentParser(description="skyer pretrain")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--ss', type=int)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


class Trainer:

    def __init__(self):

        deepspeed.init_distributed()
        self.args = parse_arguments()

        _rank = deepspeed.comm.get_rank()
        if _rank == 0:
            self.log = SummaryWriter("runs")

        self.model = Skyer()

        self.engine, self.opt, self.training_dataloader, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            training_data=MyDataset(f"{self.args.data_file}", 512),
            model_parameters=self.model.parameters(),
            config="./deepspeed_config.json"
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):

        _rank = deepspeed.comm.get_rank()

        self.engine.train()

        _, client_sd = self.engine.load_checkpoint(f"save")
        if client_sd is None:
            client_sd = {"step": 0}

        for _i, _ds in enumerate(self.training_dataloader):
            _ds = _ds.to(device=self.engine.device, dtype=torch.long)
            _xs = _ds[:, :-1]
            _ys = _ds[:, 1:]
            _os = self.engine(_xs)

            _os = _os.reshape(-1, 30000)
            _os = _os - _os.mean(-1, keepdim=True)

            _ys = _ys.reshape(-1)

            _loss = self.loss_fn(_os, _ys)

            self.engine.backward(_loss)
            self.engine.step()

            _step = client_sd['step']
            if _rank == 0 and _i % 100 == 0:
                self.log.add_scalar(f"loss", _loss, _step)
            client_sd['step'] += 1

        # hour = datetime.now().hour
        ss = self.args.ss
        self.engine.save_checkpoint(f"save", tag=f"llm_{ss}",
                                    client_state={"step": client_sd['step']})


if __name__ == '__main__':
    train = Trainer()
    train()

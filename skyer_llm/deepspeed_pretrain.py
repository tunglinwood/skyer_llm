import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter
from skyer_llm.data import SkyDataset
from skyer_llm.model import Skyer
from torch import nn
import warnings

warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self, 
                 data_file, 
                 ss, 
                 seq_len,
                 num_layers,
                 input_dim,
                 hide_dim,
                 n_q_heads,
                 n_kv_heads,
                 max_len,
                 num_vocs,):
        
        self.args = {
            'local_rank': -1, 
            'data_file': data_file,
            'ss': ss,
            'seq_len': seq_len,
            'num_layers': num_layers,
            'input_dim': input_dim,
            'hide_dim': hide_dim,
            'n_q_heads': n_q_heads,
            'n_kv_heads': n_kv_heads,
            'max_len': max_len,
            'num_vocs': num_vocs
        }

        deepspeed.init_distributed()
        _rank = deepspeed.comm.get_rank()
        if _rank == 0:
            self.log = SummaryWriter("runs")

        self.model = Skyer(self.args['num_layers'],
                           self.args['input_dim'],
                           self.args['hide_dim'],
                           self.args['n_q_heads'],
                           self.args['n_kv_heads'],
                           self.args['max_len'],
                           self.args['num_vocs'])

        self.engine, self.opt, self.training_dataloader, self.lr_scheduler = deepspeed.initialize(
            args=self.args,
            model=self.model,
            training_data=SkyDataset(self.args['data_file'], self.args['seq_len']),
            model_parameters=self.model.parameters(),
            config="./deepspeed_config.json"
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self):
        _rank = deepspeed.comm.get_rank()
        self.engine.train()

        _, client_sd = self.engine.load_checkpoint("save")
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
                self.log.add_scalar("loss", _loss, _step)
            client_sd['step'] += 1

        ss = self.args['ss']
        self.engine.save_checkpoint("save", tag=f"llm_{ss}", client_state={"step": client_sd['step']})


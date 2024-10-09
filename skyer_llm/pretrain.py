<<<<<<< HEAD
import torch
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from torch.nn import CrossEntropyLoss
from model import Skyer
from skyer_llm.data import SkyDataset
import argparse
import os
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        data_path,
        num_layers,
        input_dim,
        hide_dim,
        n_q_heads,
        n_kv_heads,
        max_len,
        num_vocs,
        learning_rate
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = SummaryWriter("runs")

        self.dataset = SkyDataset(args.data_path, args.seq_len)
        self.dataloader = DataLoader(data_path, batch_size=args.batch_size)
        self.model = Skyer(
            num_layers,
            input_dim,
            hide_dim,
            n_q_heads,
            n_kv_heads,
            max_len,
            num_vocs,
        )
        self.model = self.model.to(self.device)
        self.loss = CrossEntropyLoss(ignore_index=0)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=)

    def train(args):


        

        # Loss function and optimizer
        criterion = CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Training loop
    model.train()

    # for epoch in range(start_epoch, args.epochs):
    total_loss = 0

    for _i, _ds in enumerate(dataloader):
        _ds = _ds.to(device, dtype=torch.long)
        _xs = _ds[:, :-1]
        _ys = _ds[:, 1:]
        _os = model.forward(_xs)
        _os = _os.reshape(-1, 30000)
        _os = _os - _os.mean(-1, keepdim=True)
        _ys = _ys.reshape(-1)

        loss = criterion(_os, _ys)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1  # Increment step count

        # Log progress
        if global_step % args.log_interval == 0:
            print(f"Step: {global_step}, Loss: {loss.detach().item():.4f}")
            log.add_scalar(f"loss", loss.detach().item(), global_step)
            try:
                for _name, _layer in model.named_modules():
                    if _name.startswith("_tf_layer._layers.0") and isinstance(
                        _layer, torch.nn.Linear
                    ):
                        log.add_histogram(
                            f"weight_{_name}", _layer.weight.data, global_step
                        )
                        log.add_histogram(
                            f"grad_{_name}", _layer.weight.grad, global_step
                        )
            except Exception as e:
                print(e, "Error with weight extraction.")

        # Save the model checkpoint after each epoch
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{args.ss}.pt"))
    print(f"Model saved to {args.save_dir}. Avg loss: {total_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the Skyer Model")

    # Add arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the tokenized dataset file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save_ncps",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--seq_len", type=int, default=256, help="Sequence length for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=96, help="Batch size for training"
    )
    # parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log interval for displaying training info",
    )

    # Model specific arguments
    parser.add_argument(
        "--num_layers", type=int, default=10, help="Number of transformer layers"
    )
    parser.add_argument(
        "--input_dim", type=int, default=64, help="Input dimension size"
    )
    parser.add_argument(
        "--hide_dim", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--n_q_heads", type=int, default=8, help="Number of query heads"
    )
    parser.add_argument(
        "--n_kv_heads", type=int, default=2, help="Number of key-value heads"
    )
    parser.add_argument("--num_vocs", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--ss", type=int, default=1, help="Number of model saved.")

    args = parser.parse_args()
    train(args)
=======
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
>>>>>>> 34dbad5c20e3af3ed25fa18d14e1e61f8dd02707

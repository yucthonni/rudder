import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dim', type=int, default=3)
    parser.add_argument('--action_dim', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mini_batch_size', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr_lstm', type=float, default=1e-3)
    parser.add_argument('--lstm_hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='runs')
    return parser.parse_args()
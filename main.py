<<<<<<< HEAD
import argparse
from train import train

if __name__ == "__main__":
    # Paratemter selection
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--name", type=str, default='ORL')
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--neighbor", type=list, default=[5, 5, 5])
    parser.add_argument("--hidden_size", type=list,
                        default=[2048, 2048, 2048]) 
    parser.add_argument("--emblem_size", type=list,
                        default=[512, 512, 512]) 
    parser.add_argument("--update_interval", type=int, default=6)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=5e-5)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--beta", type=float, default=100)
    args = parser.parse_args()

    train(args)
=======
import argparse
from train import train

if __name__ == "__main__":
    # Paratemter selection
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--name", type=str, default='ORL')
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--neighbor", type=list, default=[5, 5, 5])
    parser.add_argument("--hidden_size", type=list,
                        default=[2048, 2048, 2048]) 
    parser.add_argument("--emblem_size", type=list,
                        default=[512, 512, 512]) 
    parser.add_argument("--update_interval", type=int, default=6)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=int, default=5e-5)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--beta", type=float, default=100)
    args = parser.parse_args()

    train(args)
>>>>>>> 3695aa0d5f1ff49ddeae83ed76f104d5baaf29b7

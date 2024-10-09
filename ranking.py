import argparse
import os

from runners.ReRankRunner import ReRanker
from runners.RetrievalRunner import Retriever


def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default='1,0')
    parser.add_argument("--device", default='cuda:0')

    parser.add_argument(
        "--llm_name",
        default="Meta-Llama-3-8B-Instruct",
        choices=['Meta-Llama-3-8B-Instruct', 'Qwen2-7B-Instruct'])

    parser.add_argument("--data_addr", default='data/')
    parser.add_argument("--output_addr", default='')

    parser.add_argument("--data_split",
                        default='dev',
                        choices=['train', 'dev'])
    parser.add_argument("--source", default='recency')
    parser.add_argument("--task", default="LaMP_2_time")

    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    return parser


def print_args(args):
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description='rank_stage')
    init_parser.add_argument("--rank_stage",
                             default='rerank',
                             choices=['retrieval', 'rerank'])
    init_args, init_extras = init_parser.parse_known_args()
    print_args(init_args)

    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    if init_args.rank_stage == 'retrieval':
        parser = Retriever.parse_args(parser)
    elif init_args.rank_stage == 'rerank':
        parser = ReRanker.parse_args(parser)

    opts, extras = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES

    opts.data_addr = os.path.join(opts.data_addr, opts.task)

    opts.output_addr = f"{opts.llm_name}_outputs/"
    opts.output_addr = os.path.join(opts.output_addr, opts.task)
    print_args(opts)

    if init_args.rank_stage == 'retrieval':
        retriever = Retriever(opts)
        retriever.run()
    elif init_args.rank_stage == 'rerank':
        reranker = ReRanker(opts)
        reranker.run()

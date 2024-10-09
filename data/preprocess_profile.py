import argparse
import json
import os

import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--data_phase", default='train')
parser.add_argument("--task", default='LaMP_2_time')
parser.add_argument("--ranker", default='recency')

parser.add_argument("--recency_topk", type=int, default=0)
parser.add_argument("--topk", type=int, default=30)


def print_args(args):
    for flag, value in args.__dict__.items():
        print('{}: {}'.format(flag, value))


if __name__ == "__main__":
    opts = parser.parse_args()
    print_args(opts)

    task = opts.task
    ranker = opts.ranker

    result_file_name = 'rank_merge.json'

    if opts.recency_topk:
        output_ranking_addr = os.path.join(
            'data', opts.task,
            f"{opts.data_phase}/{opts.ranker}_{opts.topk}/{result_file_name}")
    else:
        output_ranking_addr = os.path.join(
            'data', opts.task,
            f"{opts.data_phase}/{opts.ranker}/{result_file_name}")

    with open(
            os.path.join(
                'data', opts.task,
                '{}/{}_questions.json'.format(opts.data_phase,
                                              opts.data_phase)), 'r') as file:
        dataset = json.load(file)

    with open(
            os.path.join(
                'data', opts.task,
                '{}/{}_outputs.json'.format(opts.data_phase, opts.data_phase)),
            'r') as file:
        out_file = json.load(file)
        out_dataset = out_file['golds']
    assert opts.task.startswith(out_file['task'])

    iter_data = enumerate(dataset)
    rank_dict = list()

    for idx, data in tqdm.tqdm(iter_data):
        assert data['id'] == out_dataset[idx]['id']
        inp = data['input']
        profile = data['profile']

        new_profile = []
        for i in range(len(profile)):
            cur_profile = profile[i]
            cur_profile['user_id'] = data['user_id']
            new_profile.append(cur_profile)
        profile = new_profile

        profile = sorted(
            profile, key=lambda x: tuple(map(int,
                                             str(x['date']).split("-"))))
        ranked_profile = profile[::-1]
        if opts.recency_topk:
            ranked_profile = ranked_profile[:opts.topk]

        rank_dict.append({
            'id': data['id'],
            'input': data['input'],
            'profile': ranked_profile,
            'user_id': data['user_id'],
            'output': out_dataset[idx]['output']
        })

    print("save file to: {}".format(output_ranking_addr))
    if not os.path.exists(os.path.dirname(output_ranking_addr)):
        os.makedirs(os.path.dirname(output_ranking_addr))
    with open(output_ranking_addr, "w") as file:
        json.dump(rank_dict, file, indent=4, ensure_ascii=False)

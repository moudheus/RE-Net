import argparse
import numpy as np
import torch
import utils
import os
from model import RENet
from global_model import RENet_global
import pickle
from tqdm import trange
import pandas as pd


def predict(args):

    print('Loading data')
    num_nodes, num_rels, _, _, _, _, test_data, test_times, _, _ = utils.load_data(args.dataset)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed_all(999)


    model_state_file = 'models/' + args.dataset + '/rgcn.pth'
    model_graph_file = 'models/' + args.dataset + '/rgcn_graph.pth'
    model_state_global_file2 = 'models/' + args.dataset + '/max' + str(args.maxpool) + 'rgcn_global2.pth'

    print('Loading models')
    model = RENet(num_nodes,
                    args.n_hidden,
                    num_rels,
                    model=args.model,
                    seq_len=args.seq_len,
                    num_k=args.num_k)
    global_model = RENet_global(num_nodes,
                                args.n_hidden,
                                num_rels,
                                model=args.model,
                                seq_len=args.seq_len,
                                num_k=args.num_k, maxpool=args.maxpool)


    if use_cuda:
        model.cuda()
        global_model.cuda()


    with open('data/' + args.dataset+'/test_history_sub.txt', 'rb') as f:
        s_history_test_data = pickle.load(f)
    with open('data/' + args.dataset+'/test_history_ob.txt', 'rb') as f:
        o_history_test_data = pickle.load(f)

    s_history_test = s_history_test_data[0]
    s_history_test_t = s_history_test_data[1]
    o_history_test = o_history_test_data[0]
    o_history_test_t = o_history_test_data[1]


        
    checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.s_hist_test = checkpoint['s_hist']
    model.s_his_cache = checkpoint['s_cache']
    model.o_hist_test = checkpoint['o_hist']
    model.o_his_cache = checkpoint['o_cache']
    model.latest_time = checkpoint['latest_time']
    if args.dataset == "icews_know":
        model.latest_time = torch.LongTensor([4344])[0]
    model.global_emb = checkpoint['global_emb']
    model.s_hist_test_t = checkpoint['s_hist_t']
    model.s_his_cache_t = checkpoint['s_cache_t']
    model.o_hist_test_t = checkpoint['o_hist_t']
    model.o_his_cache_t = checkpoint['o_cache_t']
    with open(model_graph_file, 'rb') as f:
        model.graph_dict = pickle.load(f)

    checkpoint_global = torch.load(model_state_global_file2, map_location=lambda storage, loc: storage)
    global_model.load_state_dict(checkpoint_global['state_dict'])

    print("Using best epoch: {}".format(checkpoint['epoch']))

    test_data = torch.from_numpy(test_data)

    print("Predicting")
    model.eval()
    global_model.eval()
    for ee in range(num_nodes):
        while len(model.s_hist_test[ee]) > args.seq_len:
            model.s_hist_test[ee].pop(0)
            model.s_hist_test_t[ee].pop(0)
        while len(model.o_hist_test[ee]) > args.seq_len:
            model.o_hist_test[ee].pop(0)
            model.o_hist_test_t[ee].pop(0)

    entities = None
    try:
        fn = f'data/{args.dataset}/entities.tsv'
        entities = pd.read_csv(fn, index_col='code', sep='\t').to_dict()['entity']
    except Exception as e:
        print(f'Will not map entities because: {e}')
        
    fn = f'data/{args.dataset}/test_predictions.tsv'
    print(f'Writing predictions in {fn}')
    f = open(fn, 'w')
    f.write('head\trel\ttail\tscore\n')

    for i in trange(len(test_data)):
        batch_data = test_data[i]
        s_hist = s_history_test[i]
        o_hist = o_history_test[i]
        s_hist_t = s_history_test_t[i]
        o_hist_t = o_history_test_t[i]

        if use_cuda:
            batch_data = batch_data.cuda()

        with torch.no_grad():
            loss, sub_pred, ob_pred = model.predict(batch_data, (s_hist, s_hist_t), (o_hist, o_hist_t), global_model)
            s = int(batch_data[0].cpu().numpy())
            r = int(batch_data[1].cpu().numpy())
            if entities:
                s = entities[s]
            scores = ob_pred.cpu().numpy()
            for j, score in enumerate(scores):
                o = j
                if entities:
                    o = entities[o]
                f.write(f'{s}\t{r}\t{o}\t{score}\n')

    f.close()
    print("Done")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RENet')
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS18', help="dataset to use")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--model", type=int, default=3)
    parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--num-k", type=int, default=1000, help="cuttoff position")
    parser.add_argument("--maxpool", type=int, default=1)
    parser.add_argument('--raw', action='store_true')

    args = parser.parse_args()
    predict(args)


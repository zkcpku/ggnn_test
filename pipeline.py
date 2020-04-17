import pandas as pd
import os
from tqdm import tqdm
import pickle
import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt
def draw_dgl(G):
    nx_G = G.to_networkx()
    nx.draw(nx_G, with_labels=True)
    plt.show()

class Pipeline:
    def __init__(self,  ratio, root):
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        path = self.root+output_file
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(self.root+'programs.pkl')

            source.columns = ['id', 'code', 'label']
            print(len(source))

            # code_len = len(source['code'])
            # source_code = list(source['code'])
            # source_ast = []
            # for i in tqdm(range(code_len),ncols = 20):
                # source_ast.append(parser.parse(source_code[i]))
            # source['code'] = source_ast


            source['code'] = source['code'].apply(parser.parse)
            # with open(path,'wb') as f:
            #     pickle.dump(source,f)


            # source.to_pickle(path)
        self.sources = source
        return source

    # split data for training, developing and testing
    def split_data(self,exists = False):

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        train_path = self.root+'train/'
        check_or_create(train_path)
        self.train_file_path = train_path+'train_.pkl'
        # train.to_pickle(self.train_file_path)

        dev_path = self.root+'dev/'
        check_or_create(dev_path)
        self.dev_file_path = dev_path+'dev_.pkl'
        # dev.to_pickle(self.dev_file_path)

        test_path = self.root+'test/'
        check_or_create(test_path)
        self.test_file_path = test_path+'test_.pkl'
        # test.to_pickle(self.test_file_path)



        if exists:
            return
        data = self.sources
        data_num = len(data)
        ratios = [int(r) for r in self.ratio.split(':')]
        train_split = int(ratios[0]/sum(ratios)*data_num)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        data = data.sample(frac=1, random_state=666)
        train = data.iloc[:train_split] 
        dev = data.iloc[train_split:val_split] 
        test = data.iloc[val_split:] 

        # def check_or_create(path):
            # if not os.path.exists(path):
                # os.mkdir(path)
        # train_path = self.root+'train/'
        # check_or_create(train_path)
        # self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        # dev_path = self.root+'dev/'
        # check_or_create(dev_path)
        # self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        # test_path = self.root+'test/'
        # check_or_create(test_path)
        # self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size,exists = False):
        if exists:
            self.size = size
            return
        self.size = size
        if not input_file:
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        from prepare_data import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self,data_path,part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child)) # [父节点,[子节点1,[]],[子节点2],[...]]
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees_code = list(trees['code'])
        code_graph = []
        for e in tqdm(trees_code,ncols = 20,desc = 'generate graph'):
            # tmp = input("show graph")
            # draw_dgl(self.generate_graphs(e))
            code_graph.append(self.generate_graphs(e))
        with open(self.root + part + '/graph.pkl','wb') as f:
            pickle.dump(code_graph,f)
        # trees['code'] = trees['code'].apply(self.generate_graphs)
        trees.to_pickle(self.root+part+'/blocks.pkl')

    def add_children(self,l,g,parent_idx):
        if type(l) == type([]) and len(l) != 0:
            for child in l:
                g.add_nodes(1)
                g.nodes[g.number_of_nodes()-1].data['idx'] = torch.tensor([child[0]])
                g.add_edge(parent_idx,g.number_of_nodes()-1)
                g.add_edge(g.number_of_nodes()-1,parent_idx)
                part_idx = g.number_of_nodes()-1
                self.add_children(child[1:],g,part_idx)
        elif type(l) == type([]) and len(l) == 0:
            return
        else:
            print("maybe something error")
            print(l)
            print(g)
            assert(False)

    def generate_graphs(self,data_list):
        g = dgl.DGLGraph()
        g.add_nodes(1)
        g.nodes[g.number_of_nodes()-1].data['idx'] = torch.tensor([0])
        # g.nodes[g.number_of_nodes()-1].data['idx'] = 0
        for part in data_list:
            g.add_nodes(1)
            g.nodes[g.number_of_nodes()-1].data['idx'] = torch.tensor([part[0]])
            g.add_edge(0,g.number_of_nodes()-1)
            g.add_edge(g.number_of_nodes()-1,0)
            part_idx = g.number_of_nodes()-1
            self.add_children(part[1:],g,part_idx)
        return g


    # run for processing data to train
    def run(self):
        print('parse source code...')
        # self.parse_source(output_file='ast.pkl',option='existing')
        print('split data...')
        self.split_data(exists = True)
        print('train word embedding...')
        self.dictionary_and_embedding(None,128,exists = True)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3:1:1', 'data/')
ppl.run()



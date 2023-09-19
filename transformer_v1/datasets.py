# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data
import numpy as np

# Encoder_input    Decoder_input        Decoder_output
'''
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]                 # P: 占位符号，如果当前句子不足固定长度用P占位
src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}  # 词源字典  字：索引
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}

sentences = [['副 驾 遮 阳 板 化 妆 镜 P P', 'S 化 妆 镜 P P P P P P', '化 妆 镜 P P P P P P E'],         # S: 开始符号
             ['前 排 座 椅 靠 背 文 件 袋 P', 'S 物 品 收 纳 P P P P P', '物 品 收 纳 P P P P P E'],  # E: 结束符号
             ['后 排 功 能 控 制 P P P P', 'S 后 排 功 能 控 制 空 调 P', '后 排 功 能 控 制 空 调 P E']]                 # P: 占位符号，如果当前句子不足固定长度用P占位
src_vocab = {'P': 0, '副': 1, '驾': 2, '遮': 3, '阳': 4, '板': 5, '化': 6, '妆': 7, '镜': 8,
             '前':9, '排':10, '座':11, '椅':12, '靠':13, '背':14, '文':15, '件':16, '袋':17,
             '物':18, '品':19, '收':20, '纳':21,
             '后':22, '功':23, '能':24, '控':25, '制':26,
             '空':27, '调':28}  # 词源字典  字：索引
tgt_vocab = {'P': 0, 'S': 1, 'E': 2, '副': 3, '驾': 4, '遮': 5, '阳': 6, '板': 7, '化': 8, '妆': 9, '镜': 10,
             '前':11, '排':12, '座':13, '椅':14, '靠':15, '背':16, '文':17, '件':18, '袋':19,
             '物':20, '品':21, '收':22, '纳':23,
             '后':24, '功':25, '能':26, '控':27, '制':28,
             '空':29, '调':30}
'''

sentences = np.load('./data/prodata.npy',allow_pickle=True).tolist()
src_vocab = np.load('./data/dic_in.npy',allow_pickle=True).tolist()
tgt_vocab = np.load('./data/dic_out.npy',allow_pickle=True).tolist()

src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)  # 字典字的个数
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}   # 把目标字典转换成 索引：字的形式
tgt_vocab_size = len(tgt_vocab)                         # 目标字典尺寸
src_len = len(sentences[0][0])               # Encoder输入的最大长度
#src_len = len(sentences[0][0].split(" "))
tgt_len = len(sentences[0][1])               # Decoder输入输出最大长度
#tgt_len = len(sentences[0][1].split(" ")) 

# 把sentences 转换成字典索引
def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        #enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        #dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        #dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_input = [[src_vocab[n] for n in sentences[i][0]]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1]]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2]]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

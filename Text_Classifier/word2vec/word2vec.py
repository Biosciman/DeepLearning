import logging
import gensim
from gensim.models import word2vec

'''
使用gensim中的word2vec API来训练模型。
参考：
https://radimrehurek.com/gensim/models/word2vec.html?highlight=word2vec#gensim.models.word2vec.Word2Vec
classgensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, vector_size=100, 
alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, 
min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, 
hashfxn=<built-in function hash>, epochs=5, null_word=0, trim_rule=None, sorted_vocab=1, 
batch_words=10000, compute_loss=False, callbacks=(), comment=None, max_final_vocab=None, shrink_windows=True)
主要参数介绍如下：
1) sentences：我们要分析的语料，可以是一个列表，或者从文件中遍历读出（word2vec.LineSentence(filename) ）。
2) vector_size：词向量的维度，默认值是100。
这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。
如果是超大的语料，建议增大维度。
3) window：即词向量上下文最大距离，window越大，则和某一词较远的词也会产生上下文关系。
默认值为5，在实际使用中，可以根据实际的需求来动态调整这个window的大小。
如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5；10]之间。
4) sg：即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型；是1则是Skip-Gram模型；
默认是0即CBOW模型。
5) hs：即我们的word2vec两个解法的选择了。如果是0， 则是Negative Sampling；是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。
默认是0即Negative Sampling。
6) negative：即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
7) cbow_mean：仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。
在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。
8) min_count：需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
9) iter：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
10) alpha：在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。
11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步。
'''

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("wordEmbdiing.txt")
# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = word2vec.Word2Vec(sentences, vector_size=200, sg=1, epochs=8)
model.wv.save_word2vec_format("word2Vec" + ".bin", binary=True)
# 加载bin格式的模型
wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)
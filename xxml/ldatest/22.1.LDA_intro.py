# !/usr/bin/python
# -*- coding:utf-8 -*-

from gensim import corpora, models, similarities
from pprint import pprint

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    #用lda test 数据 就非常简单的  9行数据那个
    f = open('22.LDA_test.txt')
    #这里造一个停止词  for  a  of  这些
    stop_list = set('for a of the and to in'.split())
    # texts = [line.strip().split() for line in f]
    # print(texts)
    #把停止词都去掉
    texts = [[word for word in line.strip().lower().split() if word not in stop_list] for line in f]
    print 'Text = '
    pprint(texts)

    #得到一个词典，把语料都喂进去 得到一个词典
    dictionary = corpora.Dictionary(texts)
    #词典的长度 是V
    V = len(dictionary)
    #['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications']  把这么一个文档 变成 数字的形式 0  2  3 
    #对每一行的数据  通过词典 做成数字的向量
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print corpus
    #这个是可以直接给lda 的治理再加一个efidf
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    print 'TF-IDF:'
    for c in corpus_tfidf:
        print c

    print '\nLSI Model:'
    #把数据 给lsi 模型   做2个主题 ，词典是dictionary
    lsi = models.LsiModel(corpus_tfidf, num_topics=2, id2word=dictionary)
    # 得到lsi 之后 可以把 语料 放到 lsi 里面   结果出来 就是主题模型的结论
    #lsi[corpus_tfidf]  是主题分布
    topic_result = [a for a in lsi[corpus_tfidf]]
    pprint(topic_result)
    print 'LSI Topics:'
    #打印词分布
    pprint(lsi.print_topics(num_topics=2, num_words=5))
    # 拿文档的主题分布 计算相似度
    similarity = similarities.MatrixSimilarity(lsi[corpus_tfidf])   # similarities.Similarity()
    #这样用tf idf  也能算相似度
    #similarity = similarities.MatrixSimilarity(corpus_tfidf) 
    print 'Similarity:'
    pprint(list(similarity))

    print '\nLDA Model:'
    num_topics = 2
    #用lda 来做  corpus_tfidf  语料放进去    2个超参数  alpha  eta  就是 beta  都是 auto  是自动算的
    #minimum_probability 值的意思 是如果某一个主题  它的算的值 小于 0.001  就不要这一个了
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha='auto', eta='auto', minimum_probability=0.001)
    #得到文档的主题分布
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    print 'Document-Topic:\n'
    pprint(doc_topic)
    for doc_topic in lda.get_document_topics(corpus_tfidf):
        print doc_topic
    for topic_id in range(num_topics):
        print 'Topic', topic_id
        # pprint(lda.get_topic_terms(topicid=topic_id))
        #主题的词分布
        pprint(lda.show_topic(topic_id))
    similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
    print 'Similarity:'
    pprint(list(similarity))
    #hdp  结构化的lda
    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    print '\n\nUSE WITH CARE--\nHDA Model:'
    pprint(topic_result)
    print 'HDA Topics:'
    print hda.print_topics(num_topics=2, num_words=5)

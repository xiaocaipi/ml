'''
Created on Dec 22, 2016

@author: caidanfeng733
'''






class cfoo(object):
    def __init__(self, name='Macy'):
        self.name = name
        print 'at init:' + self.name
    def __del__(self):
        self.name = None
        print 'at del:', self.name
    def p(self):
        print self.name

c = cfoo('john')
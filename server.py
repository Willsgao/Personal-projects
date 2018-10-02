#! -*- coding:utf-8 -*-
'''
name:willsgao
date:2018-09-27
project:dictionary
email:willsgao@163.com
'''

from server.serverclass import MyDictory


# 主函数运行功能类
def main():
    ADDR = ('0.0.0.0', 8989)
    dictserver = MyDictory('mydict', ADDR)
    dictserver.bind()
    dictserver.serverForever()


if __name__ == '__main__':
    main()

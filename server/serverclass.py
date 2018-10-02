#! -*- coding:utf-8 -*-
'''
name:willsgao
date:2018-09-27
project:dictionary
email:willsgao@163.com
'''
from socket import *
from threading import Thread
import re
from .mysqlpy import Mysqlhelp
from time import ctime,sleep

# 将电子词典里的关键单词设为查询算法中的参考点
DICT_LIST = ['a','babble','cabal','dabble','each','fable',
'gab','habit','i','jab','kaleidoscope','lab',\
'macabre','n','oaf','P.M.','quack','rabbit',\
's','tabernacle','u','vacancy','wacky','xenophobe',\
'yacht','zany']

#　创建服务端的功能类
class MyDictory(object):
    def __init__(self,dbs,addr):
        self.sockfd = socket()
        self.sockfd.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.dbs = dbs      # 导入数据库字典
        self.addr = addr
        self.myhelp = Mysqlhelp(self.dbs)

    def bind(self):
        self.sockfd.bind(self.addr)

    # 采用多线程处理客户端访问的并发问题
    def serverForever(self):
        self.sockfd.listen(5)
        while True:
            conn, cliaddr = self.sockfd.accept()
            print('成功连接到', cliaddr)
            client_handle = Thread(target=self.handler,\
                args=(conn,cliaddr))
            client_handle.start()

    # 多线程循环接收客户端访问请求，并处理请求
    def handler(self, conn,cliaddr):
        while True:
            data = conn.recv(128).decode()
            if data[0] == 'R':
                self.do_register(conn,data)
            elif data[0] == 'L':
                self.do_login(conn,data)
                # print('你准备好了没？')
                continue
            elif data[0] == 'E':
                conn.close()
                print('%s已断开连接．'%cliaddr)
            else:
                pass

    # 服务端处理客户端的用户注册申请
    def do_register(self,conn,data):
        name = data.split(' ')[1]
        passwd = data.split(' ')[2]
        # print('老兄，你到底发了没有？')
        sql = "select * from user where username= '%s'"%name
        res1 = self.myhelp.getone(sql)
        # print('我已经收到！')
        if res1 != None:
            # print('我已经收到１１１１！')
            conn.send(b'Exist')
        else:
            # print('我已经收到2222！')
            sql = "insert into user (username, passwd)\
             values('{}','{}')".format(name,passwd)
            self.myhelp.work(sql)
            # print('我已经收到3333！')
            conn.send(b'OK')

    # 处理用户登录申请
    def do_login(self,conn,data):
        name = data.split(' ')[1]
        passwd = data.split(' ')[2]
        sql = "select * from user where username= '%s'"%name
        res1 = self.myhelp.getone(sql)
        # print('我已经收到！')
        if res1 != None:
            print('我已经收到１１１１！')
            sql = "select passwd from user where username= '%s'"%name
            passwd1 = self.myhelp.getall(sql)[0][0]
            print(passwd1)
            if passwd == passwd1:
                conn.send(b'OK')
                name1 = name
                self.do_work(conn,name1)
                return
            else:
                conn.send(b'Wrong')
        else:
            print('该用户不存在！')
            conn.send(b'FALL')
        return

    # 处理用户登录后的单词相关访问申请
    def do_work(self,conn,name1):
        L = DICT_LIST
        def do_search(data):
            i = 0
            msg = ''
            print('收到的data是',　data)
            while True:
                # 将电子词典中的关键单词赋值给变量a,b，便于查询
                a,b = L[i],L[i+1]
                # print('a=',a)
                # print('b=',b)
                if str('%s')%data >= str('%s')%a and\
                 str('%s')%data <= str('%s')%b:
                    word1 = a
                    word2 = b
                    print('准备查word1')
                    sql1 = "select interpret from words where \
                    word='%s'"%data
                    res2 = self.myhelp.getone(sql1)
                    # print('word1=',word1)
                    # print('word2=',word2)
                    # print('ses2=',res2)

                    # 单词查询成功，将查询历史记录信息插入数据库
                    if res2 != None:
                        msg = res2[0]
                        atime = ctime()
                        print('准备插入！')
                        do_in_hist(name1,data)
                        print('插入成功！')
                    else:
                        msg = 'No such word!'
                    break
                elif data >= b:
                    i += 1
                    continue
                else:
                    msg = 'No such word!'
                    break
            conn.send(msg.encode())
            return

        # 具体执行单词查询历史记录的插入函数
        def do_in_hist(name1,data):
            sql = "insert into hist(name,word) values\
                        ('{}','{}')".format(name1,data)
            self.myhelp.work(sql)

        # 处理客户端访问查询记录的函数
        def do_history(data):
            sql = "select * from hist where word = '%s'"%data
            res2 = self.myhelp.getall(sql)
            print(res2)
            if len(res2) > 0:
                for line in res2:
                    name = line[2]
                    word = line[3]
                    time = line[4]
                    msg = str('{}在{}查询了单词{}'.format(name,time,word))
                    conn.send(msg.encode())
                    sleep(0.1)
                conn.send('**'.encode())
            else:
                conn.send(b'Nohist')

        # 处理客户端具体访问请求的主程序
        while True:
            msg = conn.recv(128).decode()
            if msg[0] == 'W':
                data = msg.split(' ')[1]
                # print('单词已收到')
                do_search(data)
                print('单词已发出')
            elif msg[0] == 'H':
                data = msg.split(' ')[1]
                print('单词已收到')
                do_history(data)
            elif msg == '*back*':
                # print('我收到了******')
                break
            else:
                pass

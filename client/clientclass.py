#! -*- coding:utf-8 -*-
'''
name:willsgao
date:2018-09-27
project:dictionary
email:willsgao@163.com
客户端利用命令行传参访问服务端
'''
from socket import *
import sys
import re
# 导入getpass模块，实现密码隐藏的功能
from getpass import getpass

#　创建客户端访问功能类
class ClientVisit(object):
    def __init__(self):
        self.sockfd = socket()

    def connect(self,addr):
        try:
            self.sockfd.connect(addr)
        except Exception as e:
            print(e)

    # 提交用户名注册申请
    def do_register(self):
        while True:
            try:
                name = input('请输入注册用户名：')
                if ' ' in name:
                    print('用户名中不允许有空格！')
                    break
                # 隐藏密码函数
                passwd = getpass('请输入用户密码：')
                passwd1 = getpass('请再次输入密码：')
                if passwd1 != passwd:
                    print('两次密码不同，请重新输入！')
                    break
                msg = 'R {} {}'.format(name,passwd)
                self.sockfd.send(msg.encode())
                data = self.sockfd.recv(128).decode()
                # print(data)
                if data == 'OK':
                    print('注册成功，请登录！')
                    self.do_login()
                    return
                elif data == 'Exist':
                    print('用户已存在，请重新输入！')
                    break
                else:
                    print('未知原因失败，请重新输入！')
                    break
            # Ctrl+C　退回到上一级命令
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(e)

    # 提交用户登录申请
    def do_login(self):
         while True:
            try:
                name = input('请输入用户名：')
                passwd = getpass('请输入密码：')
                msg = 'L {} {}'.format(name,passwd)
                self.sockfd.send(msg.encode())
                data = self.sockfd.recv(128).decode()
                # print(data)
                if data == 'OK':
                    print('恭喜您，登录成功！')
                    self.do_work()
                    return
                elif data == 'Wrong':
                    print('密码错误，请重新输入！')
                    break
                elif data == 'FALL':
                    print('该用户不存在，请重新输入！')
                else:
                    print('未知原因失败，请重新输入！')
                    break
            except KeyboardInterrupt:
                return
            except Exception as e:
                print(e)

    # 提交用户登录申请
    def do_work(self):
        # 定义单词查询函数
        def do_search():
            while True:
                try:
                    word = input('请输入单词(Ctrl+C退出)：')
                    if not word:
                        continue
                    msg = 'W {}'.format(word)
                    self.sockfd.send(msg.encode())
                    # print('看好了我已经发出请求了',msg)
                    data = self.sockfd.recv(1024).decode()
                    print(data)
                # Ctrl+C退回到上一级命令
                except KeyboardInterrupt:
                    return
        # 定义单词查询历史的访问函数
        def do_history():
            while True:
                try:
                    m = input('请输入要查询历史记录的单词(Ctrl+C退出)：')
                    msg = 'H {}'.format(m)
                    self.sockfd.send(msg.encode())
                    while True:
                        data = self.sockfd.recv(128).decode()
                        if data == '**':
                            break
                        elif data == 'Nohist':
                            print('无%s相关历史查询记录'%m)
                            break
                        else:
                            print(data)
                # Ctrl+C退回到上一级命令
                except KeyboardInterrupt:
                    return                        


        while True:
            visit_menu()
            try:
                msg = input('请输入查询命令：')
                if msg[0] == 'W':
                    do_search()
                elif msg[0] == 'H':
                    do_history()
                else:
                    pass
            # 将客户端退出二级访问页面的消息通知给服务端，
            # 服务端配合退回到一级服务循环
            except KeyboardInterrupt:
                self.sockfd.send(b'*back*')
                return


#　客户端访问主命令菜单
def main_menu():
    print('*****操作命令*****')
    print('*****注册：R *****')
    print('*****登录：L *****')
    print('*****退出：E *****')
    print('*****************')

#　客户端访问主命令菜单
def visit_menu():
    print('*****查询命令*****')
    print('*****查词：W *****')
    print('****查记录：H ****')
    print('***返回:Ctrl+C***')
    print('*****************')



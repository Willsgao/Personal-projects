#! -*- coding:utf-8 -*-
'''
name:willsgao
date:2018-09-27
project:dictionary
email:willsgao@163.com
客户端利用命令行传参访问服务端
'''


from client.clientclass import ClientVisit
from client.menu import *


# 客户端访问命令主函数
def main():
    my_visit = ClientVisit()
    my_visit.connect(('127.0.0.1', 8989))
    while True:
        main_menu()
        msg = input('请输入命令：')
        if msg[0] == 'R':
            my_visit.do_register()
            # return
        elif msg[0] == 'L':
            my_visit.do_login()
            # return
        elif msg[0] == 'E':
            my_visit.sockfd.send(b'E')
            sys.exit('您已成功退出！')
        else:
            pass


if __name__ == '__main__':
    main()

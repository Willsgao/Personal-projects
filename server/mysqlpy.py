#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
此模块将mysql功能进行封装；
主模块执行SQL语句时调用此模块中的类方法
'''
from pymysql import *


#定义一个sql方法类，利用此类的实例调用类方法：
class Mysqlhelp(object):
    def __init__(self, database, host='localhost', user='root',\
     password='123456', charset='utf8', port=3306):
        self.database = database
        self.host = host
        self.user = user
        self.password = password
        self.charset = charset
        self.port = port

    # 创建数据库连接和游标，封装为open方法
    def open(self):
        self.conn = connect(host=self.host, user=self.user,
                            password=self.password, database=self.database,
                            charset=self.charset, port=self.port)
        self.cur = self.conn.cursor()

    # 关闭方法，在SQL语句执行完毕后，将连接变量和游标变量关闭
    def close(self):
        self.cur.close()
        self.conn.close()

    # 执行SQL语句
    def work(self, sql, L=[]):
        self.open()
        try:
            self.cur.execute(sql,L)
            self.conn.commit()
            return 1
        except Exception as e:
            self.conn.rollback()
            print("Failed", e)
            return

        self.close()

    # 执行查询方法，查询全部或一条信息
    def getall(self, sql, L=[]):
        self.open()
        self.cur.execute(sql, L)
        # print("ok")
        result = self.cur.fetchall()
        self.close()
        return result

    def getone(self, sql, L=[]):
        self.open()
        self.cur.execute(sql, L)
        result = self.cur.fetchone()
        self.close()
        return result


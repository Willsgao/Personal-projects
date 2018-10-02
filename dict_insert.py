'''
利用函数将字典内容导入到目标数据库
'''
from server.mysqlpy import Mysqlhelp
import re

def dictstore(filename,dbs):
    try:
        fd = open(filename)
    except IOError:
        print('文件打开失败！')
    files = fd.readlines()
    mysql1 = Mysqlhelp(dbs)

    for line in files:
        # l = re.split(r'\s+', line)
        # word1 =l[0]
        # interpret1 = ' '.join(l[1:])
        word1 = re.search(r'(^[\S]+\b)',line).group()
        interpret1 = re.search(r'^[\S]+\s+([\S].*$)',line).group(1)
        # print('word = ',word1)
        # print('inter = ',interpret1)
        sql = 'insert into words (word,interpret) values("{}","{}")'.format(word1, interpret1)
        mysql1.work(sql)
    fd.close()


dictstore('dict1.txt','mydict')





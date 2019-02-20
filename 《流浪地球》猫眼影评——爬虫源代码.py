
# coding: utf-8

# In[ ]:


# 导入爬虫所需工具库
import time,random
import datetime as dt
import requests
import json


# In[ ]:


# 8个备用user_agents
user_agents = [
            {'User-Agent': 'MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22;\
            CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1'},
            {'User-Agent': 'Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 \
            (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0'},
            {'User-Agent': 'Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5;\
            Trident/5.0; IEMobile/9.0; HTC; Titan)'},
            {'User-Agent': 'Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019;\
            Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124'},
            {'User-Agent': 'Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) \
            AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13'},
            {'User-Agent': 'Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us)\
            AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5'},
            {'User-Agent': 'Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) \
            AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5'},
            {'User-Agent': 'Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91)\
            AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1'},
            ]


# In[ ]:


# 创建爬虫类
class MovieSpider(object):
    def __init__(self, filename):
        self.headers = user_agents
        self.filename = filename
        
    def get_data(self, header, url):
        '''
        功能：访问url的网址，获取网页内容并返回
        参数：url,目标网页的url
        返回：目标网页的html内容
        '''
        try:
            r = requests.get(url, headers=header)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(e)
    
    def parse_data(self, html):
        '''
        功能：提取 html 页面信息中的关键信息，并整合一个数组并返回
        参数：html 根据 url 获取到的网页内容
        返回：存储有 html 中提取出的关键信息的数组
        '''
        json_data = json.loads(html)['cmts']
        comments = []
        
        try:
            for item in json_data:
                comment = []
                # 提取影评中的6条数据：nickName(昵称),cityName(城市),content(评语)，
                # score(评分),startTime(评价时间),gender(性别)
                comment.append(item['nickName'])
                comment.append(item['cityName'] if 'cityName' in item else '')
                comment.append(item['content'].strip().replace('\n', ''))
                comment.append(item['score'])
                comment.append(item['startTime'])
                comment.append(item['gender']  if 'gender' in item else '')
                
                comments.append(comment)

            return comments
        
        except Exception as e:
            print(comment)
            print(e)
    
    def save_data(self, comments):
        '''
        功能：将comments中的信息输出到文件中/或数据库中。
        参数：comments 将要保存的数据
        '''
        df = pd.DataFrame(comments)
        df.to_csv(self.filename, mode='a', encoding='utf_8_sig',
                  index=False, sep=',', header=False)
    
    def run(self, time_lists):
        '''
        功能：爬虫调度器，根据规则每次生成一个新的请求 url，爬取其内容，并保存到本地。
        '''
#         start_time = dt.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        start_time = time_lists[0]  # 电影上映时间，评论爬取到此截至
        end_time = time_lists[-1]  # 电影上映时间，评论爬取到此截至
        print('*******************')
        
        # 抓取评论信息
        i = 0
        while start_time > end_time:
            i += 1
            if i%10 ==0:
                print('已爬取%s页评论'%i)
            url = 'http://m.maoyan.com/mmdb/comments/movie/248906.json?_v_=            yes&offset=0&startTime=' + start_time.replace('  ', '%20')
            header = random.choice(self.headers)
            time.sleep(0.05)
            html = None
            
            try:
                html = self.get_data(header, url)
            except Exception as e:
                print('*************************')
                time.sleep(0.83)
                html = self.get_data(url)
                print(e)

            else:
                time.sleep(0.3)

            # 解析评论信息
            comments = self.parse_data(html)
            start_time = comments[14][4]

            start_time = dt.datetime.strptime(
                start_time, '%Y-%m-%d  %H:%M:%S') + dt.timedelta(seconds=-1)
            start_time = dt.datetime.strftime(start_time, '%Y-%m-%d  %H:%M:%S')

            self.save_data(comments)


# In[ ]:


# 通过改变时间点，选择爬取信息所处的时间段
t1 = ['2019-02-12  18:59:59','2019-02-12  00:00:00']
time_lists = t1
filename = '流浪地球%s_comments.csv'%time_lists[1].split()[0]
spider = MovieSpider(filename)
spider.run(time_lists)
print('爬取信息结束')


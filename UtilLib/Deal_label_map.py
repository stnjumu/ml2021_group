import numpy as np

class Deal_label_map:
    """[Deal_label_map类] 用于读取label_map.txt拆分单一标签为3个标签，由[0,196)的label获取品牌标签和年份标签
        可尝试品牌grand_label和年份year_label
        使用下面3个函数即可：
        grand_label, year_label = get_grand_year(label)
        True/False = judge_grand(grand_label, label)
        True/False = judge_year(year_label, label)
    """    
    
    def __init__(self, label_map_path='./dataset/label_map.txt'):
        self.label_map_list=[]
        
        self.grand_list=[]
        self.series_list=[]
        self.year_list=[]
        
        self.unique_grand_list=[]
        self.unique_series_list=[]
        self.unique_year_list=[]
        
        self.grand_numList=[]
        self.series_numList=[]
        self.year_numList=[]
        with open(label_map_path) as f:
            dates = list(f.readlines())
        for i,item in enumerate(dates, start=0): # 1 AM General Hummer SUV 2000
            item = item.split() # ['1', 'AM', 'General', 'Hummer', 'SUV', '2000']
            self.label_map_list.append(item)
            
            # print(item)
            assert i == int(item[0])-1
            
            self.grand_list.append(item[1])
            self.series_list.append(" ".join(item[2:-1]))
            self.year_list.append(int(item[-1]))
            # if i>20:
            #     break
        
        # print(1, self.grand_list)
        # print(2, self.series_list)
        # print(3, self.year_list)
        
        self.unique_grand_list = np.sort(np.unique(np.array(self.grand_list))).tolist()
        self.unique_series_list = np.sort(np.unique(np.array(self.series_list))).tolist()
        self.unique_year_list = np.sort(np.unique(np.array(self.year_list))).tolist()
        
        # print(1, len(self.unique_grand_list), self.unique_grand_list)
        # print(2, len(self.unique_series_list), self.unique_series_list)
        # print(3, len(self.unique_year_list), self.unique_year_list)
        
        self.grand_numList = [self.unique_grand_list.index(item) for item in self.grand_list]
        self.series_numList = [self.unique_series_list.index(item) for item in self.series_list]
        self.year_numList = [self.unique_year_list.index(item) for item in self.year_list]
        
        # print(1,len(self.grand_numList), self.grand_numList)
        # print(1,len(self.series_numList), self.series_numList)
        # print(1,len(self.year_numList), self.year_numList)
        
        # 测试
        assert len(self.grand_numList) == len(self.series_numList)
        assert len(self.grand_numList) == len(self.year_numList)
        for i in range(len(self.grand_numList)):
            assert self.unique_grand_list[self.grand_numList[i]] == self.grand_list[i]
            assert self.unique_series_list[self.series_numList[i]] == self.series_list[i]
            assert self.unique_year_list[self.year_numList[i]] == self.year_list[i]
            
    def get_grand_year(self, label):
        assert label>=0 and label<196
        return self.grand_numList[label], self.year_numList[label]
    
    def judge_grand(self, grand_label, label):
        assert label>=0 and label<196
        assert grand_label>=0 and grand_label<len(self.unique_grand_list)
        return self.grand_numList[label] == grand_label
    
    def judge_year(self, year_label, label):
        assert label>=0 and label<196
        assert year_label>=0 and year_label<len(self.unique_year_list)
        return self.year_numList[label] == year_label

if __name__ == '__main__':
    label_map = Deal_label_map()
    print("品牌类别个数", len(label_map.unique_grand_list))
    print("系列类别个数", len(label_map.unique_series_list))
    print("年份类别个数", len(label_map.unique_year_list))
    
    # 用法示例
    label = 0
    print(label_map.label_map_list[label])
    grand_label, year_label = label_map.get_grand_year(label)
    print("label = {} 对应的品牌label为 {} , 对应的年份label为 {} ".format(label,grand_label, year_label))
    print("品牌label = {} 即{}".format(grand_label, label_map.unique_grand_list[grand_label]))
    print("年份label = {} 即{}".format(year_label, label_map.unique_year_list[year_label]))
    
    print("判断label = {}的品牌是不是grand_label={} : {}".format(label, grand_label, label_map.judge_grand(grand_label, label)))
    print("判断label = {}的品牌是不是grand_label={} : {}".format(label, grand_label+1, label_map.judge_grand(grand_label+1, label)))
    
    print("判断label = {}的年份是不是year_label={} : {}".format(label, year_label, label_map.judge_year(year_label, label)))
    print("判断label = {}的年份是不是year_label={} : {}".format(label, year_label+1, label_map.judge_year(grand_label+1, label)))
import numpy as np

class Deal_label_map:
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
        assert label>=0 and label<=195
        return self.grand_numList[label], self.year_numList[label]

if __name__ == '__main__':
    label_map = Deal_label_map()
    label = 0
    print(label_map.label_map_list[label])
    grand_label, year_label = label_map.get_grand_year(label)
    print("label = {} 对应的品牌为 {} , 对应的年份为 {} ".format(label,grand_label, year_label))
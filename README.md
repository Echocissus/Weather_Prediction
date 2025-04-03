# Weather_Prediction
USING LSTM TO PREDICT FUTURE TEMPERATURES OF GUANGZHOU
可视化结果：https://public.tableau.com/app/profile/yuxuan.wu7298/viz/_17435940787660/1_1
code在Prediction中，运行其中的"\data_analysis\datas\dataload.py"即可展现结果
同目录下的separate_training和training分别是预测最高温和最低温的lstm模型的训练

spider目录下的三个py文件为爬虫程序

data_analysis下的data_cleaningguangzhou为数据清洗的代码，基本上爬取的数据无缺失值，缺失的数据用上一个值代替。

由于爬取数据有限，目前只有广州一个城市的预测数据

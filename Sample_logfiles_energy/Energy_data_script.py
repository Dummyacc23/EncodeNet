#!/usr/bin/env python
# coding: utf-8

# In[183]:





# In[436]:


#sample time stamp
time_stamp =['2023-03-18 15:11:49',
'2023-03-18 15:12:41',
             
'2023-03-18 15:12:46',
'2023-03-18 15:13:18',
             
             
'2023-03-18 15:13:23',
'2023-03-18 15:13:42',
             
'2023-03-18 15:13:47',
'2023-03-18 15:13:59',
             
'2023-03-18 15:14:04',
'2023-03-18 15:14:12',
             
             
'2023-03-18 15:14:17',
'2023-03-18 15:14:23',
             
             
'2023-03-18 15:14:28',
'2023-03-18 15:14:32',
             
'2023-03-18 15:14:37',
'2023-03-18 15:14:41',
             
             
'2023-03-18 15:14:46',
'2023-03-18 15:14:50'
            ]

        


# In[438]:


import pandas as pd
data = pd.read_csv(r'sorted_file.csv')   
data.columns =['Time Stamp ', 'A', 'B', 'Power', 'D','E','F','G','H','I','J','K','L','M']
for idx in range(0,len(time_stamp),2):
    start_time_stamp = time_stamp[idx]
    end_time_stamp= time_stamp[idx+1]
    
    first_index = min(data.index[data['Time Stamp '] == start_time_stamp].tolist())
    second_index = max(data.index[data['Time Stamp '] == end_time_stamp].tolist())
    #print(first_index)
    val = second_index - first_index
    #print(val)
    power_data= data['Power'][first_index:second_index+1]
    power_float=[]
    for i in power_data:
        power_float.append(float(i))
    power= ((sum(power_float))/(second_index-first_index+1))
    print(power)
#energy_list.append(power)


# In[ ]:


2023-03-13 03:27:49
2023-03-13 03:40:50


# In[394]:


import pandas as pd
data = pd.read_csv(r'sorted_file.csv')   
data.columns =['Time Stamp ', 'A', 'B', 'Power', 'D','E','F','G','H','I','J','K','L','M']
start_time_stamp = '2023-03-13 03:27:49'
end_time_stamp ='2023-03-13 03:40:50'
first_index = min(data.index[data['Time Stamp '] == start_time_stamp].tolist())
second_index = max(data.index[data['Time Stamp '] == end_time_stamp].tolist())
print(first_index)
val = second_index - first_index
print(val)
power_data= data['Power'][first_index:second_index+1]
power_float=[]
for i in power_data:
    power_float.append(float(i))
power= ((sum(power_float))/(second_index-first_index+1))
print(power)
energy_list.append(power)


# In[192]:


print(energy_list)


# In[144]:


len(energy_list)


# In[145]:


sum(energy_list)/3


# In[401]:


time = [52.3596189,
31.76748323,
18.695122,
12.1330452,
7.892506123,
5.855344534,
4.354925394,
3.704434395,
3.42743]
batch= [1,2,4,8,16,32,64,128,256]

for times , batchs in zip(time,batch) : 
    batch_inference_time = (times/(10000/batchs))*1000
    print(batch_inference_time)


# In[239]:


49.8816735744476*3.5


# In[434]:


## batch energy _claculation 
batch_time = [5.23596189,
6.353496646,
7.4780488,
9.70643616,
12.6280098,
18.73710251,
27.87152252,
47.41676026,
87.742208 ]
power = [3.563584905660376,
3.664424242424246,
3.8597499999999996,
3.9907692307692297,
4.346555555555556,
4.423285714285716,
4.6244,
4.509399999999999,
4.717]


for x , y in zip(batch_time,power) : 
    z = x*y
    print(z)


# In[435]:


##enrgy per image 
energy_consumed = [18.658794757816974,
23.281907133763536,
28.863398855799996,
38.73614676775384,
54.8883461518,
82.87955785959002,
128.889068741488,
213.82113871644395,
413.879995136]

batch= [1,2,4,8,16,32,64,128,256]


for x , y in zip(energy_consumed,batch) : 
    z = x/y
    print(z)


# In[459]:


datas= [
 163.1603539,
160.4914172,
98.9193921,
83.93493454,
87.87399881,
87.04245683,
85.49778804,
39.50451658,
40.82131054
    
]
for data in datas:
    print(data*1000)


# In[ ]:





# In[ ]:


'''
cat *.csv > merged_file.csv
sort -t ',' -k1 merged_file.csv > sorted_file.csv
'''


# In[ ]:


import os 
import gc
batchsizes= [1,2,4,8,16,32,64,128,256]
#print("Batch Sizes-----------> ", batch)
#print()
#cpu_time_a = (time.time(), psutil.cpu_times())
#start_inference= time.time()
import datetime;
#cpu_time_a = (time.time(), psutil.cpu_times())
#start_inference= time.time()\
#test= test_data.reshape(-1,784)
for batch in batchsizes:
    print(batch)
    print()
    cpu_time_a = (time.time(), psutil.cpu_times())
    start= time.time()
    ct3 = datetime.datetime.now()
    score = prunned_lenet.evaluate(test, y_test , batch_size= batch , verbose=0)
    ct4 = datetime.datetime.now()
    #print(ct3)
    #print(ct4)
    end= time.time()
    #print("Time Taken",end- start) 
    cpu_time_b = (time.time(), psutil.cpu_times())
    print(ct3)
    print(ct4)
    print("Time Taken",end- start)
    t = cpu_time_b[0] - cpu_time_a[0]
    x = calculate(cpu_time_a[1], cpu_time_b[1])
    print("CPU Usage",x)
    print()
    time.sleep(5)
#print("EncodeNet Lenet : ", t)
#print('EncodeNet Lenet Cpu usage ', x)
#end_inference = time.time()
#inference_time= end_inference- start_inference
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print("EncodeNet for Lenet time taken", inference_time)

#cpu_time_a = (time.time(), psutil.cpu_times())
#score = lenet_base.evaluate(test_data, y_test, batch_size= batch,verbose=0)
#cpu_time_b = (time.time(), psutil.cpu_times())
#base_time= end-start
#t = cpu_time_b[0] - cpu_time_a[0]

#x = calculate(cpu_time_a[1], cpu_time_b[1])


#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
gc.collect()


# In[ ]:


import psutil
import time

def calculate(t1, t2):
    # from psutil.cpu_percent()
    # see: https://github.com/giampaolo/psutil/blob/master/psutil/__init__.py
    t1_all = sum(t1)
    #print(t1_all)
    t1_busy = t1_all - t1.idle
    t2_all = sum(t2)
    t2_busy = t2_all - t2.idle
    if t2_busy <= t1_busy:
        return 0.0
    busy_delta = t2_busy - t1_busy
    all_delta = t2_all - t1_all
    busy_perc = (busy_delta / all_delta) * 100
    return round(busy_perc, 1)


# In[274]:


142.14110851287842*20


# In[ ]:





# In[ ]:





# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:24:47 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
import os
os.chdir('C:/Users/Administrator/Desktop/ecom/finandata')
 
def quantopian2pd(filename='C:/Users/Administrator/Desktop/ecom/finandata/spy.txt',
                  outname='spy_day.csv'):
    date_l=[]
    open_l=[]
    high_l=[]
    low_l=[]
    close_l=[]
    f=open(filename,'r')
    a=f.readline().strip()
    while len(a)!=0:
        if a[0]=='o':
            a=f.readline().strip()    
            continue
        else:
            date_l.append(a.split(' ')[0]+' '+a.split(' ')[1])
            counter=0
             
            for character in a.split(' ')[2:]:
                 
                if len(character)==0:
                    continue
                else:
                    if counter==0:
                        open_l.append(float(character))
                        counter+=1
                    if counter==1:
                        high_l.append(float(character))
                        counter+=1
                    if counter==2:
                        low_l.append(float(character))
                        counter+=1
                    if counter==3:
                        close_l.append(float(character))
                        counter+=1
        a=f.readline().strip()    
    f.close()
    
    spypd=pd.DataFrame()
    spypd['date']=date_l
    spypd['open']=open_l
    spypd['high']=high_l
    spypd['low']=low_l
    spypd['close']=close_l
    spypd.to_csv(outname,index=False)
    
    
 

quantopian2pd('spy_1m.txt','spy_1min.csv')
def change_freq(in_file='C:/Users/Administrator/Desktop/ecom/finandata/spy_1min.csv',\
                out_file='C:/Users/Administrator/Desktop/ecom/finandata/spy_5min.csv',out_freq='15T'):
    sh=pd.read_csv(in_file)
     
    #sh=pd.read_csv('000001.XSHG')
     
    sh=sh.set_index(sh.columns[0])
    
    
    #spy=pd.read_csv('spy.txt',sep='\t')
     #spy=spy.set_index('date')
    #
    sh.index=pd.to_datetime(sh.index)
    if out_freq[-1]=='D':
        shd=sh.resample('1D' ).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    if out_freq[-1]=='T':
        shd=sh.resample(out_freq,label='left',base=1).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    
    shd.to_csv(out_file)
    
    

shday=pd.read_csv('sh_1d.csv')
shday =shday.set_index('Unnamed: 0')
ret=shday.values[1:,0]/shday.values[:-1,3]-1
ret.mean()

spyday=pd.read_csv('spy_15min.csv')
spyday =spyday.set_index(spyday.columns[0])
spyday.index=pd.to_datetime(spyday.index)
spyday['2003-09-22']
spyday.loc[datetime.datetime(2003,9,22)]
spyret_up=spyday.values[1:,1]/spyday.values[:-1,3]-1>0.005
spyret_down=spyday.values[1:,2]/spyday.values[:-1,3]-1<-0.005

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
spyday.values[:-1]

X=spyday.values[:-1]/
y=spyret_up.astype(np.int)
trainx=X[:2000]
trainy=y[:2000]
testx=X[2000:]
testy=y[2000:]
lr=LogisticRegression(random_state=0).fit(trainx,trainy)
lr.coef_
lr.score(trainx,trainy)
lr.score(testx,testy)
lr.predict(trainx)
lr.predict(testx)

def on_date(df,i,alldays):
    return df[(df.index>=alldays[i])&(df.index<alldays[i+1]) ]

def to_daily_ret(df):
    for i in range(1,df.shape[0]):
        
def maketrain(days=1,spyday=spyday,target=0.005,pre=0):
    assert days>=1
    spyday=pd.read_csv('spy_15min.csv')
    spyday =spyday.set_index(spyday.columns[0])
    spyday.index=pd.to_datetime(spyday.index)
    day_df=spyday.resample('1D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    all_days=  day_df.index 
    total_days=all_days.shape[0]
    l1=[]
    for i in range(1,total_days-1 ):
        lastclose=on_date(spyday,i-1,all_days).values[-1,-1]
        ret_on_day_i=on_date(spyday,i,all_days).values/lastclose-1 
        l1.append(ret_on_day_i )
    #from day1
    
    bars_each_day=max(l1[0].shape[0],l1[1].shape[0])
    day2_ret=day_df.values[2:]/day_df.values[:-2,3].reshape(-1,1)-1
    #from day 2
    x=[]
    y=[]
    assert day_df.shape[0]>max(days,2)
    assert pre<bars_each_day
    for day_idx in range(max(days+1,2),total_days-1 ):
        #print(day_idx)
        l1_idx=day_idx-1
        day2_ret_idx=day_idx-2
        if l1[l1_idx].shape[0]<bars_each_day:#early close
            bars_remaining_after_close=bars_each_day-l1[l1_idx].shape[0]
            number_of_items=l1[l1_idx].shape[1]
            l1[l1_idx]=np.concatenate((l1[l1_idx],\
  l1[l1_idx][-1,-1]*np.ones( (bars_remaining_after_close,number_of_items),dtype=np.float64  )),axis=0  ) 
        
        today_now_bars=l1[l1_idx][:bars_each_day-pre]
         
         
        
        tb_append=np.concatenate((l1[l1_idx-days:l1_idx-1]+[today_now_bars]),axis=0).reshape(-1)
         
        x.append(tb_append)  
        now2_ret=l1[l1_idx][bars_each_day-pre-1,3]
        y.append( int ( day2_ret[day2_ret_idx,1]-now2_ret > target ) )
        
    X=np.stack(x,axis=0) 
    Y=np.array(y)
    train_num=int( X.shape[0]*0.7 )
    trainx=X[:train_num]
    trainy=Y[:train_num]
    testx=X[train_num:]
    testy=Y[train_num:]
    return trainx,trainy,testx,testy

trainx,trainy,testx,testy=maketrain(1,spyday,0.005,0)


lr=MLPClassifier(hidden_layer_sizes=(50),random_state=0,max_iter=1000).fit(trainx,trainy)
trainy.mean()
trainacc=lr.score(trainx,trainy)
testacc=lr.score(testx,testy)
predy=lr.predict(testx) 
print('precision={}'.format(sum( predy*testy==1)/sum(predy==1)))


print('trainacc={},testacc={}'.format(trainacc,testacc))

tshape=x[0].shape
for i in range(len(x)):
    if x[i].shape!=tshape:
        print(i)
        break


pd.read_csv('C:/Users/Administrator/Desktop/ecom/finandata/sz_1min.csv')


trainy
describe(spyret_up)
describe(spyret_down)

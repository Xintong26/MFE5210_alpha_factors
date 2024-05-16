#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime




# 剔除交易数据少于10天，st状态和停止交易状态的股票数据
def sfilter(data):
    close = pd.read_pickle(r"E:\2023Spring\Algorithmic Trading\factors\data\adj_close.pk")
    halt_status = pd.read_pickle(r"E:\2023Spring\Algorithmic Trading\factors\data\halt_status.pk")
    st_status = pd.read_pickle(r"E:\2023Spring\Algorithmic Trading\factors\data\st_status.pk")
    data_filter = data[(close.rolling(10).count() >= 10) & (st_status == False) & (halt_status == False)]
    return data_filter




def factor_process(factor_data):
    factor_processed = sfilter(factor_data)
    factor_rank = factor_processed.rank(axis=1)
    
    return factor_rank




def processing(data):
    #去极值
    lower = data.mean(axis = 1) - 3*data.std(axis = 1)
    upper = data.mean(axis = 1) + 3*data.std(axis = 1)
    
    data.clip(lower, upper, axis = 0, inplace = True)
    
    #标准化
    data_processed = data.sub(data.mean(axis = 1),axis = 0).div(data.std(axis = 1), axis = 0)
    
    return data_processed




def factor_neutralize(factor_input,x_matrix,n_stock):
    
    factor_neutralized = pd.DataFrame().reindex_like(factor_input).T

    for date in factor_input.index:
        factor_date = factor_input.T[date].dropna()
        date_num = list(factor_input.index).index(date)
        date_matrix = pd.DataFrame(x_matrix[:,date_num,:],columns = n_stock)[factor_date.index].T
        date_matrix.dropna(inplace = True)
        model = np.linalg.lstsq(date_matrix, factor_date[date_matrix.index],rcond = None)
        factor_neutralized[date] = factor_date - date_matrix@model[0]
    
    factor_neutral = factor_neutralized.T.dropna(axis = 1, how = 'all')
    return factor_neutral



class group_test:
    def __init__(self,factor_data):
        self.factor = factor_process(factor_data)
        self.close = pd.read_pickle(r"E:\2023Spring\Algorithmic Trading\factors\data\adj_close.pk")
        self.open = pd.read_pickle(r"E:\2023Spring\Algorithmic Trading\factors\data\adj_open.pk")

    
    def get_jump(self):
        jump =  sfilter(self.open/self.close.shift(1) - 1)  #隔夜收益率
        return jump
    
    def get_intra(self):
        intra = sfilter(self.close/self.open - 1) #日内收益率
        return intra


    def group(self,group_nums):

        jump = sfilter(self.open/self.close.shift(1) - 1) #隔夜收益率
        intra = sfilter(self.close/self.open - 1) #日内收益率
        
        # 根据因子值对于每一期进行分组
        groups_num = np.ceil(self.factor.rank(axis=1,pct=True)/(1/group_nums))
        self.groups_num = groups_num

        #计算每一组的权重矩阵
        weight_dict = {gnum: (groups_num == gnum).div((groups_num == gnum).sum(axis = 1),axis = 0) for gnum in range(1,group_nums+1)}
        self.weight_dict = weight_dict

        #计算分组收益率
        group_yield = {gnum: (weight_dict[gnum].shift(2,fill_value = 0)*jump + weight_dict[gnum].shift(1)*intra).sum(axis =1,min_count = 1) for gnum in range(1,group_nums+1)}
        group_ret = pd.DataFrame(group_yield)
        self.group_ret = group_ret

        #计算每一期分组换手率
        group_tor = pd.DataFrame({gnum: weight_dict[gnum].fillna(0).diff().abs().sum(axis = 1) for gnum in range(1,group_nums+1)})
        self.group_tor = group_tor
        
        # 计算每一期股票池中股票的平均收益
        weight_mk = (~np.isnan(groups_num)).div((~np.isnan(groups_num)).sum(axis = 1), axis = 0) #每一期股票池权重
        mk_ret = (weight_mk.shift(2,fill_value = 0)*jump + weight_mk.shift(1)*intra).sum(axis=1,min_count = 1) #每一期股票池收益
        self.mk_ret = mk_ret
        
        # 超额收益,gross_PnL,net_PnL
        Exceed_ret = group_ret.sub(mk_ret, axis = 0).dropna()
        Exceed_ret_net = (Exceed_ret - group_tor*0.0012).dropna()
        self.Exceed_ret = Exceed_ret
        self.Exceed_ret_net = Exceed_ret_net




# 超额收益图
def plot(return_data):
        
        # 生成中间日期
        middle_dates = pd.date_range(start=return_data.index[0], end=return_data.index[-1], freq='YS')
        # 设置x轴标签显示第一天、最后一天和中间日期
        xticks = [return_data.index[0]] + list(middle_dates)[:-1] + [return_data.index[-1]]

        return_data.plot(figsize=(12,8), title = "Exceed Gross cumPnL", xticks = xticks)
        plt.grid(axis='y', linestyle='--')
        plt.show()




def outcome(gross_return, net_return, turnover):
    
    ret_gross = gross_return.mean()*244
    std_gross = np.std(gross_return)*np.sqrt(244)
    gross_SR = ret_gross/std_gross
    
    ret_net = net_return.mean()*244
    std_net = np.std(net_return)*np.sqrt(244)
    net_SR = ret_net/std_net 
    
    turnover_mean = turnover.mean()
    
    df = pd.DataFrame([ret_gross,gross_SR,ret_net,net_SR,turnover_mean], index = ['return','Sharp','return_net','Sharp_net','turnover']).round(3).T
    
    return df




def Lgroup_plot(Exceed_ret_data, Exceed_ret_net_data,turnover):
        
        largest_group = Exceed_ret_net_data.cumsum().iloc[-1].idxmax()
        plt.plot(Exceed_ret_data.cumsum()[largest_group])
        plt.plot(Exceed_ret_net_data.cumsum()[largest_group])
        plt.legend(["gross","net"])
        plt.title("Group" + str(largest_group))
        plt.show()
    
        outcomeL_year = Exceed_ret_data[largest_group].groupby(Exceed_ret_data.index.year).apply(lambda x: outcome(x,Exceed_ret_net_data[largest_group].loc[x.index], turnover[largest_group].loc[x.index])).reset_index(level=1,drop = True)                                                           
        
        return outcomeL_year
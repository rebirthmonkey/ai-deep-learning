# 小额贷款还款风险预测

## 介绍
通过本比赛，希望参加者能够在一个小时之内写出基本可用的机器学习流程代码（数据读取分析+特征工程+模型算法+结果输出），以此来培养参加者在真实比赛当中快速建立baseline的能力

## 数据集
* 训练集：dataset/credit-risk/train.csv
* 测试集：dataset/credit-risk/test.csv
* 答案：dataset/credit-risk/answer.csv

## 数据说明
* ID	Unique ID of each client
* LIMIT_BAL	Amount of given credit (USD dollars):  It includes both the individual consumer credit and his/her family (supplementary) credit 
* SEX	Gender (1=male, 2=female)
* EDUCATION	(1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* MARRIAGE	Marital status (1=married, 2=single, 3=divorced)
* AGE	Age of the client
* PAY_1	Repayment status in September, 2005 (-2, -1, 0=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
* PAY_2	Repayment status in August, 2005 (scale same as above)
* PAY_3	Repayment status in July, 2005 (scale same as above)
* PAY_4	Repayment status in June, 2005 (scale same as above)
* PAY_5	Repayment status in May, 2005 (scale same as above)
* PAY_6	Repayment status in April, 2005 (scale same as above)
* BILL_AMT1	Amount of bill statement in September, 2005 (USD dollar)
* BILL_AMT2	Amount of bill statement in August, 2005 (USD dollar)
* BILL_AMT3	Amount of bill statement in July, 2005 (USD dollar)
* BILL_AMT4	Amount of bill statement in June, 2005 (USD dollar)
* BILL_AMT5	Amount of bill statement in May, 2005 (USD dollar)
* BILL_AMT6	Amount of bill statement in April, 2005 (USD dollar)
* PAY_AMT1	Amount of previous payment in September, 2005 (USD dollar)
* PAY_AMT2	Amount of previous payment in August, 2005 (USD dollar)
* PAY_AMT3	Amount of previous payment in July, 2005 (USD dollar)
* PAY_AMT4	Amount of previous payment in June, 2005 (USD dollar)
* PAY_AMT5	Amount of previous payment in May, 2005 (USD dollar)
* PAY_AMT6	Amount of previous payment in April, 2005 (USD dollar)

## 预测目标
* target 预测目标，即用户下一个月（October 2005）是否会延期还款（1=违约，0=正常还款）
* 实际比赛和业务中，一般不直接预测事实（即会不会违约），对每一个sample，常见的业务要求是给出风险评分
* 本次比赛当中要求对每个sample给出风险评分（0-1 分数越高风险越高）

## 提交要求
* 提交答案要求：参考sample_submission.csv，对每个sample预测其还款概率(0-1)。提交文件中应包括ID，target两列，ID值和test.csv文件一一对应，target值为0-1之间的小数
    
* 评价标准：AUC

* 总验证集：6000条数据

* public board：3000条

* private board：3000条
  
  ​        
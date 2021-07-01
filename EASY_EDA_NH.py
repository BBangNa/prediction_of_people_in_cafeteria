#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 필요한 모듈 임포트하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# 0. 데이터 불러오기

train_raw = pd.read_csv('./datasets/train.csv', encoding='utf-8')
test_raw = pd.read_csv('./datasets/test.csv', encoding='utf-8')
submission = pd.read_csv('./datasets/sample_submission.csv', encoding='utf-8')

print(train_raw.head(2))
print(test_raw.head(2))
print('train shape :', train_raw.shape, '\n', 'test.shape', test_raw.shape)


# In[3]:


# 1. 데이터 전처리
""" 칼럼명 수정하기 """
train = train_raw.copy()
train.columns = ['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'brk', 'ln', 'dn', 'target_ln', 'target_dn']
print(train.head(2))

test = test_raw.copy()
test.columns = ['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'brk', 'ln', 'dn']
print(test.head(2))


# In[4]:


# 1-1 날짜와 요일
def to_datetime(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['dow'] = pd.to_datetime(df[date]).dt.weekday + 1
to_datetime(train, 'date');to_datetime(test,'date')

print(train.head(2))
print(test.head(2))
print(train.info())
print(test.info())


# In[5]:


# 1-2 메뉴명
# 일별 점심메뉴를 작은 리스트로 갖는 리스트 만들기 -> [[일별 점심메뉴],[일별 점심메뉴]]
lunch = []
for day in range(len(train)):
    tmp = train.iloc[day, 8].split(' ') # 문자열 공백으로 구분하기
    tmp = ' '.join(tmp).split()  # 빈 원소는 삭제하기
    search = '(' #원산지 정보 삭제하기
    for menu in tmp:
        if search in menu:
            tmp.remove(menu)

    lunch.append(tmp)

print(lunch[0:5])
print(lunch[1065:1070])
print(np.array(train[ (train.index > 1064) & (train.index < 1069)][['date', 'ln']]))

# 2020-06-13 ~ 2020-06-30 까지는 구내식당 영업을 안했음. 또 메뉴 순서가 바뀐다.


# In[6]:


# 메뉴는 밥, 국, 반찬, 김치, 사이드로 구분함
# lunch train data에 메뉴명별 칼럼 만들기
bob = []
gook = []
banchan1 = []
banchan2 = []
banchan3 = []
kimchi = []
side = []
for i, day_menu in enumerate(lunch):
    bob_tmp = day_menu[0]
    bob.append(bob_tmp)
    gook_tmp = day_menu[1]
    gook.append(gook_tmp)
    banchan1_tmp = day_menu[2]
    banchan1.append(banchan1_tmp)
    banchan2_tmp = day_menu[3]
    banchan2.append(banchan2_tmp)
    banchan3_tmp = day_menu[4]
    banchan3.append(banchan3_tmp)

    if i < 1067:
        kimchi_tmp = day_menu[-1]
        kimchi.append(kimchi_tmp)
        side_tmp = day_menu[-2]
        side.append(side_tmp)
    else:
        kimchi_tmp = day_menu[-2]
        kimchi.append(kimchi_tmp)
        side_tmp = day_menu[-1]
        side.append(side_tmp)

train_ln = train[['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'ln', 'target_ln']]
train_ln['bob'] = bob
train_ln['gook'] = gook
train_ln['banchan1'] = banchan1
train_ln['banchan2'] = banchan2
train_ln['banchan3'] = banchan3
train_ln['kimchi'] = kimchi
train_ln['side'] = side

print(train_ln.iloc[1066:1070, 7:])


# In[8]:


# 국 종류 확인하기
gook_df = pd.DataFrame(train_ln['gook'].value_counts().reset_index())
print(gook_df.head(10)) 
print(gook_df.tail(5))
print(gook_df.gook.describe())
# 서로 다른 국 메뉴는 272가지
# 같은 국은 최대 46번 나옴


# In[9]:


# 반찬1,2,3
print(train_ln['banchan1'][0:3])


# In[10]:


banchan_list = []
for i in range(3):
    tmp = train_ln[f'banchan{i+1}']
    for j in range(len(train_ln)):
        tmp2 = tmp[j]
        banchan_list.append(tmp2)

banchan_df = pd.DataFrame(pd.DataFrame(banchan_list).value_counts())
banchan_df.columns = ['banchan']
banchan_df.reset_index(inplace=True)
banchan_df.columns = ['index', 'banchan']

print(banchan_df.head(10))
print(banchan_df.tail(4))
print(banchan_df.banchan.describe()) # 서로 다른 반찬 종류 1174가지, 같은 반찬은 최대 35번 나옴.


# In[11]:


# 2. 시각화

print(train.head(2))


# In[12]:


# 2-1 점심 및 저녁 이용자 수
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 4))
sns.distplot(train["target_ln"], ax = ax[0], color = 'orange', kde = False, rug = True)
sns.distplot(train["target_dn"], ax = ax[1], color = 'green', kde = False, rug = True)
plt.show()


# In[ ]:


# 점심 이용자 수는 200-1600명 가량 이용하는 것으로 보인다.
# 저녁 이용자 수는 100-800명 정도 이용 한다. 하지만 0명에 값이 높은 것을 확인 가능함.


# In[13]:


# 2-2 코로나
# 시계열 시각화
train.plot(x = 'date', y = ['target_ln', 'target_dn'], figsize = (40, 5))
plt.show()
# 2020년 코로나 이후 구내식당에서 식사를 많이 함..왜?


# In[14]:


before_covid = train[train['date'].dt.year == 2019][['date', 'target_ln', 'target_dn']]
before_covid.plot(x = 'date', y = ['target_ln', 'target_dn'], figsize = (30, 5), grid = True)
plt.title('Before Covid19', fontsize = 25)
plt.show()


# In[15]:


after_covid = train[train['date'].dt.year >= 2020][['date', 'target_ln', 'target_dn']]
after_covid.plot(x = 'date', y = ['target_ln', 'target_dn'], figsize = (30, 5), grid = True)
plt.title('After Covid19', fontsize = 25)
plt.show()


# ### 점심, 저녁 이용자 수 평균

# In[16]:


print('점심:', '2019년에는', round(before_covid.target_ln.mean(), 2), ', 2020년에는', round(after_covid.target_ln.mean(), 2))
print('저녁:', '2019년에는', round(before_covid.target_dn.mean(), 2), ', 2020년에는', round(after_covid.target_dn.mean(), 2))


# In[17]:


train.plot(x = 'date', y = 'employees', figsize = (40, 8), c = "#e74c3c")
plt.title("Number of Employees", fontsize = 20)
plt.show()


# 점심 이용자수가 늘어난 이유가 직원 수가 늘어나서 그런 것은 아님

# In[18]:


train[train.target_dn == 0][['date',  'dayoff', 'bustrip', 'ovtime', 'remote', 'dow', 'dn', 'target_dn']]


# In[19]:


# 2-3 변수 간 상관관계

print(train.head(2))


# In[20]:


df = train[['target_ln', 'target_dn', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote']]
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
plt.rcParams['font.size'] = 15

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(df.corr(), 
            annot=True, 
            cmap="BrBG", 
            mask = mask)
ax.set_title('Correlation Heatmap', pad = 10)
plt.show()


# 점심의 경우 시간 외 근무자(ovtime)과 상관관계가 높다는 점.(양의 관계)
# 출장자 수(bustrip)는 적을 수록 이용자 수가 많아진다는 점.(음의 관계)
# 
# 저녁의 경우 시간 외 근무자의 상관관계가 더 높아짐을 알 수 있다.
# 당연한 결과, 야근하는 사람일수록 식당 이용률이 높을 것.

# In[21]:


# 변수별 산점도
fig, ax = plt.subplots(figsize = (18, 8), ncols = 4, nrows = 2, sharey=True)
plt.rcParams['font.size'] = 12
sns.color_palette("Paired")
train_features = ['employees', 'dayoff', 'bustrip', 'ovtime', 'employees', 'dayoff', 'bustrip', 'ovtime']
for i, feature in enumerate(train_features):
    row = int(i/4)
    col = i%4 
    if i < 4:
        sns.regplot(x=feature, y = 'target_ln', data = train, ax = ax[row][col], color = 'salmon', marker = '+')
    else: 
        sns.regplot(x=feature, y = 'target_dn', data = train, ax = ax[row][col], color = 'skyblue', marker = '+')


# In[22]:


# 2-4 월별, 요일별 패턴
# 2-4-1 점심 및 저녁 이용자 수

print(train.head(3))


# In[23]:


# Heatmap
# 월별 & 요일별 점심과 저녁 이용자 수의 평균을 시각화해봄.
tmp = train[['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'target_ln', 'target_dn']]
tmp['month'] = tmp['date'].dt.strftime("%m")

tmp_ln = tmp.groupby(['dow', 'month'])['target_ln'].mean().reset_index().pivot('dow', 'month', 'target_ln')
tmp_dn = tmp.groupby(['dow', 'month'])['target_dn'].mean().reset_index().pivot('dow', 'month', 'target_dn')


# In[24]:


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 4))

sns.heatmap(tmp_ln, cmap='RdYlGn_r', ax=ax[0])
ax[0].set_title('Lunch', pad = 12)
sns.heatmap(tmp_dn, cmap='RdYlGn_r', ax=ax[1])
ax[1].set_title('Dinner', pad = 12)

plt.show()


# 특히 월요일 이용자 수가 많다.

# In[25]:


# 자기개발의 날 삭제 
idx = train[train.target_dn == 0].index
tmp = train.drop(idx)
tmp['month'] = tmp['date'].dt.strftime("%m")
tmp_dn2 = tmp.groupby(['dow', 'month'])['target_dn'].mean().reset_index().pivot('dow', 'month', 'target_dn')

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 4))

sns.heatmap(tmp_ln, cmap='RdYlGn_r', ax=ax[0])
ax[0].set_title('Lunch', pad = 12)
sns.heatmap(tmp_dn2, cmap='RdYlGn_r', ax=ax[1])
ax[1].set_title('Dinner - NEW', pad = 12)

plt.show()


# In[26]:


# 2-4-2 회사에 있는 직원 수
before = train['date'].dt.year < 2020
after = train['date'].dt.year >= 2020

train[before]['remote'].value_counts()  # 코로나19 전 재택근무자수 


# In[27]:


before = train['date'].dt.year < 2020
after = train['date'].dt.year >= 2020

def heatmap_viz(df): 
    df['month'] = df['date'].dt.strftime("%m")
    before = df['date'].dt.year < 2020
    after = df['date'].dt.year >= 2020

    tmp_dayoff = df.groupby(['dow', 'month'])['dayoff'].mean().reset_index().pivot('dow', 'month', 'dayoff')
    tmp_bustrip = df.groupby(['dow', 'month'])['bustrip'].mean().reset_index().pivot('dow', 'month', 'bustrip')
    tmp_ovtime = df.groupby(['dow', 'month'])['ovtime'].mean().reset_index().pivot('dow', 'month', 'ovtime')
    tmp_remote_after = df[after].groupby(['dow', 'month'])['remote'].mean().reset_index().pivot('dow', 'month', 'remote')

    fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (30, 5), sharey = True)

    sns.heatmap(tmp_dayoff, cmap='Oranges', ax=ax[0])   #1 
    ax[0].set_title('Dayoff', pad = 12)
    sns.heatmap(tmp_bustrip, cmap='Greens', ax=ax[1])   #2 
    ax[1].set_title('Business Trip', pad = 12)
    sns.heatmap(tmp_ovtime, cmap='Blues', ax=ax[2])   #3
    ax[2].set_title('Overtime', pad = 12)
    sns.heatmap(tmp_remote_after, cmap='Purples', ax=ax[3])   # 4
    ax[3].set_title('Remote (2020-21 only)', pad = 12)
   
    plt.show()

df = train[['date', 'dow', 'dayoff', 'bustrip', 'ovtime', 'remote']]
heatmap_viz(df)


# In[28]:


df = train[['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'target_ln', 'target_dn']]
df['in_office'] = df['employees'] - (df['dayoff'] + df['bustrip'] + df['remote'])
df['month'] = df['date'].dt.strftime("%m")
df.head(3)


# In[29]:


tmp = df.groupby(['dow', 'month'])['in_office'].mean().reset_index().pivot('dow', 'month', 'in_office')

# Heatmap
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 4))

sns.heatmap(tmp, cmap='YlGnBu', ax = ax[0])    # 1 
ax[0].set_title('No. of Employees in Office', pad = 12)

df_corr = df[['target_ln', 'target_dn', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'in_office']]   # 2
mask = np.triu(np.ones_like(df_corr.corr(), dtype=np.bool))
sns.heatmap(df_corr.corr(), 
            annot=True, 
            cmap="BrBG", 
            mask = mask, 
            ax =ax[1])
ax[1].set_title('Correlation Heatmap ("in_office" added)', pad = 10)
plt.show()


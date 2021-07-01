
# 필요한 모듈 임포트하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
plt.rcParams['font.size'] = 15

import warnings
warnings.filterwarnings(action='ignore')

# 0. 데이터 불러오기

train_raw = pd.read_csv('./datasets/train.csv', encoding='utf-8')
test_raw = pd.read_csv('./datasets/test.csv', encoding='utf-8')
submission = pd.read_csv('./datasets/sample_submission.csv', encoding='utf-8')

# print(train_raw.head(2))
# print(test_raw.head(2))
# print('train shape :', train_raw.shape, '\n', 'test.shape', test_raw.shape)

# 1. 데이터 전처리
""" 칼럼명 수정하기 """
train = train_raw.copy()
train.columns = ['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'brk', 'ln', 'dn', 'target_ln', 'target_dn']
#print(train.head(2))

test = test_raw.copy()
test.columns = ['date', 'dow', 'employees', 'dayoff', 'bustrip', 'ovtime', 'remote', 'brk', 'ln', 'dn']
#print(test.head(2))

# 1-1 날짜와 요일
def to_datetime(df, date):
    df['date'] = pd.to_datetime(df[date])
    df['dow'] = pd.to_datetime(df[date]).dt.weekday + 1
to_datetime(train, 'date');to_datetime(test,'date')

# print(train.head(2))
# print(test.head(2))
# print(train.info())
# print(test.info())

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

#print(lunch[0:5])
"""
[['쌀밥/잡곡밥', '오징어찌개', '쇠불고기', '계란찜', '청포묵무침', '요구르트', '포기김치'], ['쌀밥/잡곡밥', '김치찌개', '가자미튀김', '모둠소세지구이', '마늘쫑무침', '요구르트', '배추겉절이'],
 ['카레덮밥', '팽이장국', '치킨핑거', '쫄면야채무침', '견과류조림', '요구르트', '포기김치'], ['쌀밥/잡곡밥', '쇠고기무국', '주꾸미볶음', '부추전', '시금치나물', '요구르트', '포기김치'], 
 ['쌀밥/잡곡밥', '떡국', '돈육씨앗강정', '우엉잡채', '청경채무침', '요구르트', '포기김치']]
"""
#print(lunch[1065:1070])
"""
[['쌀밥/잡곡밥', '매운소고기국', '굴비구이', '토마토프리타타', '도라지오이무침', '배추겉절이'], ['돈육버섯고추장덮밥', '팽이무국', '양파링카레튀김', '모듬어묵볶음', '참나물생채', '요구르트', '포기김치'], 
['쌀밥/잡곡밥', '냉모밀국수', '매운돈갈비찜', '메밀전병*간장', '고구마순볶음', '포기김치', '양상추샐러드*딸기요거트'], ['쌀밥/잡곡밥', '대파육개장', '홍어미나리초무침', '어묵잡채', '콩자반', '배추겉절이', '양상추샐러드*오리엔탈'],
['카레라이스', '동태알탕', '부추고추전*간장', '쫄면야채무침', '과일요거트샐러드', '포기김치', '요구르트']]
"""

""" 밥/국/반찬/사이드/김치 -> 밥/국/반찬/김치/사이드 이를 고려해 메뉴명 추출 """
#print(np.array(train[ (train.index > 1064) & (train.index < 1069)][['date', 'ln']]))
"""
[[Timestamp('2020-06-11 00:00:00')
  '쌀밥/잡곡밥 (쌀,현미,흑미:국내산) 매운소고기국  굴비구이  토마토프리타타  도라지오이무침  배추겉절이 (배추국내,고추가루:중국산) ']
 [Timestamp('2020-06-12 00:00:00')
  '돈육버섯고추장덮밥 (쌀,돈육:국내산) 팽이무국  양파링카레튀김  모듬어묵볶음  참나물생채 요구르트 포기김치 (김치:국내산) ']
 [Timestamp('2020-07-01 00:00:00')
  '쌀밥/잡곡밥 냉모밀국수 매운돈갈비찜 메밀전병*간장 고구마순볶음 포기김치 양상추샐러드*딸기요거트 ']
 [Timestamp('2020-07-02 00:00:00')
  '쌀밥/잡곡밥 대파육개장 홍어미나리초무침 어묵잡채 콩자반 배추겉절이 양상추샐러드*오리엔탈 ']]
"""
# 2020-06-13 ~ 2020-06-30 까지는 구내식당 영업을 안했음. 또 메뉴 순서가 바뀐다.
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

#print(train_ln.iloc[1066:1070, 7:])
"""
                                                     ln  ...          side
1066  돈육버섯고추장덮밥 (쌀,돈육:국내산) 팽이무국  양파링카레튀김  모듬어묵볶음  참나...  ...          요구르트
1067  쌀밥/잡곡밥 냉모밀국수 매운돈갈비찜 메밀전병*간장 고구마순볶음 포기김치 양상추샐러드...  ...  양상추샐러드*딸기요거트
1068  쌀밥/잡곡밥 대파육개장 홍어미나리초무침 어묵잡채 콩자반 배추겉절이 양상추샐러드*오리엔탈   ...   양상추샐러드*오리엔탈
1069     카레라이스 동태알탕 부추고추전*간장 쫄면야채무침 과일요거트샐러드 포기김치 요구르트   ...          요구르트
"""

# 국 종류 확인하기
gook_df = pd.DataFrame(train_ln['gook'].value_counts().reset_index())
# print(gook_df.head(10))
"""
   index  gook
0    맑은국    46
1   콩나물국    44
2   된장찌개    37
3    어묵국    31
4  배추된장국    28
5  가쯔오장국    28
6    아욱국    28
7    근대국    26
8    꽃게탕    25
9  순두부찌개    23
"""
# print(gook_df.tail(5))
"""
        index  gook
267  통계란꼬치어묵탕     1
268    매운계란파국     1
269    쇠고기매운국     1
270     들깨시락국     1
271      순두부탕     1
"""
# print(gook_df.gook.describe())
"""
count    272.000000  서로 다른 국 메뉴는 272가지.
mean       4.430147
std        7.022545
min        1.000000
25%        1.000000
50%        1.000000
75%        5.000000
max       46.000000  같은 국은 최대 46번 나옴.
Name: gook, dtype: float64
"""
# 반찬1,2,3
# print(train_ln['banchan1'][0:3])
"""
0     쇠불고기
1    가자미튀김
2     치킨핑거
"""
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

# print(banchan_df.head(10))
# print(banchan_df.tail(4))
# print(banchan_df.banchan.describe())
"""
     index  banchan
0     오이무침       35
1    오징어볶음       32
2      닭갈비       30
3    버섯불고기       29
4    콩나물무침       28
5     계란말이       28
6   훈제오리구이       27
7  돈육굴소스볶음       25
8      계란찜       24
9     숙주나물       24

1170    부들어묵볶음        1
1171   부추고추장무침        1
1172  부추고추전*간장        1
1173   히레카츠*소스        1

count    1174.000000  서로 다른 반찬 종류 1174가지
mean        3.079216
std         4.371070
min         1.000000
25%         1.000000
50%         1.000000
75%         3.000000
max        35.000000  같은 반찬은 최대 35번 나옴.
Name: banchan, dtype: float64
"""

# 2. 시각화

# print(train.head(2))
"""
        date  dow  ...  target_ln  target_dn
0 2016-02-01    1  ...     1039.0      331.0
1 2016-02-02    2  ...      867.0      560.0
"""
# 2-1 점심 및 저녁 이용자 수
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 4))
sns.distplot(train["target_ln"], ax = ax[0], color = 'orange', kde = False, rug = True)
sns.distplot(train["target_dn"], ax = ax[1], color = 'green', kde = False, rug = True)
# plt.show()

# 점심 이용자 수는 200-1600명 가량 이용하는 것으로 보인다.
# 저녁 이용자 수는 100-800명 정도 이용 한다. 하지만 0명에 값이 높은 것을 확인 가능함.

# 2-2 코로나
# 시계열 시각화
train.plot(x = 'date', y = ['target_ln', 'target_dn'], figsize = (40, 5))
plt.show()
















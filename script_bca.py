# %%
#import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score,mean_absolute_error

# %%
#import dataset
df = pd.read_csv(r'D:\skripsi\BBCA\BBCA.csv')
BBCA = df.head()
BBCA.to_csv("Dataset_BBCA.csv",index=False)

# %%
saham = df[['Date','Open', 'High', 'Low', 'Close']]
saham_bbca = saham.head()
saham_bbca.to_csv("Data_BBCA.csv")

# %%
saham.describe()

# %%
saham.info()

# %%
var_diperlukan = saham[['Open','High','Low', 'Close']]
sns.heatmap(var_diperlukan.corr(), cmap='coolwarm', annot=True)
plt.savefig('heatmap.png')  
plt.show() 
plt.figure(figsize=(3, 3))  

# %%
plt.figure(figsize=(5,5))
sns.boxplot(data=saham)
plt.savefig('boxplot.png') 
plt.show()

# %%

# Menghitung kuartil dan IQR
q1 = np.percentile(var_diperlukan, 25)
q2 = np.percentile(var_diperlukan, 50  )
q3 = np.percentile(var_diperlukan, 75)
iqr = q3 - q1

# Menghitung batas bawah dan atas (whiskers)
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

#Mendeteksi outlier
outliers = var_diperlukan[(var_diperlukan < lower_whisker) | (var_diperlukan > upper_whisker)]
# Membuat boxplot
plt.boxplot(var_diperlukan, labels=['Open', 'High', 'Low', 'Close'], medianprops=dict(color="black"))


# Menampilkan boxplot
plt.show()

# %%
X = saham[['Open', 'High', 'Low']].to_numpy()
Y = saham[['Close']].to_numpy()
date = saham['Date'].to_numpy()
                        
#Splitiing data
X_train,X_test,Y_train,Y_test,date_train,date_test= train_test_split(X,Y,date,test_size=0.2,random_state=42)
print(f'shape data X train {X_train.shape}')
print(f'shape data Y train {Y_train.shape}')
print(f'shape data X test {X_test.shape}')
print(f'shape data Y test {Y_test.shape}')

# %%
print(X.shape)
print(Y.shape)

# %%
X_train

# %%
X_test

# %% [markdown]
# Model KNN regresi



# %%
#Mencari K optimal menggunakan elbow method dengan metric RMSE

rmse_list = []#membuat list kosong untuk nantinya hasil perhitungan metric RMSE disimpan

for k in range(1, 11):

    # Membuat prediksi menggunakan fungsi knn_regression
    Y_pred = knn_regression(X_train, Y_train, X_test, k) # memanggil fungsi untuk knn regresi yang telah dibuat

    # Menghitung MSE
    mse = mean_squared_error(Y_test, Y_pred) #melakukan perhitungan MSE antara data aktual dan data prediksi
    #Menghitung RMSE
    rmse = np.sqrt(mse)
    # Menyimpan RMSE dalam list
    rmse_list.append((k, rmse))
#Membuat grafik elbow method

k_values, rmse_values = zip(*rmse_list) #proses ini disebut unzipping atau unpacking untuk mengambil elemen pertama (k) dan kedua(rmse) pada tuple
plt.plot(k_values,rmse_values, marker='o',color='black',linewidth=0.7)
plt.title('Elbow Method untuk Optimal k')
plt.xlabel('jumlah tetangga (k)')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.savefig('Elbow.png')
plt.show()

# %%
# cetak RMSE
for k, rmse in rmse_list :
    print(f'RMSE untuk k={k} adalah {rmse}')

# %%
# Mengurutkan berdasarkan nilai RMSE terkecil
sorted_rmse_list = sorted(rmse_list,key=lambda x: x[1])
for k, rmse in sorted_rmse_list:
    print(f'RMSE untuk k={k} adalah {rmse}')
    

# %% [markdown]
# **Melakukan prediksi**

# %%
# memprediksi data test dengan mengganti nilai k dangn k optimal yang diperoleh
y_pred = knn_regression(X_train, Y_train, X_test, 4)
#Membuat kedalam data frame yang berisi date, Y_test dan Y_pred
hasil_KNN = pd.DataFrame({'Tanggal': [item for item in date_test],'Aktual': [item[0] for item in Y_test],'Prediksi' : y_pred })
#hasil_KNN
hasil_KNN.to_csv("Hasil_Prediksi_BBCA.csv")
#metric.to_csv('metrics.csv',index=False)

# %%
#Melihat nilai r2_score, MAPE, dan RMSE dari hasil prediksi KNN Regresi

#Melihat r2_score KNN
scr_KNN = r2_score(Y_test,y_pred)

#Melihat MAPE KNN
mape_KNN = mean_absolute_percentage_error(Y_test, y_pred)

#Melihat MAE KNN
mse_KNN = mean_squared_error(Y_test,y_pred)
rmse_KNN= np.sqrt(mse_KNN)
mae_KNN = mean_absolute_error(Y_test,y_pred)
#melihat mse KNN
for k, rmse in rmse_list:
   if k == 4 :
      rmse_KNN=rmse # mengambil RMSE untuk K = 4 yang telah dihitung

#Melihat hasilnya dalam dataframe
metric = pd.DataFrame({'Metric' : ['R2_score','MAPE Score','RMSE Score','MAE Score'], 'Score' : [scr_KNN, mape_KNN, rmse_KNN,mae_KNN]})
#metric.to_csv('metrics.csv',index=False)
metric

# %%
#MEmbuat diagram garis antara prediksi dan nilai aktual;

indeks = np.arange(len(y_pred))
# Membuat diagram garis
plt.figure(figsize=(15, 5))
plt.plot(indeks, y_pred, label='Prediksi', linestyle='-', color='blue')
plt.plot(indeks, Y_test, label='Aktual', linestyle='-', color='green',)
# Menambahkan label dan judul
plt.title('Diagram Garis Nilai Prediksi vs Nilai Aktual KNN Regresi')
plt.ylabel('Nilai Close')
plt.legend()

# Menampilkan grafik
plt.grid(True)
plt.savefig('aktualvsprediksi.png')
plt.show()

# %% [markdown]
# ARA dan ARB

# %%
saham['perubahan'] = ((saham['Close'] - saham['Open']) / saham['Open'])*100

saham['perubahan_poisitif'] = saham['perubahan'].apply(lambda x:x if x > 0 else 0)
saham['perubahan_negatif'] = saham['perubahan'].apply(lambda x: abs(x) if x < 0 else 0)

print(saham)

# %%
avg_gain = saham['perubahan_poisitif'].mean()
avg_loss = saham['perubahan_negatif'].mean()
print(avg_gain)
print(avg_loss)

# %%
data_ara = pd.read_csv(r'D:\skripsi\BBCA\BBCA_15-17.csv')

# %%
bbca_ara = data_ara[['Date','Open','Close']]

col_name = 'High'
bbca_ara[col_name] = 0  # Kolom pertama diisi dengan 0 untuk setiap baris
for j in range(1, len(bbca_ara)):  # Mulai dari baris kedua
    bbca_ara.loc[j, col_name] = bbca_ara.loc[j-1, 'Close'] + ((bbca_ara.loc[j-1, 'Close'])*(avg_gain/100))
    
col_name = 'Low'
bbca_ara[col_name] = 0  # Kolom pertama diisi dengan 0 untuk setiap baris
for j in range(1, len(bbca_ara)):  # Mulai dari baris kedua
    bbca_ara.loc[j, col_name] = bbca_ara.loc[j-1, 'Close'] - ((bbca_ara.loc[j-1, 'Close'])*(avg_loss/100))

# Menampilkan data bbca_ara dengan kolom Close yang ditambah nilai persentase
print(bbca_ara)
bbca_ara[['Date','Open','High','Low','Close']].to_csv('bbca_ara_arb.csv',index=False)


# %% [markdown]
# Perhitungan ARA dan ARB

# %%
BBCAara_arb = pd.read_csv(r'D:\skripsi\BBCA\bbca_ara_arb.csv')
BBCAara_arb1 = BBCAara_arb[['Date','Open','High','Low']]
BBCAara_arb1

# %%
bbca = BBCAara_arb[['Open', 'High', 'Low']]
bbca_seleksi = bbca.drop(0).reset_index(drop=True)
bbca_arr = np.array(bbca_seleksi)
prediksi = knn_regression(X_train, Y_train, bbca_arr,4)
hasil = pd.DataFrame({'Nilai Prediksi': prediksi, 'Nilai Aktual': BBCAara_arb['Close'].drop(0).reset_index(drop=True)})
hasil['Residual Error'] = (abs(hasil['Nilai Prediksi'] - hasil['Nilai Aktual'])/hasil['Nilai Aktual'])*100
hasil.to_csv('hasilPrediksiARAdanARBBBCA.csv', index=False)
print(hasil)
avg_re = hasil['Residual Error'].mean()
print(avg_re)

# %% [markdown]
# Prediksi Perjam

# %%
# x adalah awal data
# y adalah akhir data
def ambil_data(x,y,data_n) :
    databbca = data_n[x:y]
    return databbca

# %% [markdown]
# ***PREDIKSI TANGGAL 15***
# 

# %%
data_15 = pd.read_csv(r'D:\skripsi\BBCA\menit\data15.csv')


#prediksi tanggal 16 jam 11
print('\nPrediksi Jam 11 Tanggal 15')
prediksi2 = prediksimenit16(13,23,X_train,Y_train,data_15)
print(prediksi2)
prediksi22 = prediksimenit16(13,22,X_train,Y_train,data_15)
print(prediksi22)
prediksi222 = prediksimenit16(13,21,X_train,Y_train,data_15)

#prediksi tanggal 16 jam 13.30
print('\nPrediksi Jam 13.30 Tanggal 15')
prediksi3 = prediksimenit16(24,33,X_train,Y_train,data_15 )
print(prediksi3)
prediksi33 = prediksimenit16(24,32,X_train,Y_train,data_15 )
print(prediksi33)
prediksi333 = prediksimenit16(24,31,X_train,Y_train,data_15 )

#prediksi tanggal 16 jam 14
print('\nPrediksi Jam 14 Tanggal 15')
prediksi4 = prediksimenit16(34,38,X_train,Y_train,data_15 )
print(prediksi4)
prediksi44 = prediksimenit16(34,37,X_train,Y_train,data_15 )
print(prediksi44)
prediksi444 = prediksimenit16(34,36,X_train,Y_train,data_15 )
print(prediksi444)

# prediksi tanggal 16 jam 15
print('\nPrediksi Jam 15 Tanggal 15')
prediksi5 = prediksimenit16(39,49,X_train,Y_train,data_15)
print(prediksi5)
prediksi55 = prediksimenit16(39,48,X_train,Y_train,data_15)
print(prediksi55)
prediksi555 = prediksimenit16(39,47,X_train,Y_train,data_15)
print(prediksi555)

# prediksi tanggal 16 penutupan
print('\nPrediksi Penutupan Tanggal 15')
prediksi6 = prediksimenit16(50,58,X_train,Y_train,data_15)
print(prediksi6)
prediksi66 = prediksimenit16(50,57,X_train,Y_train,data_15)
print(prediksi66)
prediksi666 = prediksimenit16(50,56,X_train,Y_train,data_15)
print(prediksi666)

# %% [markdown]
# ***PREDIKSI TANGGAL 16***

# %%
data_15 = pd.read_csv(r'D:\skripsi\BBCA\menit\data16.csv')

#prediksi tanggal 16 jam 10
print('Prediksi Jam 10 Tanggal 16')
prediksi1 = prediksimenit16(0,12,X_train,Y_train,data_15)
print(prediksi1)
prediksi11 = prediksimenit16(0,11,X_train,Y_train,data_15)
print(prediksi11)
prediksi111 = prediksimenit16(0,10,X_train,Y_train,data_15)
print(prediksi111)

#prediksi tanggal 16 jam 11
print('\nPrediksi Jam 11 Tanggal 16')
prediksi2 = prediksimenit16(13,23,X_train,Y_train,data_15)
print(prediksi2)
prediksi22 = prediksimenit16(13,22,X_train,Y_train,data_15)
print(prediksi22)
prediksi222 = prediksimenit16(13,21,X_train,Y_train,data_15)

#prediksi tanggal 16 jam 13.30
print('\nPrediksi Jam 13.30 Tanggal 16')
prediksi3 = prediksimenit16(24,34,X_train,Y_train,data_15 )
print(prediksi3)
prediksi33 = prediksimenit16(24,33,X_train,Y_train,data_15 )
print(prediksi33)
prediksi333 = prediksimenit16(24,32,X_train,Y_train,data_15 )

#prediksi tanggal 16 jam 14
print('\nPrediksi Jam 14 Tanggal 16')
prediksi4 = prediksimenit16(35,39,X_train,Y_train,data_15 )
print(prediksi4)
prediksi44 = prediksimenit16(35,38,X_train,Y_train,data_15 )
print(prediksi44)
prediksi444 = prediksimenit16(35,37,X_train,Y_train,data_15 )
print(prediksi444)

# prediksi tanggal 16 jam 15
print('\nPrediksi Jam 15 Tanggal 16')
prediksi5 = prediksimenit16(40,50,X_train,Y_train,data_15)
print(prediksi5)
prediksi55 = prediksimenit16(40,49,X_train,Y_train,data_15)
print(prediksi55)
prediksi555 = prediksimenit16(40,48,X_train,Y_train,data_15)
print(prediksi555)

# prediksi tanggal 16 penutupan
print('\nPrediksi Penutupan Tanggal 16')
prediksi6 = prediksimenit16(51,59,X_train,Y_train,data_15)
print(prediksi6)
prediksi66 = prediksimenit16(51,58,X_train,Y_train,data_15)
print(prediksi66)
prediksi666 = prediksimenit16(51,57,X_train,Y_train,data_15)
print(prediksi666)

# %% [markdown]
# ***PREDIKSI TANGGAL 17**

# %%
data_17 = pd.read_csv(r'D:\skripsi\BBCA\menit\data17.csv')
#prediksi tanggal 17 jam 10
print('Prediksi Jam 10 Tanggal 17')
prediksi1 = prediksimenit16(0,12,X_train,Y_train,data_17)
print(prediksi1)
prediksi11 = prediksimenit16(0,11,X_train,Y_train,data_17)
print(prediksi11)
prediksi111 = prediksimenit16(0,10,X_train,Y_train,data_17)
print(prediksi111)

#prediksi tanggal 17 jam 11
print('\nPrediksi Jam 11 Tanggal 17')
prediksi2 = prediksimenit16(13,23,X_train,Y_train,data_17)
print(prediksi2)
prediksi22 = prediksimenit16(13,22,X_train,Y_train,data_17)
print(prediksi222)
prediksi222 = prediksimenit16(13,21,X_train,Y_train,data_17)
print(prediksi222)

#prediksi tanggal 17 jam 14
print('\nPrediksi Jam 14 Tanggal 17')
prediksi3 = prediksimenit16(24,27,X_train,Y_train,data_17)
print(prediksi3)
prediksi33 = prediksimenit16(24,26,X_train,Y_train,data_17)
print(prediksi33)
prediksi333 = prediksimenit16(24,25,X_train,Y_train,data_17)
print(prediksi333)

#prediksi tanggal 17 jam 15
print('\nPrediksi Jam 15 Tanggal 17')
prediksi4 = prediksimenit16(28,38,X_train,Y_train,data_17)
print(prediksi4)
prediksi44 = prediksimenit16(28,37,X_train,Y_train,data_17)
print(prediksi44)
prediksi444 = prediksimenit16(28,36,X_train,Y_train,data_17)
print(prediksi444)

# prediksi tanggal 17 jam penutupan
print('\nPrediksi penutupan Tanggal 17')
prediksi5 = prediksimenit16(39,48,X_train,Y_train,data_17)
print(prediksi5)
prediksi55 = prediksimenit16(39,47,X_train,Y_train,data_17)
print(prediksi55)
prediksi555 = prediksimenit16(39,46,X_train,Y_train,data_17)
print(prediksi555)

# %% [markdown]
# ***PREDIKSI TANGGAL 20***

# %%
data_15 = pd.read_csv(r'D:\skripsi\BBCA\menit\data_20.csv')

#prediksi tanggal 20 jam 10
print('Prediksi Jam 10 Tanggal 20')
prediksi1 = prediksimenit16(0,12,X_train,Y_train,data_15)
print(prediksi1)
prediksi11 = prediksimenit16(0,11,X_train,Y_train,data_15)
print(prediksi11)
prediksi111 = prediksimenit16(0,10,X_train,Y_train,data_15)
print(prediksi111)

#prediksi tanggal 20 jam 11
print('\nPrediksi Jam 11 Tanggal 20')
prediksi2 = prediksimenit16(13,23,X_train,Y_train,data_15)
print(prediksi2)
prediksi22 = prediksimenit16(13,22,X_train,Y_train,data_15)
print(prediksi22)
prediksi222 = prediksimenit16(13,21,X_train,Y_train,data_15)

#prediksi tanggal 20 jam 13.30
print('\nPrediksi Jam 13.30 Tanggal 20')
prediksi3 = prediksimenit16(24,34,X_train,Y_train,data_15 )
print(prediksi3)
prediksi33 = prediksimenit16(24,33,X_train,Y_train,data_15 )
print(prediksi33)
prediksi333 = prediksimenit16(24,32,X_train,Y_train,data_15 )

#prediksi tanggal 20 jam 14
print('\nPrediksi Jam 14 Tanggal 20')
prediksi4 = prediksimenit16(35,39,X_train,Y_train,data_15 )
print(prediksi4)
prediksi44 = prediksimenit16(35,38,X_train,Y_train,data_15 )
print(prediksi44)
prediksi444 = prediksimenit16(35,37,X_train,Y_train,data_15 )
print(prediksi444)

# prediksi tanggal 20 jam 15
print('\nPrediksi Jam 15 Tanggal 20')
prediksi5 = prediksimenit16(40,50,X_train,Y_train,data_15)
print(prediksi5)
prediksi55 = prediksimenit16(40,49,X_train,Y_train,data_15)
print(prediksi55)
prediksi555 = prediksimenit16(40,48,X_train,Y_train,data_15)
print(prediksi555)

# prediksi tanggal 20 penutupan
print('\nPrediksi Penutupan Tanggal 20')
prediksi6 = prediksimenit16(51,59,X_train,Y_train,data_15)
print(prediksi6)
prediksi66 = prediksimenit16(51,58,X_train,Y_train,data_15)
print(prediksi66)
prediksi666 = prediksimenit16(51,57,X_train,Y_train,data_15)
print(prediksi666)

# %% [markdown]
# ***PREDIKSI TANGGAL 21***
# 

# %%
data_15 = pd.read_csv(r'D:\skripsi\BBCA\menit\data21.csv')

#prediksi tanggal 21 jam 10
print('Prediksi Jam 10 Tanggal 21')
prediksi1 = prediksimenit16(0,12,X_train,Y_train,data_15)
print(prediksi1)
prediksi11 = prediksimenit16(0,11,X_train,Y_train,data_15)
print(prediksi11)
prediksi111 = prediksimenit16(0,10,X_train,Y_train,data_15)
print(prediksi111)


#prediksi tanggal 21 jam 11
print('\nPrediksi Jam 11 Tanggal 21')
prediksi2 = prediksimenit16(13,23,X_train,Y_train,data_15)
print(prediksi2)
prediksi22 = prediksimenit16(13,22,X_train,Y_train,data_15)
print(prediksi22)
prediksi222 = prediksimenit16(13,21,X_train,Y_train,data_15)
print(prediksi222)


#prediksi tanggal 21 jam 13.30
print('\nPrediksi Jam 13.30 Tanggal 21')
prediksi3 = prediksimenit16(24,34,X_train,Y_train,data_15 )
print(prediksi3)
prediksi33 = prediksimenit16(24,33,X_train,Y_train,data_15 )
print(prediksi33)
prediksi333 = prediksimenit16(24,32,X_train,Y_train,data_15 )
print(prediksi333)


#prediksi tanggal 21 jam 14
print('\nPrediksi Jam 14 Tanggal 21')
prediksi4 = prediksimenit16(35,39,X_train,Y_train,data_15 )
print(prediksi4)
prediksi44 = prediksimenit16(35,38,X_train,Y_train,data_15 )
print(prediksi44)
prediksi444 = prediksimenit16(35,37,X_train,Y_train,data_15 )
print(prediksi444)


# prediksi tanggal 21 jam 15
print('\nPrediksi Jam 15 Tanggal 21')
prediksi5 = prediksimenit16(40,50,X_train,Y_train,data_15)
print(prediksi5)
prediksi55 = prediksimenit16(40,49,X_train,Y_train,data_15)
print(prediksi55)
prediksi555 = prediksimenit16(40,48,X_train,Y_train,data_15)
print(prediksi555)


# prediksi tanggal 21 penutupan
print('\nPrediksi Penutupan Tanggal 21')
prediksi6 = prediksimenit16(51,59,X_train,Y_train,data_15)
print(prediksi6)
prediksi66 = prediksimenit16(51,58,X_train,Y_train,data_15)
print(prediksi66)
prediksi666 = prediksimenit16(51,57,X_train,Y_train,data_15)
print(prediksi666)

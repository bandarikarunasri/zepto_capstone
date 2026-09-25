import seaborn as sns
# ONE time load - internet kavali
df = sns.load_dataset('titanic')
print(df.shape) # 891,15 ravali
print(df.isnull().mean()*100) # Missing %

df.to_csv('analytics/titanic.csv', index=False)
print("analytics/titanic.csv DONE - 891 rows")
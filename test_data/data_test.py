from modules.utils import pd, np

df = pd.read_csv('call_daily.csv')

print(df.select_dtypes(np.number).columns)

# df['Demand'] = df['Calls'] * df['Loaded AHT']
# print(df.head())
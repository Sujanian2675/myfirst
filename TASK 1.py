import pandas as pd

fruits = pd.Series({'Apple': 10, 'Banana': 5, 'Orange': 20, 'Kiwi': 15})
print("Pandas Series:")
print(fruits)


fruits_df = fruits.reset_index()
fruits_df.columns = ['Fruit', 'Quantity']

prices = [0.5, 0.3, 1.2, 1.5] 
fruits_df['Price per Unit'] = prices

print(fruits_df)





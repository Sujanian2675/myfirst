
import pandas as pd

sales_data = {
    'Transaction_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Product_Category': ['Electronics', 'Furniture', 'Electronics', 'Furniture', 'Clothing', 'Clothing', 'Electronics', 'Clothing', 'Furniture', 'Electronics'],
    'Sales_Value': [200, 150, 300, 250, 50, 80, 400, 60, 100, 500]
}
df = pd.DataFrame(sales_data)

print("Previous DataFrame:")
print(df)

group_df = df.groupby('Product_Category')['Sales_Value'].sum().reset_index()

print("Total Sales Value by Product Category:")
print(group_df)

-----------------------------

import numpy as np

array_3x3 = np.random.randint(1, 11, size=(3, 3))
print("Previous 3x3 Array:")
print(array_3x3)

array_1D = array_3x3.reshape(-1)
print("Reshaped 1D Array:")
print(array_1D)

-------------------------------

import numpy as np

array_3x1 = np.random.randint(1, 11, size=(3, 1))
print("Original 3x1 Array:")
print(array_3x1)

array_1x3 = np.random.randint(1, 11, size=(1, 3))
print("Original 2x3 Array:")
print(array_1x3)

added_array = array_3x1 + array_1x3
print("Array after element wise addition):")
print(added_array)

multiplied_array = array_3x1 * 3
print("First array after multiplying each element by 3:")
print(multiplied_array)

----------------------------------

import numpy as np

dataset = np.random.randn(100) 
print("Random Dataset:")
print(dataset)

mean_value = np.mean(dataset)
print("Mean:")
print(mean_value)

median_value = np.median(dataset)
print("Median:")
print(median_value)

std_deviation = np.std(dataset)
print("Standard Deviation:")
print(std_deviation)

variance_value = np.var(dataset)
print("Variance:")
print(variance_value)

---------------------------------------

import numpy as np

array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Original NumPy Array:")
print(array)

subset_array = array[::2]
print("Subset of every second Array:")
print(subset_array)





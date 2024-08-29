# number = int(input("Enter a number greater than 1: "))

# while number <= 1:
#     print("Invalid input. Please enter a number greater than 1.")
#     number = int(input("Enter again: "))

# if number == 2 or number % 2 != 0:
#     print(f"{number} is a prime number.")
# elif number % 2 == 0:
#     print(f"{number} is not a prime number.")



import pandas as pd

data = {'Name': ['Atul', 'Bharat', 'Chandan', 'Durga', 'Esha'],'Age': [25, 35, 45, 20, 50],'Occupation': ['Engineer', 'Doctor', 'Artist', 'Student', 'Teacher']}
df = pd.DataFrame(data)

print("Original DataFrame: ")
print(df)

filtered_df = df[df['Age'] > 33]

print("Filtered DataFrame (Age > 33):")
print(filtered_df)

filtered_df_query = df.query('Age > 33')

print("Filtered DataFrame using query (Age > 33):")
print(filtered_df_query)



 

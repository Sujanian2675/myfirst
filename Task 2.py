
# numbers = [36,25,99,104,22,1002,15,3,44,29,27,339,468]


# largest_number = numbers[0]
# smallest_number = numbers[0]

# for number in numbers:
#     if number > largest_number:
#         largest_number = number
#     if number < smallest_number:
#         smallest_number = number

# print(f"The largest number in the list is: {largest_number}")
# print(f"The smallest number in the list is: {smallest_number}")



import pandas as pd
import numpy as np

Students = {'Name': ['Atul', 'Bharat', 'Chandan', 'Durga', 'Esha'],'Age': [20, 21, 22, 20, 21],'Grade': [85, 42, 58, 90, 47]}
students_df = pd.DataFrame(Students)

print("Previous DataFrame:")
print(students_df)

students_df['Pass/Fail'] = np.where(students_df['Grade'] >= 50, 'Pass', 'Fail')

print("DataFrame with Pass/Fail Column:")
print(students_df)


       
 
          

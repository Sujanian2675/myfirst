
a = int(input("Enter a number: "))
c = input("Enter the operation (+, -, *, /): ")
b = int(input("Enter another number: "))
    
if c == "+":
     print(a + b)
elif c == "-":
    print(a - b)
elif c == "*":
    print(a * b)
elif c == "/":
    if b != 0:
        print(a / b)
    else:
            print("Error: Division by zero is not allowed.")
else:
        print("Error: Invalid operation entered.")



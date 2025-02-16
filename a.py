# 获取用户输入的第一个数字
num1 = float(input("请输入第一个数字: "))
# 获取用户输入的第二个数字
num2 = float(input("请输入第二个数字: "))
# 获取用户输入的运算符
operator = input("请输入运算符（+、-、*、/）: ")

# 根据运算符进行相应的运算
if operator == '+':
    result = num1 + num2
    print(f"{num1} + {num2} = {result}")
elif operator == '-':
    result = num1 - num2
    print(f"{num1} - {num2} = {result}")
elif operator == '*':
    result = num1 * num2
    print(f"{num1} * {num2} = {result}")
elif operator == '/':
    # 检查除数是否为零
    if num2 == 0:
        print("错误：除数不能为零。")
    else:
        result = num1 / num2
        print(f"{num1} / {num2} = {result}")
else:
    print("错误：输入的运算符无效，请输入 +、-、* 或 /。")
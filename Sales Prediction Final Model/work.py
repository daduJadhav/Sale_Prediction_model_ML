from joblib import load

value_convertor = load('final_poly_convertor.joblib')
value_model = load('Final_Sales_Model.joblib')

# u1 = int(input("Enter TV Sales :"))
# u2 = int(input("Enter radio Sales :"))
# u3 = int(input("Enter newspaper Sales :"))

u1 = 149
u2 = 22
u3 = 12

def function1(num1,num2,num3):
    campain = [[num1,num2,num3]]
    v1 = value_convertor.fit_transform(campain)
    return value_model.predict(v1)

print(function1(u1,u2,u3))




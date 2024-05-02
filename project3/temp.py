from logicPlan import *
from logic import *

A = Expr('A')
print(findModel(A))
findModelUnderstandingCheck()

print(findModel(A).values())

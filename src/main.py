# #!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"


from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

p1 = make_pipeline(StandardScaler(), SVC(C=1))
p3 = make_pipeline(StandardScaler(), SVC(C=1))

temp = {}
temp[str(p1)] = 0.5
temp[str(p3)] = 0.8
print(len(temp))
a = str(p3)
p4 = exec(a)
if p1.steps == p3.steps:
    print('HI')

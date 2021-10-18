from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Create your views here.
def index(request) :
    if request.GET :
        dict = request.GET
        data = list(dict.values())
        try:
            test = [np.array(data, dtype='float64')]
        
            dataset = pd.read_csv(r'C:\Users\LENOVO\Desktop\heart disease prediction\predictor\data.csv')

            y = dataset['target']    
            X = dataset[['age', 'sex', 'cp', 'thalach','exang', 'oldpeak','slope']]

            svc_classifier = SVC(kernel = 'linear')
            svc_classifier.fit(X,y)

            z = svc_classifier.predict(test)

            if z[0] == 0 :
                text = "Don't worry, you are at no risk. "
            elif z[0] == 1 :
                text = 'You might have heart disease.'

            context = {
                'variable':text
            }
            return render(request, 'index.html', context)
        except:
            context = {
                'variable':"Please fill all entries"
            }
            return render(request, 'index.html', context)
    else :
        return render(request, 'index.html')


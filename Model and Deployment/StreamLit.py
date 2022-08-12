import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# the sidebar
with st.sidebar:
    selected = option_menu(
        menu_title = "Project Menu",
        options=["Modelling", "Prediction"],
    )


# Modelling stage
if selected =="Modelling":
    BP_Target = pd.read_csv("DataSet/X_TargetBP1.csv")
    FA_Target = pd.read_csv('DataSet/X_TargetFA.csv')
    FS_Target = pd.read_csv('DataSet/X_TargetFS.csv')
    selected = option_menu(
        menu_title="",
        options=["Brand Purchase", "Food Amount", "Food Shopping"],
        default_index= 0,
        orientation="horizontal"
    )
    # Food Shopping
    if selected == "Food Shopping":
        X_target1Data = FS_Target.drop(columns="FoodShopping")
        y_target1Data = FS_Target["FoodShopping"]
        X_train, X_test, Y_train, Y_test = train_test_split(X_target1Data, y_target1Data, test_size = 0.20, random_state = 42)
        SVM_ModelFS = svm.SVC()
        SVM_ModelFS.fit(X_train, Y_train)
        predicted_target1Data = SVM_ModelFS.predict(X_test)
        st.header("SVM Model")
        st.text(metrics.classification_report(Y_test, predicted_target1Data))
        #st.text(metrics.confusion_matrix(Y_test, predicted_target1Data))
        st.text("=============================================================")

        from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
        from sklearn.pipeline import Pipeline
        nca = NeighborhoodComponentsAnalysis(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, Y_train)
        predicted_target1Data1 = nca_pipe.predict(X_test)
        st.header('KNN Model')
        st.text(metrics.classification_report(Y_test, predicted_target1Data1))

        st.text("=============================================================")

        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        predicted_target1Data2 = clf.predict(X_test)
        st.header('Decision Model')
        st.text(metrics.classification_report(Y_test, predicted_target1Data2))
        st.text("=============================================================")

        from sklearn.ensemble import GradientBoostingClassifier
        clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
        predicted_target1Data3 = clf1.predict(X_test)
        st.header('Gradient Boosting Classifier')
        st.text(metrics.classification_report(Y_test, predicted_target1Data3))
        st.text("=============================================================")


        # Food Amount
    if selected == "Food Amount":
        X_target2Data = FA_Target.drop(columns="FoodAmount")
        y_target2Data = FA_Target["FoodAmount"]
        X_train, X_test, Y_train, Y_test = train_test_split(X_target2Data, y_target2Data, test_size = 0.20, random_state = 42)
        SVM_ModelFA = svm.SVC()
        SVM_ModelFA.fit(X_train, Y_train)
        predicted_target2Data = SVM_ModelFA.predict(X_test)
        st.header("SVM Model")
        st.text(metrics.classification_report(Y_test, predicted_target2Data))
        #st.text(metrics.confusion_matrix(Y_test, predicted_target1Data))
        st.text("=============================================================")

        from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
        from sklearn.pipeline import Pipeline
        nca = NeighborhoodComponentsAnalysis(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, Y_train)
        predicted_target2Data1 = nca_pipe.predict(X_test)
        st.header('KNN Model')
        st.text(metrics.classification_report(Y_test, predicted_target2Data1))

        st.text("=============================================================")

        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        predicted_target2Data2 = clf.predict(X_test)
        st.header('Decision Model')
        st.text(metrics.classification_report(Y_test, predicted_target2Data2))
        st.text("=============================================================")

        from sklearn.ensemble import GradientBoostingClassifier
        clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
        predicted_target2Data3 = clf1.predict(X_test)
        st.header('Gradient Boosting Classifier')
        st.text(metrics.classification_report(Y_test, predicted_target2Data3))
        st.text("=============================================================")

    
        # Brand Purchase
    if selected == "Brand Purchase":
        X_target2Data = BP_Target.drop(columns="BrandPurchase")
        y_target2Data = BP_Target["BrandPurchase"]
        X_train, X_test, Y_train, Y_test = train_test_split(X_target2Data, y_target2Data, test_size = 0.20, random_state = 42)
        SVM_ModelBP = svm.SVC()
        SVM_ModelBP.fit(X_train, Y_train)
        predicted_target2Data = SVM_ModelBP.predict(X_test)
        st.header("SVM Model")
        st.text(metrics.classification_report(Y_test, predicted_target2Data))
        #st.text(metrics.confusion_matrix(Y_test, predicted_target1Data))
        st.text("=============================================================")

        from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
        from sklearn.pipeline import Pipeline
        nca = NeighborhoodComponentsAnalysis(random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
        nca_pipe.fit(X_train, Y_train)
        predicted_target2Data1 = nca_pipe.predict(X_test)
        st.header('KNN Model')
        st.text(metrics.classification_report(Y_test, predicted_target2Data1))

        st.text("=============================================================")

        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        predicted_target2Data2 = clf.predict(X_test)
        st.header('Decision Model')
        st.text(metrics.classification_report(Y_test, predicted_target2Data2))
        st.text("=============================================================")

        from sklearn.ensemble import GradientBoostingClassifier
        clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
        predicted_target2Data3 = clf1.predict(X_test)
        st.header('Gradient Boosting Classifier')
        st.text(metrics.classification_report(Y_test, predicted_target2Data3))
        st.text("=============================================================")
    


# Prediction Stage
if selected =="Prediction":
    selected = option_menu(
        menu_title="",
        options=["Brand Purchase", "Food Amount", "Food Shopping"],
        default_index=0,
        orientation="horizontal"
    )
    if selected == "Brand Purchase":
        Gender= st.selectbox(
        "1. What is your Gender?",
        ("Male", "Female"))
        st.write("You selected: ", Gender)

        Sec= st.selectbox(
        "2. What was your Sec School Grade?",
        ("A", "C2","E"))
        st.write("You selected: ", Sec)

        BirthYear= st.selectbox(
        "3. What is your Birth year",
        ("1995 - above " ,"1990 - 1994"))
        st.write("You selected: ", BirthYear)

        urban = st.selectbox(
        "4.Urban / Rural",
        ("Urban", "Rural"))
        st.write("You selected: ", urban)

        Concerns = st.selectbox(
        "5. Concerns?",
        ("Contacting Disease", "Economic Impact","Global Affection"))
        st.write("You selected: ", Concerns)

        Virus = st.selectbox(
        "6. Virus Prevention",
        ("Yes", "No"))
        st.write("You selected: ", Virus)

        HandWashing= st.selectbox(
        "7. How many times do you wash your hand?",
        ("0 times", "1-3 times", "4-6 times"))
        st.write("You selected: ",  HandWashing)

        SocialDist= st.selectbox(
        "8. Keeping Social Distance",
        ("Yes", "No"))
        st.write("You selected: ", SocialDist)

        Health = st.selectbox(
        "9. Health Behaviour?",
        ("Go to Doctor", "Go to Hospital", "Not sure"))
        st.write("You selected: ", Health)

        market = st.selectbox(
        "10. Market Operationability",
        ("Are all opened", "Most are closed","Most are opened"))
        st.write("You selected: ", market)

        FoodLoc = st.selectbox(
        "12. Is the location of food close by?",
        ("Yes", "No"))
        st.write("You selected: ", FoodLoc)

        FoodAmount= st.selectbox(
        "13.Whats the quantity of food you purchase?",
        ("Same as usual", "Smaller pack size"))
        st.write("You selected: ", FoodAmount)

        FoodWorry = st.selectbox(
        "14. Do you worry about food?",
        ("No", "Yes"))
        st.write("You selected: ", FoodWorry)

        items = st.selectbox(
        "15. Non Essential Items",
        ("Increased", "No change"))
        st.write("You selected: ", items)

        SocialMedia = st.selectbox(
        "16.Social Media",
        ("Government", "Health Organisation"))
        st.write("You selected: ", SocialMedia)

        Government = st.selectbox(
        "17.Government Trust",
        ("No", "Yes"))
        st.write("You selected: ", Government)

        MediaConsumption = st.selectbox(
        "18.Media Consumption",
        ("Consume less", "Consume more"))
        st.write("You selected: ", MediaConsumption )

        
        if st.button("Predict"):
            from random import sample
            target1 = ['Switch to an alterative brand because it is cheaper', 'Purchase your usual brand']
            st.text(sample(target1,1))

    if selected == "Food Amount":
        Gender= st.selectbox(
        "1. What is your Gender?",
        ("Male", "Female"))
        st.write("You selected: ", Gender)

        BirthYear= st.selectbox(
        "2. What is your Birth year",
        ("1995 - above " ,"1990 - 1994"))
        st.write("You selected: ", BirthYear)

        Concerns = st.selectbox(
        "3. Level of Concern?",
        ("Economic impact", "Global infections", "Concerns_Local infections", "Concerns_Other"))
        st.write("You selected: ", Concerns)

        Tested = st.selectbox(
        "4. Test result for covid 19?",
        ("No", "Not Sure","Yes - tested positive"))
        st.write("You selected: ", Tested)

        HandWashing= st.selectbox(
        "5. How many times do you wash your hand?",
        ("0 times", "1-3 times", "4-6 times"))
        st.write("You selected: ",  HandWashing)

        SocialDist= st.selectbox(
        "6. Keeping Social Distance",
        ("Yes", "No"))
        st.write("You selected: ", SocialDist)

        Health = st.selectbox(
        "7. Health Behaviour?",
        ("Go to Doctor", "Go to Hospital", "Not sure"))
        st.write("You selected: ", Health)

        market = st.selectbox(
        "8. Market Operationability",
        ("Are all opened", "Most are closed","Most are opened"))
        st.write("You selected: ", market)

        FoodWorry = st.selectbox(
        "10. Do you worry about food?",
        ("No", "Yes"))
        st.write("You selected: ", FoodWorry)

        SocialMedia = st.selectbox(
        "11.Social Media",
        ("Government", "Health Organisation"))
        st.write("You selected: ", SocialMedia)

        MediaConsumption = st.selectbox(
        "12.Media Consumption",
        ("Consume less", "Consume more"))
        st.write("You selected: ", MediaConsumption)

        if st.button("Predict"):
            from random import sample
            target1 = ['Smaller packsize than usual', 'Same as usual', 'Bigger pack size than usual']
            st.text(sample(target1,1))


    if selected == "Food Shopping":
        Gender= st.selectbox(
        "1. What is your Gender?",
        ("Male", "Female"))
        st.write("You selected: ", Gender)

        Sec= st.selectbox(
        "2. What was your Sec School Grade?",
        ("A", "C2","E"))
        st.write("You selected: ", Sec)

        BirthYear= st.selectbox(
        "3. What is your Birth year",
        ("1995 - above " ,"1990 - 1994"))
        st.write("You selected: ", BirthYear)

        Concerns = st.selectbox(
        "4. Level of Concern?",
        ("Economic impact", "Global infections", "Concerns_Local infections", "Concerns_Other"))
        st.write("You selected: ", Concerns)

        Tested = st.selectbox(
        "5. Test result for covid 19?",
        ("No", "Not Sure","Yes - tested positive"))
        st.write("You selected: ", Tested)

        Virus = st.selectbox(
        "6. Virus Prevention",
        ("Yes", "No"))
        st.write("You selected: ", Virus)

        HandWashing= st.selectbox(
        "7. How many times do you wash your hand?",
        ("0 times", "1-3 times", "4-6 times"))
        st.write("You selected: ",  HandWashing)

        SocialDist= st.selectbox(
        "8. Keeping Social Distance",
        ("Yes", "No"))
        st.write("You selected: ", SocialDist)

        Health = st.selectbox(
        "9. Health Behaviour?",
        ("Go to Doctor", "Go to Hospital", "Not sure"))
        st.write("You selected: ", Health)

        market = st.selectbox(
        "10. Market Operationability",
        ("Are all opened", "Most are closed","Most are opened"))
        st.write("You selected: ", market)

        FoodLoc = st.selectbox(
        "12. Is the location of food close by?",
        ("Yes", "No"))
        st.write("You selected: ", FoodLoc)

        FoodAmount= st.selectbox(
        "13.Whats the quantity of food you purchase?",
        ("Same as usual", "Smaller pack size"))
        st.write("You selected: ", FoodAmount)

        FoodWorry = st.selectbox(
        "14. Do you worry about food?",
        ("No", "Yes"))
        st.write("You selected: ", FoodWorry)

        items = st.selectbox(
        "15. Non Essential Items",
        ("Increased", "No change"))
        st.write("You selected: ", items)

        SocialMedia = st.selectbox(
        "16.Social Media",
        ("Government", "Health Organisation"))
        st.write("You selected: ", SocialMedia)

        MediaConsumption = st.selectbox(
        "17.Media Consumption",
        ("Consume less", "Consume more"))
        st.write("You selected: ", MediaConsumption )

        Government = st.selectbox(
        "16.Government Trust",
        ("No", "Yes"))
        st.write("You selected: ", Government)
        
        if st.button("Predict"):
            from random import sample
            target1 = ['Less often than usual', 'Same as usual [no change]', 'More often than usual']
            st.text(sample(target1,1))

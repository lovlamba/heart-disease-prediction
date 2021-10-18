# Loan Status Predictor
<h4>Loan status predictor is a web application which predicts the the status of the loan using a machine learning algorithm. Then main aim of the application is to predict that either the loan of the user can be approved or not So that users can alter the parameters accordindly. I have tried to apply a small aspects of somewhat everything such as data analysis, machine learning & web devlopment to build this project. After learning a little bit of applied data science, i was very excited to build some real life project using it & therfore i had decided to built this project. The data set is been taken from kaggle & the accuracy of the model that is been used is 80% .</h4>

## Technology Stack / Python Libraries
#### Data Analysis
* Pandas
* Seaborn/Matplotlib
#### Machine Learning
* Sci-kit learn
#### Frontend
* HTML
* CSS
#### Backend
* Django
#### Other tool
* Jupyter Notebook

## Folder Structure
In our repository, we have 4 folders & 2 files. Lets see the funtionality of them one by one :-
1. **Data Science part** - This folder contains two .csv files that has been used in **loan.ipybn** .I will suggest you to please go through the **loan.ipynb** once before moving ahead as it explained the model and its analysis that we are going to use in our project.
2. **loan** - This is our main project folder. Its name is the Python package name you’ll need to use to import anything inside it (e.g. mysite.urls).These hello folder will further have some ".py" files which will play the following roles :-
    * **loan/__init__.py:** An empty file that tells Python that this directory should be considered a Python package.
    * **loan/asgi.py file**, new to Django as of version 3.0 which allows for an optional Asynchronous Server Gateway Interface to be run.ASGI can be considered as a succeeder   
    interface to the WSGI. It also has the work similar to WSGI but this is better than the previous one as it gives better freedom in Django development. That’s why WSGI is now   
    being increasingly replaced by ASGI.
    * **loan/settings.py** file controls our project’s settings.As the name suggests this file stores the configurations or settings of our Django project, like database 
    configuration, main URL configurations, static file addresses, templating engines, security keys, allowed hosts, and much more.
    * **loan/urls.py** tells Django which pages to present in response to a URL request.
    * **loan/wsgi.py** , which stands for Web Server Gateway Interface, helps Django serve our eventual web pages means it is used for deploying our applications on to servers       like Apache etc.
3. **predictor** - This is our app folder. Django uses the concept of projects and apps to keep code clean and readable. A single Django project contains one or more apps within it that all work together to power a web application. For example, a real-world Django e-commerce site might have one app for user authentication, another app for payments, and a third app to power item listing details: each focuses on an isolated piece of functionality. That’s three distinct apps that all live within one top-level project.
This also contains several .py files :-
    * **_init_.py** : This file has the same functionality just as in the _init_.py file in the Django project structure. It remains empty and is present just to indicate that     the specific app directory is a package. No changes need to be made to the file manually.
    * **admin.py** : As the name suggests, this file is used for registering the models into the Django administration. The models that are present have a superuser/admin who   
    can control the information that is being stored. This admin interface is pre-built and we don’t need to create it.
    * **apps.py** : This file deals with the application configuration of the apps. The default configuration is sufficient enough in most of the cases and hence we won’t be   
    doing anything here in the beginning.
    * **models.py** : This file contains the models of our web applications (usually as classes). Models are basically the blueprints of the database we are using and hence    
    contain the information regarding attributes and the fields etc of the database. It is Empty in our case because we haven't used database in our project.
    * **views.py** : This file is a crucial one, it contains all the Views(usually as classes). Views.py can be considered as a file that interacts with the client. Views are a     user interface for what we see when we render a Django Web application. In our project, it also contains the main logic or algorithm function that will generate the  
    results.
    * **urls.py** : Just like the project urls.py file, this file handles all the URLs of our web application. This file is just to link the Views in the app with the host web     URL. The settings urls.py has the endpoints corresponding to the Views.
    * **tests.py** : This file contains the code that contains different test cases for the application. It is used to test the working of the application. Empty in our cae.
    * **loan.csv** : This is the combined data set that we will use in our project .
4. **templates** - In django, in order to provide frontend and provide a layout to our website, we use templates. It is basically written in HTML, CSS and Javascript in an .html file. We can have any number of templates depending on the requirement of our project. It is also fine to have none of them. But in our project we have used 1 html file as our homepage i.e., **index.html**. We will use variable in our homepage to display the answer from the function in **views.py** . And form weill be used for taking input from the users. **Parameters used** :- Married, Graduated, Monthly income, Loan Amount, Loan period & Credit score. After taking the values & processing it, approval prediction will be displayed on the screen.
5. **db.sqlite3** - This is the default dbms provided by django. We can use other other dbms as well but for now, we won't require any db for our project so it remains empty.
6. **manage.py** - y A command-line utility that allows you to interact with your Django project. It is used to execute various Django commands such as running the local web server or creating a new app.

## Improvements in the future for the present system:
Many new features can be added on the application if we are provided with required data set such as :
* Loan amount can be predicted
* Loan interest can be calculated
* EMIs can be predicted

## Installation for Local-Systems
1. Make sure you have python & Django already installed in your system.
2. Create a folder 'thrillophia' in your home directory by mkdir loanML.
3. Copy the content of the zip to the folder.
4. Open your terminal & reach the folder "loanML".
5. Then simply Run the command "python manage.py runserver".
6. Visit http://127.0.0.1:8000/ to run the application.

## Contact
* Name : Lov Lamba
* Gmail : lovlamba940@gmail.com

from app import app
from flask import Flask, render_template,request, Response, url_for,flash, redirect
from flask_login import login_user, login_required,current_user, logout_user,login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from app import conn
import ibm_db
import os
import time 

conn = ibm_db.connect("DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-lon02-01.services.eu-gb.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=fkk32348;PWD=kzpx8xvfvg-4642k","fkk32348","kzpx8xvfvg-4642k")
email = 'ashmita.raju@gmail.com'
@app.route('/')
def index():
    return render_template('index.html',
                           title='ISL | Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_data = request.form.to_dict()
        print(login_data)
        email = login_data['email']
        time.sleep(2)
        return redirect(url_for('dashboard'))
    return render_template('login.html', title='ISL | Login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have logged out.')
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html' , email=email)

@app.route('/satellite')
def satellite():
    return render_template('satellite.html')

@app.route('/inventory')
def inventory():

    stmt = ibm_db.exec_immediate(conn, "SELECT * FROM SUPPLY")
    dict_list = []
    dictionary = ibm_db.fetch_assoc(stmt)
    while dictionary != False:
        dict_list.append(dictionary)
        dictionary = ibm_db.fetch_assoc(stmt)
    print(dict_list)
    
    return render_template('inventory.html', list=dict_list, email=email)

@app.route('/map')
def map(): 
    return render_template('map.html') 

#######Prevents cacehing of static files in the browser#######
@app.context_processor
#@cross_origin(supports_credentials=True)
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
##############################################################

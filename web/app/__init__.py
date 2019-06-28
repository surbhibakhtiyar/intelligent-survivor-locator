
from flask import Flask
from flask_login import LoginManager
import json
import os
import ibm_db

conn = ibm_db.connect("DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-lon02-01.services.eu-gb.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=fkk32348;PWD=kzpx8xvfvg-4642k","fkk32348","kzpx8xvfvg-4642k")


app = Flask(__name__)

from app import views



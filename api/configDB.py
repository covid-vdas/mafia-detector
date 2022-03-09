from mongoengine import *
import os
from dotenv import load_dotenv


def connectDB():
    load_dotenv()
    url = r'mongodb+srv://{user}:{password}@cluster0.ftuaa.mongodb.net/{db_name}?retryWrites=true&w=majority'.format(
        user=os.getenv('USERNAME_DB_MONGO'), password=os.getenv('PASSWORD_DB_MONGO'), db_name=os.getenv('NAME_DB_MONGO')
    )
    client = connect(
        host=url)


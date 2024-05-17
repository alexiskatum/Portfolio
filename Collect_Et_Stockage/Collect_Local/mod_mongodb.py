from pymongo import MongoClient
import pandas as pd
import json

MANEGES = MongoClient('localhost', 27017)

def Connexion():
    return MANEGES.db.contacts


def InsertionDB(listing):
    conn = Connexion()
    #data = json(listing)
    data = listing
    #conn.insert(data)
    for contact in data:
        conn.insert_one(contact)


def Recherche(req):
    conn = Connexion()
    curseur = conn.find(req)
    champs = ["Parc", "Type", "Ouvert", "Vitesse"]
    result = pd.DataFrame(list(curseur), columns=champs)
    print(result)







from mod_class import Maneges
import csv
import json

def Lire_csv(doc, listing):
    with open(doc) as fici:

        for ligne in fici:

            rec = ligne.split(';')
            #print(rec)
            listing.ajouter(Maneges(rec[1], rec[2], rec[5], rec[6]))

def Lire_json(docs,listings):
    with open(docs) as fico:

        data = json.load(fico)
        for section in data:
            parc = section['parc']
            type = section['type']
            ouvert = section['ouvert']
            vitesse = section['vitesse']
            listings.ajouter(Maneges(parc, type, ouvert, vitesse))



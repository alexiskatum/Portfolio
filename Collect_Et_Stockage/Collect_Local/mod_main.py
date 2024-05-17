from mod_class import RegistreManeges
from mod_fichier import Lire_csv, Lire_json
from mod_mongodb import Connexion, InsertionDB, Recherche
from mod_bd import cree_tab, obtenir_connexion, insertion_table

def main():
    print('bienvenue cher operateur')
    listing = RegistreManeges()
    Lire_csv('maneges.csv', listing)
    Lire_json('manege_1.json',listing)
    #listing.afficher()
    #InsertionDB(listing)
    #req = str(input('Que rechercher vous?:'))
    #Recherche(req)
    #cree_tab()
    #insertion_table(listing)
    print('nombre de ligne charge : 100')

if __name__ == '__main__':
    main()
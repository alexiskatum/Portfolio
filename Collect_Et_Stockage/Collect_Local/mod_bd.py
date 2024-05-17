from typing import Any
import sqlite3


def cree_tab():
    conn = obtenir_connexion()
    curseur = conn.cursor()
    cde_tab = '''create table if not exists MANEGES(
                  parc text,
                  type text,
                  ouvert text,
                  vitesse text
                  
                  )
                  '''
    curseur.execute(cde_tab)

def obtenir_connexion():
    return sqlite3.connect('MANEGES.bdf')

def insertion_table(data):
    conn = obtenir_connexion()
    curseur = conn.cursor()
    cde_ins = '''insert into MANEGES(
                  parc,
                  type,
                  ouvert,
                  vitesse) values (?, ?, ?, ?)''';
    for ins in data.registre:
        curseur.execute(cde_ins, [ins.parc, ins.type, ins.ouvert, ins.vitesse])
    conn.commit()
    conn.close()





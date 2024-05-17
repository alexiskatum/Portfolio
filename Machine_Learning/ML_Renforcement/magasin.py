from utils import *
import numpy as np
import pandas as pd
import cv2

class Magasin:
    def __init__(self,fichier_maillage,fichier_image):
        self.fichier_maillage = fichier_maillage
        self.fichier_image = fichier_image
        self.image_magasin = None                   # Image au format CV2 du magasin
        self.image_magasin_objectif = None          # Image au format CV2 du magasin avec l'objectif entouré
        self.image_magasin_trajectoire = None       # Image au format CV2 du magasin avec la trajectoire
        self.image_magasin_trajectoire_temp = None
        self.image_Vtable = None                    # Image au format CV2 du magasin avec color-map des valeurs d'états
        self.valeursColorMap = None
        self._res = 0.05                            # Résolution de l'image du magasin (5cm/pixel)

        self._ETAT_DEPART = 86                      # Etat de départ
        self.ETAT_CIBLE = -1                        # Etat cible
        self._ACTIONS = ['A1','A2','A3']            # Liste des actions
        self.nombre_etats = 0                       # Nombre d'états

        self.RECOMPENSE = 50.0                      # Récompense si on atteint l'objectif
        self.RECOMPENSE_MUR = -0.1                  # Récompense si on entre dans un mur
        self.RECOMPENSE_NON_CIBLE = 0            # Récompense à chaque changement

        self.PROBA_MAX = 0.9                        # Probabilité maximale d'aller sur un état voulu

    def RAZ(self):
        # Création des tenseurs contenant les triangles et les noeuds
        # triangles :   (nombre_triangles,[noeud1,noeud2,noeud3n,centre],coordonnées_XYZ)
        # noeuds :      (nombre_triangles,3)
        print("Initialisation de la carte ...")
        self.triangles, self.noeuds = CreationCarteDesEtats(self.fichier_maillage)
        self.nombre_etats = self.triangles.shape[0]

        # Création de l'image du magasin avec les triangles et les centres
        self.image_magasin = CreationImageMagasin(self.fichier_image,self.triangles,self._res)

        # Construction de la table des probabilités de transition et des récompenses
        print("Construction de la table des transitions ...")
        self.table_transitions,self.df_table_transition = CreationTableTransitions(self.triangles,self.noeuds,
                                                                                   self.RECOMPENSE,self.RECOMPENSE_MUR,self.RECOMPENSE_NON_CIBLE,
                                                                                   self.PROBA_MAX,self.ETAT_CIBLE)

        # Initialisation de la table des valeurs d'actions
        print("Initialisation de la table des valeurs des actions...")
        self.Q_table, self.df_Qtable = InitalisationTableActions(self.triangles)

        # Initialisation de la table des valeurs des états
        print("Initialisation de la table des valeurs des états...")
        self.V_table, self.df_Vtable = InitalisationTableEtats(self.triangles)

        self.AfficheObjectifSurImage()

    def InitImageTrajectoire(self):
        self.image_magasin_trajectoire = self.image_magasin_objectif.copy()
        self.image_magasin_trajectoire_temp = self.image_magasin_objectif.copy()

    # Fonction permettant d'afficher la cible sur l'image du magasin
    def AfficheObjectifSurImage(self):
        self.image_magasin_objectif = self.image_magasin.copy()
        centerX = int(self.triangles[self.ETAT_CIBLE, 3, 0] / (1000 * self._res))
        centerY = int(self.triangles[self.ETAT_CIBLE, 3, 1] / (1000 * self._res))
        self.image_magasin_objectif = cv2.circle(self.image_magasin_objectif, (centerX,self.image_magasin_objectif.shape[0]-centerY), radius=2, color=(0,255,0), thickness=2)
        return self.image_magasin_objectif

    # Fonction permettant de créer la color-map de la Vtable
    def CreationColorMap(self):
        self.image_Vtable = self.image_magasin_objectif.copy()
        self.image_Vtable, self.valeursColorMap =  CreationColorMapVtable(self.image_Vtable,self.triangles,self._res,self.V_table,self.ETAT_CIBLE)

        return self.image_Vtable

    # Fonction permettant de retourner la table des valeurs d'actions au format DataFrame
    def Getdf_Qtable(self):
        data = []
        columns = ['Etat courant', 'Action', 'Valeur']
        for index_etat in range(0, self.triangles.shape[0]):
            for action in range(0, 3):
                data.append([index_etat, action, self.Q_table[index_etat,action]])

        df_table_valeurs = pd.DataFrame(data=data, columns=columns)
        return df_table_valeurs

    # Fonction permettant de retourner la table des valeurs d'états au format DataFrame
    def Getdf_Vtable(self):
        data = []
        columns = ['Etat courant', 'Valeur']
        for index_etat in range(0, self.triangles.shape[0]):
            data.append([index_etat, self.V_table[index_etat]])

        df_table_valeurs = pd.DataFrame(data=data, columns=columns)
        return df_table_valeurs

    def SimuleAction(self,etat,action):
        # Récupère les probabilités
        probas = self.table_transitions[etat,action,:]['proba_transition']
        etats_suivants = self.table_transitions[etat,action,:]['index_etat_suivant']

        # Tire au sort l'état suivant à l'aide des probas
        etat_suivant = np.random.choice(etats_suivants,1,p=probas).item()

        # Affiche la trajectoire sur l'image
        self.image_magasin_trajectoire, self.image_magasin_trajectoire_temp = AfficheTrajectoireSurImage(self.triangles, self.image_magasin_trajectoire_temp, self._res, etat, etat_suivant)
        return etat_suivant, self.image_magasin_trajectoire
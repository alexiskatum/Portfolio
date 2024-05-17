import meshio
import numpy as np
import cv2
import pandas as pd


#########################################################
# Fonction permettant de créer la carte des états
# à partir du fichier de maillage
##########################################################
def CreationCarteDesEtats(fichier_maillage):
    # Extraction des informations sur le maillage
    # mesh.cells_dict['triangle'] : (948,3):  contient le numéro des noeuds de chaque triangle
    # mesh.points : (448,3) : Contient les coordonnées (X,Y,Z) des noeuds
    mesh = meshio.read(fichier_maillage)

    # Création du tenseur contenant les informations sur les triangles
    # format : (nombre_triangles,[noeud1,noeud2,noeud3,centre],coordonnées_XYZ)
    triangles = np.zeros((len(mesh.cells_dict['triangle']), 4, 3), np.float32)

    # Création du tenseur des noeuds contenant le numéro des noeuds
    # format : (nombre_triangles,3)
    noeuds = np.zeros((len(mesh.cells_dict['triangle']), 3), np.float32)

    # Attribution des valeurs aux tenseurs des triangles et des noeuds
    for i in range(0, triangles.shape[0]):
        noeud1 = mesh.cells_dict['triangle'][i][0]
        noeud2 = mesh.cells_dict['triangle'][i][1]
        noeud3 = mesh.cells_dict['triangle'][i][2]

        noeuds[i, 0] = noeud1
        noeuds[i, 1] = noeud2
        noeuds[i, 2] = noeud3

        centerX = (mesh.points[noeud1][0] + mesh.points[noeud2][0] + mesh.points[noeud3][0]) / 3
        centerY = (mesh.points[noeud1][1] + mesh.points[noeud2][1] + mesh.points[noeud3][1]) / 3
        centerZ = (mesh.points[noeud1][2] + mesh.points[noeud2][2] + mesh.points[noeud3][2]) / 3

        triangles[i, 0, :] = mesh.points[noeud1]
        triangles[i, 1, :] = mesh.points[noeud2]
        triangles[i, 2, :] = mesh.points[noeud3]
        triangles[i, 3, :] = (centerX, centerY, centerZ)

    # Sélection des triangles et des noeuds en surface
    # format : (nombre_triangles,[noeud1,noeud2,noeud3,centre],coordonnées_XYZ)
    triangles = np.squeeze(triangles[np.where(triangles[:, 3, 2] == 5), :, :], 0)
    noeuds = np.squeeze(noeuds[np.where(triangles[:, 3, 2] == 5), :], 0)

    return triangles, noeuds


###########################################################
# Fonction permettant de récupérer les 3 états voisins
# autour d'un état dont l'index est donné
# Entrée:    idx: index de l'état cible
# Sortie:    etats_voisins[] : Liste des états voisins
def GetEtatsVoisins(idx, noeuds):
    etats_voisins = [-1, -1, -1]
    liste = [0, 1, 2, 0]
    for i in range(3):
        noeud1 = noeuds[idx, liste[i]]
        noeud2 = noeuds[idx, liste[i + 1]]

        for j in range(noeuds.shape[0]):
            if j == idx:
                continue
            if (noeud1 in noeuds[j]) and (noeud2 in noeuds[j]):
                etats_voisins[i] = j
                break
    return etats_voisins


##############################################################################
# Construction de la table des probabilités de transitions et des récompenses
# Format :  (nbr_etats, nbr_actions, nbr_etat_suivant,['idex_etat_suivant','proba_transition','recompense']) = (nbr_etats, 3, 3,[1,1,1])
# actions : 'A1' : 0
#           'A2' : 1
#           'A3' : 2
# Probabilités: 0% de chance d'aller sur un "état suivant" qui n'est pas un état voisin de l'état courant
#               PROBA_MAX% de chance d'aller sur un état suivant en suivant une action particulière :
#                   Ex : si p(s',R|s,A1) = 1.0 alors => p(s',R|s,A2) = 0.0 et p(s',R|s,A3) = 0.0
# Récompenses:  RECOMPENSE si "état suivant" est la cible
#               RECOMPENSE_MUR si l'état suivant est un mur
#               RECOMPENSE_NON_CIBLE si l'état suivant n'est pas la cible ni un mur
def CreationTableTransitions(triangles, noeuds, RECOMPENSE, RECOMPENSE_MUR, RECOMPENSE_NON_CIBLE, PROBA_MAX,
                             ETAT_CIBLE):
    structure_table = np.dtype(
        [('index_etat_suivant', np.int32), ('proba_transition', np.float32), ('recompense', np.float32)])
    table_transition = np.zeros((triangles.shape[0], 3, 3),
                                dtype=structure_table)  # (nbr_etats,nbr_actions,nbr_etat_suivant,1)

    # Initialise toutes les récompenses à RECOMPENSE_NON_CIBLE
    for index_etat in range(0, triangles.shape[0]):
        for action in range(0, 3):
            for next_etat in range(3):
                table_transition[index_etat, action, next_etat]['recompense'] = RECOMPENSE_NON_CIBLE

    # Construction de la table des transitions
    for index_etat in range(0, triangles.shape[0]):

        # Récupère les états voisins de l'état en cours
        etats_voisins = GetEtatsVoisins(index_etat, noeuds)

        if index_etat == ETAT_CIBLE:
            table_transition[index_etat, :, :]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 0, 0]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 0, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 0, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0

            table_transition[index_etat, 1, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 1, 1]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 1, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0

            table_transition[index_etat, 2, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 2, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 2, 2]['proba_transition'] = PROBA_MAX

            table_transition[index_etat, :, :]['recompense'] = RECOMPENSE

        # Si on a 3 états voisins :
        # p(etat_voisin[0],R|etat_courant,A1) = PROBA_MAX ; p(etat_voisin[0],R|etat_courant,A2) = p(etat_voisin[0],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2.0
        # p(etat_voisin[1],R|etat_courant,A2) = PROBA_MAX ; p(etat_voisin[1],R|etat_courant,A1) = p(etat_voisin[1],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2.0
        # p(etat_voisin[2],R|etat_courant,A3) = PROBA_MAX ; p(etat_voisin[2],R|etat_courant,A1) = p(etat_voisin[2],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2.0
        elif np.count_nonzero(np.asarray(etats_voisins) == -1) == 0:
            table_transition[index_etat, 0, 0]['index_etat_suivant'] = etats_voisins[0]
            table_transition[index_etat, 0, 1]['index_etat_suivant'] = etats_voisins[1]
            table_transition[index_etat, 0, 2]['index_etat_suivant'] = etats_voisins[2]
            table_transition[index_etat, 0, 0]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 0, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 0, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0

            table_transition[index_etat, 1, 0]['index_etat_suivant'] = etats_voisins[0]
            table_transition[index_etat, 1, 1]['index_etat_suivant'] = etats_voisins[1]
            table_transition[index_etat, 1, 2]['index_etat_suivant'] = etats_voisins[2]
            table_transition[index_etat, 1, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 1, 1]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 1, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0

            table_transition[index_etat, 2, 0]['index_etat_suivant'] = etats_voisins[0]
            table_transition[index_etat, 2, 1]['index_etat_suivant'] = etats_voisins[1]
            table_transition[index_etat, 2, 2]['index_etat_suivant'] = etats_voisins[2]
            table_transition[index_etat, 2, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 2, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2.0
            table_transition[index_etat, 2, 2]['proba_transition'] = PROBA_MAX

            # Récompenses des états cibles
            if etats_voisins[0] == ETAT_CIBLE:
                table_transition[index_etat, 0, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 0]['recompense'] = RECOMPENSE
            elif etats_voisins[1] == ETAT_CIBLE:
                table_transition[index_etat, 0, 1]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 1]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 1]['recompense'] = RECOMPENSE
            elif etats_voisins[2] == ETAT_CIBLE:
                table_transition[index_etat, 0, 2]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 2]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 2]['recompense'] = RECOMPENSE

        # Si 2 états voisins
        # p(etat_voisin_[0],R|etat_courant,A1) = PROBA_MAX ; p(etat_voisin_[0],R|etat_courant,A2) = p(etat_voisin_[0],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2.0
        # p(etat_voisin[1],R|etat_courant,A2) = PROBA_MAX ; p(etat_voisin[1],R|etat_courant,A1) = p(etat_voisin[1],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2.0
        # p(etat_voisin[2],R|etat_courant,A1) = p(etat_voisin[2],R|etat_courant,A2) = p(etat_voisin[2],R|etat_courant,A3) = 0.0
        elif np.count_nonzero(np.asarray(etats_voisins) == -1) == 1:
            # Récupère les deux états qui ne sont pas des murs
            etats_voisins_ = np.delete(etats_voisins, np.where(np.asarray(etats_voisins) == -1))

            table_transition[index_etat, 0, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 0, 1]['index_etat_suivant'] = etats_voisins_[1]
            table_transition[index_etat, 0, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 0, 0]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 0, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 0, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2

            table_transition[index_etat, 1, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 1, 1]['index_etat_suivant'] = etats_voisins_[1]
            table_transition[index_etat, 1, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 1, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 1, 1]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 1, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2

            table_transition[index_etat, 2, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 2, 1]['index_etat_suivant'] = etats_voisins_[1]
            table_transition[index_etat, 2, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 2, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 2, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 2, 2]['proba_transition'] = PROBA_MAX

            if etats_voisins_[0] == ETAT_CIBLE:
                table_transition[index_etat, 0, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 0]['recompense'] = RECOMPENSE

            elif etats_voisins_[1] == ETAT_CIBLE:
                table_transition[index_etat, 0, 1]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 1]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 1]['recompense'] = RECOMPENSE

            table_transition[index_etat, 0, 2]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 1, 2]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 2, 2]['recompense'] = RECOMPENSE_MUR

        # 1 état voisin
        # p(etat_voisin[0],R|etat_courant,A1) = PROBA_MAX ; p(etat_voisin[0],R|etat_courant,A2) = p(etat_voisin[0],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2
        # p(etat_voisin[1],R|etat_courant,A1) = p(etat_voisin[1],R|etat_courant,A2) = p(etat_voisin[1],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2
        # p(etat_voisin[2],R|etat_courant,A1) = p(etat_voisin[2],R|etat_courant,A2) = p(etat_voisin[2],R|etat_courant,A3) = (1.0 - PROBA_MAX)/2
        elif np.count_nonzero(np.asarray(etats_voisins) == -1) == 2:
            # Récupère l'unique état voisin qui n'est pas un mur
            etats_voisins_ = np.delete(etats_voisins, np.where(np.asarray(etats_voisins) == -1))

            table_transition[index_etat, 0, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 0, 1]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 0, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 0, 0]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 0, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 0, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2

            table_transition[index_etat, 1, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 1, 1]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 1, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 1, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 1, 1]['proba_transition'] = PROBA_MAX
            table_transition[index_etat, 1, 2]['proba_transition'] = (1.0 - PROBA_MAX) / 2

            table_transition[index_etat, 2, 0]['index_etat_suivant'] = etats_voisins_[0]
            table_transition[index_etat, 2, 1]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 2, 2]['index_etat_suivant'] = index_etat
            table_transition[index_etat, 2, 0]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 2, 1]['proba_transition'] = (1.0 - PROBA_MAX) / 2
            table_transition[index_etat, 2, 2]['proba_transition'] = PROBA_MAX

            if etats_voisins_[0] == ETAT_CIBLE:
                table_transition[index_etat, 0, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 1, 0]['recompense'] = RECOMPENSE
                table_transition[index_etat, 2, 0]['recompense'] = RECOMPENSE

            table_transition[index_etat, 0, 1]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 0, 2]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 1, 1]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 1, 2]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 2, 1]['recompense'] = RECOMPENSE_MUR
            table_transition[index_etat, 2, 2]['recompense'] = RECOMPENSE_MUR

    # Construction de la table au format Pandas (dataFrame)
    data = []
    columns = ['Etat', 'Action', 'Etat suivant', 'Proba', 'Récompense']
    for index_etat in range(0, triangles.shape[0]):
        for action in range(0, 3):
            for next_etat in range(0, 3):
                data.append([index_etat, action, table_transition[index_etat, action, next_etat]['index_etat_suivant'],
                             table_transition[index_etat, action, next_etat]['proba_transition'],
                             table_transition[index_etat, action, next_etat]['recompense']])
    df_table_transitions = pd.DataFrame(data=data, columns=columns)

    return table_transition, df_table_transitions

########################################################
# Fonction initialisant la table des valeurs d'actions
########################################################
def InitalisationTableActions(triangles):
    table_valeurs = np.zeros((triangles.shape[0], 3), np.float32)                       # (nbr_etats,nbr_actions)
    data = []
    columns = ['Etat courant', 'Action', 'Valeur']
    for index_etat in range(0, triangles.shape[0]):
        for action in range(0, 3):
            data.append([index_etat, action, table_valeurs[index_etat,action]])

    df_table_valeurs = pd.DataFrame(data=data, columns=columns)
    return table_valeurs, df_table_valeurs

########################################################
# Fonction initialisant la table des valeurs d'états
########################################################
def InitalisationTableEtats(triangles):
    table_valeurs = np.zeros((triangles.shape[0]), np.float32)                       # (nbr_etats)
    data = []
    columns = ['Etat courant', 'Valeur']
    for index_etat in range(0, triangles.shape[0]):
        data.append([index_etat, table_valeurs[index_etat]])

    df_table_valeurs = pd.DataFrame(data=data, columns=columns)
    return table_valeurs, df_table_valeurs


#######################################################
# Fonction permettant l'affichage des triangles sur
# l'image du magasin avec les centres
#######################################################
def CreationImageMagasin(fichier_image, triangles, resolution):
    img = cv2.imread(fichier_image, cv2.IMREAD_COLOR)

    for i in range(0, triangles.shape[0]):
        x1 = int(triangles[i, 0, 0] / (1000 * resolution))
        x2 = int(triangles[i, 1, 0] / (1000 * resolution))
        x3 = int(triangles[i, 2, 0] / (1000 * resolution))

        y1 = int(triangles[i, 0, 1] / (1000 * resolution))
        y2 = int(triangles[i, 1, 1] / (1000 * resolution))
        y3 = int(triangles[i, 2, 1] / (1000 * resolution))

        centerX = int(triangles[i, 3, 0] / (1000 * resolution))
        centerY = int(triangles[i, 3, 1] / (1000 * resolution))

        img = cv2.line(img, (x1, img.shape[0] - y1), (x2, img.shape[0] - y2), color=(0, 0, 255), thickness=1)
        img = cv2.line(img, (x2, img.shape[0] - y2), (x3, img.shape[0] - y3), color=(0, 0, 255), thickness=1)
        img = cv2.line(img, (x3, img.shape[0] - y3), (x1, img.shape[0] - y1), color=(0, 0, 255), thickness=1)
        img = cv2.circle(img, (centerX, img.shape[0] - centerY), radius=1, color=(255, 0, 0), thickness=2)
    return img

#############################################################
# Fonction permettant de créer la color-Map de la Vtable
#############################################################
def CreationColorMapVtable(image, triangles, resolution, Vtable, EtatCible):
    # Normalise la Vtable entre 0 et 255
    valeurs = np.interp(Vtable, (np.amin(Vtable), np.amax(Vtable)), (0,255))
    valeurs = valeurs.astype(int)

    img = image.copy()
    for i in range(0, triangles.shape[0]):
        x1 = int(triangles[i, 0, 0] / (1000 * resolution))
        x2 = int(triangles[i, 1, 0] / (1000 * resolution))
        x3 = int(triangles[i, 2, 0] / (1000 * resolution))

        y1 = int(triangles[i, 0, 1] / (1000 * resolution))
        y2 = int(triangles[i, 1, 1] / (1000 * resolution))
        y3 = int(triangles[i, 2, 1] / (1000 * resolution))

        pt1 = (x1,img.shape[0] - y1)
        pt2 = (x2,img.shape[0] - y2)
        pt3 = (x3,img.shape[0] - y3)

        img = cv2.polylines(img,[np.array([pt1,pt2,pt3])],isClosed=True,color=(0,0,255),thickness=1)
        img = cv2.fillPoly(img,[np.array([pt1,pt2,pt3])],color=(0,int(valeurs[i]),0))

        # Affiche l'objectif
        centerX = int(triangles[EtatCible, 3, 0] / (1000 * resolution))
        centerY = int(triangles[EtatCible, 3, 1] / (1000 * resolution))
        img = cv2.circle(img, (centerX,img.shape[0]-centerY), radius=2, color=(255,0,0), thickness=2)

    for i in range(0, triangles.shape[0]):
        x1 = int(triangles[i, 0, 0] / (1000 * resolution))
        x2 = int(triangles[i, 1, 0] / (1000 * resolution))
        x3 = int(triangles[i, 2, 0] / (1000 * resolution))

        y1 = int(triangles[i, 0, 1] / (1000 * resolution))
        y2 = int(triangles[i, 1, 1] / (1000 * resolution))
        y3 = int(triangles[i, 2, 1] / (1000 * resolution))

        img = cv2.line(img, (x1, img.shape[0] - y1), (x2, img.shape[0] - y2), color=(0, 0, 255), thickness=1)
        img = cv2.line(img, (x2, img.shape[0] - y2), (x3, img.shape[0] - y3), color=(0, 0, 255), thickness=1)
        img = cv2.line(img, (x3, img.shape[0] - y3), (x1, img.shape[0] - y1), color=(0, 0, 255), thickness=1)
    return img, valeurs

#############################################################
# Fonction permettant de tracer la trajectoire sur l'image
#############################################################
def AfficheTrajectoireSurImage(triangles, image, res, etat_courant, etat_suivant):
    img = image.copy()

    pos_ETAT_COURANT = triangles[etat_courant,3,:]
    pos_ETAT_SUIVANT = triangles[etat_suivant,3,:]

    x1 = int(pos_ETAT_COURANT[0]/(1000*res))
    x2 = int(pos_ETAT_SUIVANT[0]/(1000*res))
    y1 = int(pos_ETAT_COURANT[1]/(1000*res))
    y2 = int(pos_ETAT_SUIVANT[1]/(1000*res))

    centerX = int(triangles[etat_suivant,3,0]/(1000*res))
    centerY = int(triangles[etat_suivant,3,1]/(1000*res))
    img = cv2.line(img, (x1,img.shape[0]-y1),(x2,img.shape[0]-y2), color=(0x1C,0x8E,0x00), thickness=2)
    img_temp = img.copy()

    img = cv2.circle(img, (centerX,img.shape[0]-centerY), radius=2, color=(255,0,0), thickness=2)

    return img, img_temp


#############################################################
# Fonction pour convertir le format OpenCV -> IPywidget Image
#############################################################
def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

class Maneges:
    def __init__(self, parc, type, ouvert, vitesse):
        self.parc = parc
        self.type = type
        self.ouvert = ouvert
        vitesse = str(vitesse)
        vitesse = vitesse.strip('\n')
        self.vitesse = vitesse




    def __str__(self):
        return "{}, {}, {}, {}".format(self.parc, self.type, self.ouvert, self.vitesse)

class RegistreManeges:
    def __init__(self):
        self.registre = []

    def ajouter(self, data):
        self.registre.append(data)

    def afficher(self):
        for info in self.registre:
            print(info)
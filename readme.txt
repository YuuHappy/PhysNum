fonction densité de probabilité autour des défenseurs (on pourrait considérer que c'est genre un nuage électronique autour d'un ion/atome)
Entrées:
    Taille des bras (??)
    skills de def (?)
Sotie:
    fonction densité

Fonction interaction (monte carlo)
Entrée:
    fonction densité de probabilité
    position et taille de l'objet (ballon ou joueur offensif)
Sortie:
    Vrai (sans interaction)
    Faux (interaction)

classe couverture défensive
Contient:
    défenseurs - position
    déplacements selon types de couvetuer
Méthode: déplacement des joueurs (dans le temps)

classe formation offensive
Contient:
    défenseurs - position
    déplacements selon types de jeux
Méthode: déplacement des joueurs (dans le temps)


Après, on veut rouler des jeux pleins de fois pour avoir des stats lié au gain moyen
On pourrait(devrait) aussi inclure le fait que si le def interagit avec le ballon avant qu'il se rende au joueur ça marche pas pour les passes.
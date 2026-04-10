"""
Explications en langage clair pour le chiffre estimé, selon l’unité UNICEF (UNIT_MEASURE).
"""

# Libellés courts pour l’interface (codes du jeu de données)
UNIT_CAPTION_FR = {
    "D": "Nombre de décès (comptage, pas un %)",
    "D_PER_1000_B": "Décès pour 1 000 naissances vivantes (taux)",
    "D_PER_1000_1": "Taux pour 1 000 (code UNICEF, tranche liée à l’âge)",
    "D_PER_1000_5": "Taux pour 1 000 (code UNICEF, tranche liée à l’âge)",
    "D_PER_1000_10": "Taux pour 1 000 (code UNICEF, tranche liée à l’âge)",
    "D_PER_1000_15": "Taux pour 1 000 (code UNICEF, tranche liée à l’âge)",
    "D_PER_1000_20": "Taux pour 1 000 (code UNICEF, tranche liée à l’âge)",
}


def unit_caption_fr(unit_measure: str | None) -> str:
    u = (unit_measure or "").strip().upper()
    return UNIT_CAPTION_FR.get(u, "Voir la documentation de l’indicateur pour l’unité exacte.")


def explain_estimate_fr(
    unit_measure: str | None,
    indicator_label: str | None,
    value: float,
) -> str:
    """
    Retourne un court texte expliquant ce que signifie le nombre affiché pour l’utilisateur.
    """
    unit = (unit_measure or "").strip().upper()
    ind = (indicator_label or "").strip()

    try:
        s_short = ("{:.2f}".format(value)).replace(".", ",")
    except Exception:
        s_short = str(value)

    base = f"Ce chiffre ({s_short}) est une estimation pour votre sélection"
    if ind:
        base += f" (indicateur: {ind})"
    base += ". "

    if unit == "D":
        return (
            base
            + "Ici, l'unité signifie un nombre de décès. "
            "Ce n'est pas un pourcentage."
        )

    if unit == "D_PER_1000_B":
        return (
            base
            + f"Ici, l'unité est un taux: environ {s_short} décès pour 1 000 naissances vivantes. "
            "Ce n'est pas un pourcentage direct."
        )

    if unit.startswith("D_PER_1000"):
        return (
            base
            + "Ici, l'unité correspond a un taux pour 1 000. "
            "Ce n'est pas un total de décès."
        )

    return (
        base
        + "Pour bien l'interpreter, regardez l'unité et l'indicateur juste en dessous."
    )

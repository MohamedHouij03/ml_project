from django import forms

# Libellés en français pour les colonnes UNICEF
FIELD_LABELS = {
    "INDICATOR": "Code indicateur",
    "Indicator": "Indicateur",
    "SEX": "Sexe de l'enfant",
    "SERIES_NAME": "Type de série",
    "SERIES_YEAR": "Année de la série",
    "TIME_PERIOD": "Période étudiée",
    "OBS_STATUS": "Statut des données",
    "UNIT_MEASURE": "Unité d'affichage",
    "REF_DATE": "Année de référence",
    "MODEL": "Modèle d’estimation",
}


SEX_LABELS_FR = {
    "F": "Féminin",
    "M": "Masculin",
    "B": "Les deux sexes",
    "T": "Total",
}


OBS_STATUS_LABELS_FR = {
    "A": "Officiel",
    "E": "Estimé",
    "F": "Prévision",
    "M": "Manquant",
    "P": "Provisoire",
}


UNIT_MEASURE_LABELS_FR = {
    "D": "Deces (nombre)",
    "D_PER_1000_1": "Deces pour 1 000 naissances vivantes - age < 1 an",
    "D_PER_1000_5": "Deces pour 1 000 naissances vivantes - moins de 5 ans",
    "D_PER_1000_10": "Deces pour 1 000 naissances vivantes - 5 a 9 ans",
    "D_PER_1000_15": "Deces pour 1 000 naissances vivantes - 10 a 14 ans",
    "D_PER_1000_20": "Deces pour 1 000 naissances vivantes - 15 a 19 ans",
    "D_PER_1000_B": "Deces pour 1 000 naissances vivantes - tous ages",
}


INDICATOR_LABELS_FR = {
    "Child Mortality rate age 1-4": "Taux de mortalite des enfants de 1 a 4 ans",
    "Child deaths age 1 to 4": "Nombre de deces des enfants de 1 a 4 ans",
    "Deaths age 10 to 14": "Nombre de deces de 10 a 14 ans",
    "Deaths age 10 to 19": "Nombre de deces de 10 a 19 ans",
    "Deaths age 15 to 19": "Nombre de deces de 15 a 19 ans",
    "Deaths age 15 to 24": "Nombre de deces de 15 a 24 ans",
    "Deaths age 20 to 24": "Nombre de deces de 20 a 24 ans",
    "Deaths age 5 to 14": "Nombre de deces de 5 a 14 ans",
    "Deaths age 5 to 24": "Nombre de deces de 5 a 24 ans",
    "Deaths age 5 to 9": "Nombre de deces de 5 a 9 ans",
    "Infant deaths": "Nombre de deces infantiles",
    "Infant mortality rate": "Taux de mortalite infantile",
    "Mortality rate age 10-14": "Taux de mortalite de 10 a 14 ans",
    "Mortality rate age 10-19": "Taux de mortalite de 10 a 19 ans",
    "Mortality rate age 15-19": "Taux de mortalite de 15 a 19 ans",
    "Mortality rate age 15-24": "Taux de mortalite de 15 a 24 ans",
    "Mortality rate age 20-24": "Taux de mortalite de 20 a 24 ans",
    "Mortality rate age 5-14": "Taux de mortalite de 5 a 14 ans",
    "Mortality rate age 5-24": "Taux de mortalite de 5 a 24 ans",
    "Mortality rate age 5-9": "Taux de mortalite de 5 a 9 ans",
    "Under-five deaths": "Nombre de deces des moins de 5 ans",
    "Under-five mortality rate": "Taux de mortalite des moins de 5 ans",
}


UNIT_TEXT_REPLACEMENTS_FR = {
    "per 1,000 live births": "pour 1 000 naissances vivantes",
    "live births": "naissances vivantes",
    "deaths": "décès",
    "rate": "taux",
    "number": "nombre",
}


def _choice_label_fr(column_name, raw_value):
    text = str(raw_value)

    if column_name == "Indicator":
        return INDICATOR_LABELS_FR.get(text, text)

    if column_name == "SEX":
        return SEX_LABELS_FR.get(text, text)

    if column_name == "OBS_STATUS":
        if text in OBS_STATUS_LABELS_FR:
            return f"{text} - {OBS_STATUS_LABELS_FR[text]}"
        return text

    if column_name == "UNIT_MEASURE":
        if text in UNIT_MEASURE_LABELS_FR:
            return f"{UNIT_MEASURE_LABELS_FR[text]} ({text})"
        lowered = text.lower()
        for source, target in UNIT_TEXT_REPLACEMENTS_FR.items():
            lowered = lowered.replace(source, target)
        return lowered.capitalize()

    return text


class PredictionForm(forms.Form):
    """
    Builds the prediction form dynamically from ML metadata.
    Categorical fields use select dropdowns when options are available.
    """

    def __init__(
        self,
        *args,
        feature_columns=None,
        numeric_features=None,
        categorical_options=None,
        ref_date_bounds=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        feature_columns = feature_columns or []
        numeric_features = set(numeric_features or [])
        categorical_options = categorical_options or {}
        ref_date_bounds = ref_date_bounds or {}

        input_cls = "field-input"
        select_cls = "field-select"

        priority = {
            "Indicator": 0,
            "TIME_PERIOD": 1,
            "SEX": 2,
            "UNIT_MEASURE": 3,
        }
        ordered_columns = sorted(
            feature_columns,
            key=lambda c: (priority.get(c, 99), feature_columns.index(c)),
        )

        for col in ordered_columns:
            label = FIELD_LABELS.get(col, col.replace("_", " ").title())

            if col == "REF_DATE":
                attrs = {
                    "class": input_cls,
                    "step": "1",
                    "inputmode": "numeric",
                }
                if ref_date_bounds:
                    if "min" in ref_date_bounds:
                        attrs["min"] = int(float(ref_date_bounds["min"]))
                    if "max" in ref_date_bounds:
                        attrs["max"] = int(float(ref_date_bounds["max"]))
                self.fields[col] = forms.IntegerField(
                    required=True,
                    label=label,
                    widget=forms.NumberInput(attrs=attrs),
                )
            elif col in numeric_features:
                attrs = {
                    "class": input_cls,
                    "step": "any",
                    "inputmode": "decimal",
                }
                self.fields[col] = forms.FloatField(
                    required=True,
                    label=label,
                    widget=forms.NumberInput(attrs=attrs),
                )
            elif col in categorical_options and categorical_options[col]:
                choices = [("", "— Choisir une option —")]
                for v in categorical_options[col]:
                    s = str(v)
                    choices.append((s, _choice_label_fr(col, s)))
                self.fields[col] = forms.ChoiceField(
                    required=True,
                    label=label,
                    choices=choices,
                    widget=forms.Select(attrs={"class": select_cls}),
                )
            else:
                self.fields[col] = forms.CharField(
                    required=True,
                    label=label,
                    max_length=512,
                    widget=forms.TextInput(attrs={"class": input_cls}),
                )

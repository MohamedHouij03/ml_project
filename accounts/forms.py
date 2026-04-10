from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
    """Formulaire d’inscription avec libellés en français."""

    class Meta:
        model = User
        fields = ("username", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].label = "Nom d’utilisateur"
        self.fields["username"].help_text = (
            "Lettres, chiffres et @/./+/-/_ uniquement. 150 caractères max."
        )
        self.fields["password1"].label = "Mot de passe"
        self.fields["password2"].label = "Confirmation du mot de passe"
        # Sans validateurs globaux, on retire le long texte d’aide sur la complexité du mot de passe.
        self.fields["password1"].help_text = "Choisissez un mot de passe (aucune contrainte imposée)."
        self.fields["password2"].help_text = "Saisissez le même mot de passe pour confirmation."

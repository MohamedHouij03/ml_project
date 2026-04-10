from django.contrib import messages
from django.contrib.auth.views import LoginView
from django.urls import reverse_lazy
from django.views.generic import CreateView

from .forms import SignUpForm


class SignUpView(CreateView):
    form_class = SignUpForm
    template_name = "accounts/signup.html"
    success_url = reverse_lazy("login")

    def form_valid(self, form):
        messages.success(
            self.request,
            "Compte créé avec succès. Vous pouvez maintenant vous connecter.",
        )
        return super().form_valid(form)


class AppLoginView(LoginView):
    template_name = "accounts/login.html"
    redirect_authenticated_user = True

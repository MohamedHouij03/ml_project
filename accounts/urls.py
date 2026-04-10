from django.contrib.auth.views import LogoutView
from django.urls import path

from .views import AppLoginView, SignUpView

urlpatterns = [
    path("login/", AppLoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("signup/", SignUpView.as_view(), name="signup"),
]

from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("predict/", views.predict_view, name="predict"),
    path("diagrams/", views.diagrams, name="diagrams"),
    path("charts/<str:chart_name>/", views.chart_image, name="chart_image"),
    path("api/predict/", views.api_predict, name="api_predict"),
]

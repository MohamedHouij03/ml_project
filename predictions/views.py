import json
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from .charts import CHART_REGISTRY, get_chart_png
from .forms import PredictionForm
from .interpretation import explain_estimate_fr, unit_caption_fr
from .ml_loader import load_ml_artifacts


def home(request):
    return render(request, "predictions/home.html")


@login_required
def predict_view(request):
    model, metadata = load_ml_artifacts()

    feature_columns = metadata["feature_columns"]
    numeric_features = metadata.get("numeric_features", [])

    prediction = None
    prediction_display_fr = None
    error = None
    prediction_context = None
    prediction_explanation = None

    categorical_options = metadata.get("categorical_options") or {}
    ref_date_bounds = metadata.get("ref_date_bounds") or {}

    if request.method == "POST":
        form = PredictionForm(
            request.POST,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_options=categorical_options,
            ref_date_bounds=ref_date_bounds,
        )
        if form.is_valid():
            try:
                row = {col: form.cleaned_data[col] for col in feature_columns}
                X_input = pd.DataFrame([row], columns=feature_columns)
                y_pred = model.predict(X_input)[0]
                prediction = float(y_pred)
                prediction_context = {
                    "indicator_label": row.get("Indicator"),
                    "unit_measure": row.get("UNIT_MEASURE"),
                    "unit_caption_fr": unit_caption_fr(row.get("UNIT_MEASURE")),
                    "ref_date": row.get("REF_DATE"),
                    "sex": row.get("SEX"),
                    "series_name": row.get("SERIES_NAME"),
                }
                prediction_explanation = explain_estimate_fr(
                    row.get("UNIT_MEASURE"),
                    row.get("Indicator"),
                    prediction,
                )
                if str(row.get("UNIT_MEASURE", "")).strip().upper() == "D":
                    rounded_count = int(
                        Decimal(str(prediction)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
                    )
                    prediction_display_fr = str(rounded_count)
                else:
                    prediction_display_fr = "{:.2f}".format(prediction).replace(".", ",")
            except Exception as e:  # noqa: BLE001
                error = f"Échec de la prédiction : {type(e).__name__} : {e}"
        else:
            error = "Veuillez corriger les erreurs du formulaire et réessayer."
    else:
        form = PredictionForm(
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_options=categorical_options,
            ref_date_bounds=ref_date_bounds,
        )

    return render(
        request,
        "predictions/index.html",
        {
            "form": form,
            "prediction": prediction,
            "prediction_display_fr": prediction_display_fr,
            "prediction_context": prediction_context,
            "prediction_explanation": prediction_explanation,
            "error": error,
            "model_name": metadata.get("best_model_name", "ML model"),
            "metrics": metadata.get("results", {}),
        },
    )


@login_required
def diagrams(request):
    charts = [{"slug": k, "title": v[0]} for k, v in CHART_REGISTRY.items()]
    return render(
        request,
        "predictions/diagrams.html",
        {"charts": charts},
    )


@login_required
def chart_image(request, chart_name: str):
    try:
        png = get_chart_png(chart_name)
    except KeyError as e:
        raise Http404("Graphique inconnu.") from e
    return HttpResponse(png, content_type="image/png")


@login_required
def api_predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Seule la méthode POST est acceptée."}, status=405)

    model, metadata = load_ml_artifacts()
    feature_columns = metadata["feature_columns"]

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Corps JSON invalide."}, status=400)

    missing = [c for c in feature_columns if c not in payload]
    if missing:
        return JsonResponse(
            {"error": "Champs manquants.", "missing": missing},
            status=400,
        )

    row = {c: payload[c] for c in feature_columns}
    
    # Ensure numeric types are correctly cast for the Scikit-learn pipeline
    numeric_features = metadata.get("numeric_features", [])
    for col in numeric_features:
        if col in row:
            try:
                row[col] = float(row[col])
            except (ValueError, TypeError):
                return JsonResponse({"error": f"Le champ {col} doit être numérique."}, status=400)

    X_input = pd.DataFrame([row], columns=feature_columns)

    y_pred = model.predict(X_input)[0]
    return JsonResponse({"prediction": float(y_pred)})

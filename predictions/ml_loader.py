import json
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# Cache by file mtimes so metadata refreshes without a server restart
# when you re-run build_model.py or python manage.py train_model
_cached = None  # tuple: (model_mtime, meta_mtime, model, metadata)


def load_ml_artifacts():
    """
    Loads the saved sklearn pipeline and metadata required to build the form.
    Reloads when artifact files on disk change.
    
    Returns:
        tuple: (model, metadata)
        
    Raises:
        FileNotFoundError: If model or metadata artifacts are missing.
    """

    global _cached

    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "ml_artifacts"

    model_path = artifacts_dir / "best_model.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing ML model artifact: {model_path}. "
            f"Run 'python manage.py train_model' first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing ML metadata artifact: {metadata_path}. "
            f"Run 'python manage.py train_model' first."
        )

    model_mtime = model_path.stat().st_mtime
    meta_mtime = metadata_path.stat().st_mtime

    # Return cached version if files haven't changed
    if _cached and _cached[0] == model_mtime and _cached[1] == meta_mtime:
        return _cached[2], _cached[3]

    # Load fresh artifacts
    logger.info(f"Loading ML model and metadata from {artifacts_dir}")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    
    _cached = (model_mtime, meta_mtime, model, metadata)
    return model, metadata


def clear_cache():
    """Force reload on next load_ml_artifacts() call."""
    global _cached
    _cached = None

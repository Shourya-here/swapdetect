"""Quality-aware fusion: the ONLY trained model. Pretrained face/liveness/voice models stay frozen.

Output is a calibrated-ish log-likelihood ratio (genuine vs impostor) per check, usable by CUSUM.
Class-balanced training => equal priors => LLR = logit(p).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURE_SETS = {
    "full":       ["face", "live", "voice", "has_voice", "qual", "face_x_qual", "voice_x_qual"],
    "no_quality": ["face", "live", "voice", "has_voice"],
    "face_only":  ["face"],
    "fail_flag":  ["face", "live", "voice", "has_voice", "face_found"],   # binary failure flag, no continuous quality
}


def build_features(X, names):
    face, live, voice, qual = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    has = ~np.isnan(voice)
    v = np.where(has, voice, 0.0)
    allf = {"face": face, "live": live, "voice": v, "has_voice": has.astype(float),
            "qual": qual, "face_x_qual": face * qual, "voice_x_qual": v * qual,
            "face_found": (qual > 0).astype(float)}
    return np.column_stack([allf[n] for n in names])


class Fusion:
    def __init__(self, feature_set="full", clip=6.0):
        self.names = FEATURE_SETS[feature_set]
        self.clip = clip
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(class_weight="balanced", max_iter=1000))

    def fit(self, X_gen, X_imp):
        X = np.vstack([X_gen, X_imp])
        y = np.r_[np.ones(len(X_gen)), np.zeros(len(X_imp))]
        self.model.fit(build_features(X, self.names), y)
        return self

    def llr(self, X):
        """Log-likelihood ratio of 'genuine' vs 'impostor' per check (positive = genuine)."""
        s = self.model.decision_function(build_features(X, self.names))
        return np.clip(s, -self.clip, self.clip)

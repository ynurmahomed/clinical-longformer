from clinical_longformer.model.metrics import ClinicalBERTBinnedPRCurve


def test_clinicalbert_binned_auc_pr_init():
    metric = ClinicalBERTBinnedPRCurve()

    assert metric.c == 2
    assert metric.hadm_id == []
    assert metric.preds == []
    assert metric.target == []

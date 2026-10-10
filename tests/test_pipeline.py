import unittest

import pandas as pd

from src.config import TARGET_COLUMNS
from src.data_split import patient_group_split
from src.target_definition import define_targets


class PipelineTests(unittest.TestCase):
    def test_define_targets_returns_binary_columns(self):
        df = pd.DataFrame(
            {
                "lab_test": [0, 1, 0],
                "imaging": [1, 0, 1],
                "instrument_test": [0, 1, 0],
                "medical_exam": [1, 0, 0],
                "patient_id": ["p1", "p1", "p2"],
            }
        )

        y = define_targets(df)

        self.assertEqual(list(y.columns), TARGET_COLUMNS)
        self.assertIn(y["lab_test"].dtype.kind, {"i", "u", "b"})
        self.assertEqual(y.shape, (3, len(TARGET_COLUMNS)))

    def test_patient_group_split_keeps_patients_out_of_both_sets(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
                "feature_a": [1, 2, 3, 4, 5, 6],
                "lab_test": [0, 1, 0, 1, 0, 1],
            }
        )

        train_df, test_df = patient_group_split(df, target_col="lab_test", test_size=0.5)

        self.assertTrue(set(train_df["patient_id"]).isdisjoint(set(test_df["patient_id"])))
        self.assertEqual(len(set(train_df["patient_id"]) | set(test_df["patient_id"])), 3)

    def test_patient_group_split_stratifies_target_positive_patients(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
                "medical_exam": [1, 0, 1, 0, 0, 0, 0, 0],
            }
        )

        train_df, test_df = patient_group_split(
            df,
            target_col="medical_exam",
            test_size=0.5,
        )

        self.assertGreater(train_df["medical_exam"].sum(), 0)
        self.assertGreater(test_df["medical_exam"].sum(), 0)

    def test_patient_group_split_balances_all_targets(self):
        rows = []
        for patient_index in range(20):
            row = {
                "patient_id": f"p{patient_index}",
                "medical_exam": int(patient_index % 2 == 0),
                "lab_test": int(patient_index % 3 == 0),
                "imaging": int(patient_index % 4 == 0),
                "instrument_test": int(patient_index % 5 == 0),
            }
            rows.append(row)
        df = pd.DataFrame(rows)

        train_df, test_df = patient_group_split(
            df,
            target_col="medical_exam",
            test_size=0.25,
        )

        for target in TARGET_COLUMNS:
            full_rate = df[target].mean()
            test_rate = test_df[target].mean()
            self.assertLess(abs(full_rate - test_rate), 0.2)

    def test_model_table_has_recent_history_features(self):
        from src.build_model_table import build_table

        df = build_table()

        self.assertIn("patient_total_encounters", df.columns)
        self.assertNotIn("patient_last_encounter", df.columns)
        first_encounters = df.groupby("patient_id", sort=False).head(1)
        self.assertTrue((first_encounters["patient_total_encounters"] == 0).all())
        self.assertTrue((first_encounters["days_since_last_encounter"] == -1).all())
        self.assertIn("prior_conditions_count_7d", df.columns)
        self.assertIn("prior_conditions_count_30d", df.columns)
        self.assertIn("days_since_conditions_last", df.columns)
        self.assertIn("prior_procedures_count_90d", df.columns)
        self.assertIn("prior_procedures_count_180d", df.columns)
        self.assertNotIn("screening", df.columns)
        self.assertNotIn("patient_total_cost", df.columns)
        self.assertNotIn("RACE", df.columns)
        self.assertNotIn("ETHNICITY", df.columns)
        self.assertIn("screening_prior_height", df.columns)
        self.assertIn("screening_prior_weight", df.columns)
        self.assertIn("screening_prior_bmi", df.columns)
        self.assertIn("screening_prior_bmi_percentile", df.columns)
        self.assertIn("screening_prior_measurement_count", df.columns)
        self.assertEqual(list(df.columns[-len(TARGET_COLUMNS):]), TARGET_COLUMNS)

    def test_workflow_events_use_procedures_for_instrument_tests(self):
        from src.build_model_table import make_workflow_events

        events = make_workflow_events()

        self.assertNotIn("screening", events)
        self.assertGreater(len(events["instrument_test"]), 0)
        self.assertTrue(events["instrument_test"]["DESCRIPTION"].notna().all())

    def test_procedure_mapping_classifies_representative_descriptions(self):
        from src.procedure_mapping import classify_procedure

        self.assertEqual(classify_procedure("Spirometry (procedure)")[0], "instrument_test")
        self.assertEqual(classify_procedure("Hemoglobin / Hematocrit / Platelet count")[0], "lab_test")
        self.assertEqual(classify_procedure("Physical examination")[0], "medical_exam")
        self.assertEqual(classify_procedure("Medication Reconciliation (procedure)")[0], "other")

    def test_linear_svm_model_is_supported(self):
        from src.train_baselines import make_base_estimator

        estimator = make_base_estimator("linear_svm")
        self.assertTrue(hasattr(estimator, "decision_function"))

    def test_saved_model_artifact_produces_workflow_predictions(self):
        import tempfile
        from pathlib import Path

        import joblib

        from src.evaluate_models import tune_thresholds
        from src.model_artifact import load_model_artifact, predict_workflows
        from src.preprocess import build_preprocessor
        from src.train_baselines import train_model

        features = pd.DataFrame(
            {
                "age_at_encounter": [20, 30, 40, 50, 60, 70, 25, 55],
                "GENDER": ["F", "M", "F", "M", "F", "M", "F", "M"],
            }
        )
        targets = pd.DataFrame(
            {
                target: [0, 1, 0, 1, 0, 1, 0, 1]
                for target in TARGET_COLUMNS
            }
        )
        feature_columns = list(features.columns)
        preprocessor = build_preprocessor(features, feature_columns)
        transformed = preprocessor.fit_transform(features)
        model = train_model("random_forest", transformed, targets)
        thresholds = tune_thresholds(
            targets,
            [estimator.predict_proba(transformed)[:, 1] for estimator in model.estimators_],
            0.5,
        )
        artifact = {
            "schema_version": 1,
            "feature_columns": feature_columns,
            "target_columns": TARGET_COLUMNS,
            "thresholds": thresholds,
            "preprocessor": preprocessor,
            "model": model,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "model.joblib"
            joblib.dump(artifact, artifact_path)
            loaded = load_model_artifact(artifact_path)
            predictions = predict_workflows(features.iloc[:2], loaded)

        self.assertEqual(len(predictions), 2)
        self.assertIn("medical_exam_score", predictions.columns)
        self.assertIn("medical_exam_predicted", predictions.columns)
        self.assertTrue(predictions["medical_exam_predicted"].isin([True, False]).all())

    def test_saved_model_rejects_missing_features(self):
        from src.model_artifact import predict_workflows

        artifact = {"feature_columns": ["age_at_encounter"]}
        with self.assertRaisesRegex(ValueError, "Missing required model features"):
            predict_workflows(pd.DataFrame({"GENDER": ["F"]}), artifact)

    def test_model_comparison_uses_supported_metrics(self):
        from src.compare_models import compare_models

        results = {
            "logistic_regression": pd.DataFrame(
                {"f1": [0.4], "precision": [0.5], "recall": [0.3]}
            )
        }
        summary = compare_models(results)
        self.assertEqual(list(summary.columns), ["model", "mean_f1", "mean_precision", "mean_recall"])

    def test_manual_history_records_generate_lookback_and_recency_features(self):
        from datetime import date, timedelta

        from src.manual_features import RECORD_TYPE_TO_FEATURE, build_model_features

        evaluation_date = date(2025, 6, 1)
        records = [
            {"Record type": "Condition", "Record date": evaluation_date - timedelta(days=7)},
            {"Record type": "Condition", "Record date": evaluation_date - timedelta(days=20)},
            {"Record type": "Condition", "Record date": evaluation_date - timedelta(days=200)},
            {"Record type": "Condition", "Record date": evaluation_date},
            {"Record type": "Encounter", "Record date": evaluation_date - timedelta(days=15)},
            {"Record type": "Encounter", "Record date": evaluation_date + timedelta(days=1)},
        ]
        current_features = {
            "GENDER": "F",
            "age_at_encounter": 40.0,
            "ENCOUNTERCLASS": "outpatient",
            "encounter_code": "DEMO",
            "reason_code": "NO_REASON_RECORDED",
        }
        feature_columns = list(current_features)
        feature_columns.extend(["patient_total_encounters", "days_since_last_encounter"])
        for feature_type in RECORD_TYPE_TO_FEATURE.values():
            feature_columns.append(f"days_since_{feature_type}_last")
            feature_columns.extend(
                f"prior_{feature_type}_count_{window}d"
                for window in [7, 30, 90, 180]
            )

        features = build_model_features(
            current_features,
            records,
            evaluation_date,
            feature_columns,
        )

        self.assertEqual(features["prior_conditions_count_7d"], 1.0)
        self.assertEqual(features["prior_conditions_count_30d"], 2.0)
        self.assertEqual(features["prior_conditions_count_180d"], 2.0)
        self.assertEqual(features["days_since_conditions_last"], 7)
        self.assertEqual(features["patient_total_encounters"], 1)
        self.assertEqual(features["days_since_last_encounter"], 15)
        self.assertEqual(features["prior_allergies_count_30d"], 0.0)
        self.assertEqual(features["days_since_allergies_last"], -1)

    def test_history_agent_returns_reviewable_records_with_source_and_model(self):
        from datetime import date
        from types import SimpleNamespace

        from src.history_agent import (
            ExtractedHistoryRecord,
            HistoryExtraction,
            extract_history_records,
        )

        class FakeResponses:
            def parse(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    output_parsed=HistoryExtraction(
                        records=[
                            ExtractedHistoryRecord(
                                record_type="Condition",
                                record_date=date(2022, 3, 14),
                                details="asthma",
                                source_text="Asthma was diagnosed on 2022-03-14.",
                                confidence=0.98,
                            )
                        ],
                        clarifying_questions=[],
                    )
                )

        responses = FakeResponses()
        result = extract_history_records(
            "Asthma was diagnosed on 2022-03-14.",
            client=SimpleNamespace(responses=responses),
            model="test-extraction-model",
        )

        self.assertEqual(result["model"], "test-extraction-model")
        self.assertEqual(result["records"][0]["Record type"], "Condition")
        self.assertEqual(result["records"][0]["Record date"], date(2022, 3, 14))
        self.assertIn("2022-03-14", result["records"][0]["Source text"])
        self.assertEqual(responses.kwargs["model"], "test-extraction-model")
        self.assertIs(responses.kwargs["text_format"], HistoryExtraction)
        self.assertEqual(responses.kwargs["temperature"], 0)

    def test_history_agent_prompt_names_every_record_type_and_the_evaluation_date(self):
        from datetime import date

        from src.history_agent import SYSTEM_PROMPT
        from src.manual_features import RECORD_TYPES

        prompt = SYSTEM_PROMPT.format(evaluation_date=date(2026, 10, 10))
        for record_type in RECORD_TYPES:
            self.assertIn(f"- {record_type}:", prompt)
        self.assertIn("2026-10-10", prompt)

    def test_history_agent_rejects_unverifiable_source_quote(self):
        from datetime import date
        from types import SimpleNamespace

        from src.history_agent import ExtractedHistoryRecord, HistoryExtraction, extract_history_records

        fake_response = SimpleNamespace(
            output_parsed=HistoryExtraction(
                records=[
                    ExtractedHistoryRecord(
                        record_type="Condition",
                        record_date=date(2022, 3, 14),
                        details="asthma",
                        source_text="COPD diagnosed in 2019.",
                        confidence=0.9,
                    )
                ],
                clarifying_questions=[],
            )
        )
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(parse=lambda **kwargs: fake_response)
        )

        with self.assertRaisesRegex(ValueError, "source quote"):
            extract_history_records(
                "Asthma was diagnosed.",
                client=fake_client,
            )

    def test_history_agent_accepts_quote_with_different_spacing_and_retries_once(self):
        from datetime import date
        from types import SimpleNamespace

        from src.history_agent import ExtractedHistoryRecord, HistoryExtraction, extract_history_records

        def response(quote):
            return SimpleNamespace(
                output_parsed=HistoryExtraction(
                    records=[
                        ExtractedHistoryRecord(
                            record_type="Condition",
                            record_date=date(2022, 3, 14),
                            details="asthma",
                            source_text=quote,
                            confidence=0.9,
                        )
                    ],
                    clarifying_questions=[],
                )
            )

        replies = iter([response("COPD was diagnosed."), response("asthma was diagnosed on 2022-03-14.")])
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(parse=lambda **kwargs: next(replies))
        )

        result = extract_history_records(
            "Asthma was\ndiagnosed on 2022-03-14.",
            client=fake_client,
        )

        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(list(replies), [])

if __name__ == "__main__":
    unittest.main()

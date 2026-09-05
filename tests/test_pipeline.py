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

    def test_model_comparison_uses_supported_metrics(self):
        from src.compare_models import compare_models

        results = {
            "logistic_regression": pd.DataFrame(
                {"f1": [0.4], "precision": [0.5], "recall": [0.3]}
            )
        }
        summary = compare_models(results)
        self.assertEqual(list(summary.columns), ["model", "mean_f1", "mean_precision", "mean_recall"])


if __name__ == "__main__":
    unittest.main()

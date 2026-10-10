"""Drive the Streamlit app headlessly with each available model."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.config import MODEL_ARTIFACT_PATH

APP = str(Path(__file__).resolve().parents[1] / "app.py")

try:
    from workflow_ml.serving import BUNDLE_PATH
except ImportError:
    BUNDLE_PATH = None


def open_page(model_prefix: str, page: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180).run()
    model = next(option for option in app.sidebar.radio[1].options if option.startswith(model_prefix))
    app.sidebar.radio[1].set_value(model)
    app.sidebar.radio[0].set_value(page)
    return app.run()


def submit_manual_application(app: AppTest) -> AppTest:
    app.text_input[0].set_value("TEST-1")
    submit = next(button for button in app.button if button.proto.is_form_submitter)
    return submit.click().run()


class AppTests(unittest.TestCase):
    @unittest.skipUnless(BUNDLE_PATH and BUNDLE_PATH.exists(), "ensemble bundle not exported")
    def test_ensemble_scores_a_held_out_patient_and_shows_the_outcome(self):
        app = open_page("Ensemble", "New application")
        next(button for button in app.button if button.key == "demo_generate").click().run()
        self.assertFalse(app.exception)

        case = app.session_state["applications"][0]
        self.assertEqual(len(case["predictions"]), 4)
        self.assertEqual(set(case["actual"]), set(case["predictions"]))
        self.assertTrue(any(result["proposed"] for result in case["predictions"].values()))

        app.sidebar.radio[0].set_value("Underwriter review")
        self.assertFalse(app.run().exception)

    @unittest.skipUnless(
        BUNDLE_PATH and BUNDLE_PATH.exists() and MODEL_ARTIFACT_PATH.exists(), "needs both models"
    )
    def test_original_model_scores_a_held_out_patient_next_to_the_ensemble(self):
        app = open_page("Original", "New application")
        next(button for button in app.button if button.key == "demo_generate").click().run()
        self.assertFalse(app.exception)

        case = app.session_state["applications"][0]
        self.assertTrue(case["model"].startswith("Original"))
        self.assertEqual(set(case["actual"]), set(case["predictions"]))
        (other_name, other_predictions), = case["comparison"].items()
        self.assertTrue(other_name.startswith("Ensemble"))
        self.assertEqual(set(other_predictions), set(case["predictions"]))

        app.sidebar.radio[0].set_value("Underwriter review")
        self.assertFalse(app.run().exception)

    @unittest.skipUnless(BUNDLE_PATH and BUNDLE_PATH.exists(), "ensemble bundle not exported")
    def test_ensemble_scores_a_manual_application(self):
        app = submit_manual_application(open_page("Ensemble", "New application"))
        self.assertFalse(app.exception)
        case = app.session_state["applications"][0]
        self.assertTrue(all(0 <= result["score"] <= 1 for result in case["predictions"].values()))

    def test_every_example_fills_the_form_with_valid_choices(self):
        from src.demo_examples import EXAMPLES

        app = AppTest.from_file(APP, default_timeout=180).run()
        app.sidebar.radio[0].set_value("New application")
        app.run()
        for name, example in EXAMPLES.items():
            app.selectbox(key="example").set_value(name).run()
            self.assertFalse(app.exception, name)
            self.assertEqual(app.session_state["medical_history_text"], example["history"])
            self.assertEqual(app.selectbox(key="visit_reason").value, example["reason"])
            self.assertTrue(app.selectbox(key="visit_type").value.startswith(example["visit"][0]))

        app.selectbox(key="example").set_value("Blank form").run()
        self.assertEqual(app.session_state["medical_history_text"], "")

    @unittest.skipUnless(MODEL_ARTIFACT_PATH.exists(), "original model not exported")
    def test_original_model_still_scores_a_manual_application(self):
        app = submit_manual_application(open_page("Original", "New application"))
        self.assertFalse(app.exception)
        case = app.session_state["applications"][0]
        self.assertTrue(case["model"].startswith("Original"))
        self.assertEqual(len(case["predictions"]), 4)


if __name__ == "__main__":
    unittest.main()

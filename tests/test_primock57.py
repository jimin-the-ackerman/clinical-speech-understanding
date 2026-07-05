from pathlib import Path

from stt_eval.datasets.primock57 import merge_reference

FIXTURES = Path(__file__).parent / "fixtures"


def test_merge_orders_by_time_and_strips_markup():
    ref = merge_reference(FIXTURES / "mini_doctor.TextGrid", FIXTURES / "mini_patient.TextGrid")
    assert ref == "Hello, how can I help? I have a headache. I see. Any fever? No fever."

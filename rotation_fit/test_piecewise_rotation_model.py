"""Stdlib tests for the deployable piecewise rotation response."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import FrozenInstanceError, replace

try:  # Supports both discovery from rotation_fit/ and import as a namespace package.
    from .piecewise_rotation_model import LogRotationModel
except ImportError:  # pragma: no cover - exercised by unittest discovery style
    from piecewise_rotation_model import LogRotationModel


def make_model(**changes) -> LogRotationModel:
    # a=5 deg/s^2 for 2 s -> max rate 10 deg/s
    values = {
        "dead_time_sec": 1.0,
        "accel_duration_sec": 2.0,
        "accel_deg_s2": 5.0,
        "max_rate_deg_s": 10.0,
        "inertia_intercept_deg": 1.0,
        "inertia_slope_sec": 0.2,
        # total(1.2)=0.1 command + 1.2 coast = 1.3 deg
        "fitted_min_hold_sec": 1.2,
        "fitted_max_hold_sec": 6.0,
        "fitted_min_angle_deg": 1.3,
        # total(6)=40 command + 3 coast = 43 deg
        "fitted_max_angle_deg": 43.0,
    }
    values.update(changes)
    return LogRotationModel(**values)


class ForwardModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()

    def test_model_and_plan_are_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            self.model.dead_time_sec = 2.0  # type: ignore[misc]
        plan = self.model.command_duration(4.5)
        with self.assertRaises(FrozenInstanceError):
            plan.hold_sec = 3.0  # type: ignore[misc]

    def test_dead_time_has_no_motion_or_coast(self):
        for hold in (0.0, 0.5, 1.0):
            with self.subTest(hold=hold):
                self.assertEqual(self.model.driven_duration(hold), 0.0)
                self.assertEqual(self.model.rate_at_command_end(hold), 0.0)
                self.assertEqual(self.model.angle_at_command_end(hold), 0.0)
                self.assertEqual(self.model.total_angle(hold), 0.0)

    def test_acceleration_and_cruise_boundaries(self):
        self.assertAlmostEqual(self.model.driven_duration(1.5), 0.5)
        self.assertAlmostEqual(self.model.rate_at_command_end(2.0), 5.0)
        self.assertAlmostEqual(self.model.angle_at_command_end(2.0), 2.5)

        accel_end = self.model.dead_time_sec + self.model.accel_duration_sec
        self.assertAlmostEqual(self.model.rate_at_command_end(accel_end), 10.0)
        self.assertAlmostEqual(self.model.angle_at_command_end(accel_end), 10.0)
        self.assertAlmostEqual(self.model.rate_at_command_end(4.0), 10.0)
        self.assertAlmostEqual(self.model.angle_at_command_end(4.0), 20.0)

    def test_response_is_continuous_at_phase_boundaries(self):
        eps = 1e-7
        dead = self.model.dead_time_sec
        accel_end = dead + self.model.accel_duration_sec
        self.assertLess(
            abs(
                self.model.angle_at_command_end(dead + eps)
                - self.model.angle_at_command_end(dead - eps)
            ),
            1e-5,
        )
        self.assertLess(
            abs(
                self.model.angle_at_command_end(accel_end + eps)
                - self.model.angle_at_command_end(accel_end - eps)
            ),
            1e-5,
        )
        self.assertLess(
            abs(
                self.model.rate_at_command_end(accel_end + eps)
                - self.model.rate_at_command_end(accel_end - eps)
            ),
            1e-5,
        )

    def test_inertia_uses_speed_magnitude_and_never_goes_negative(self):
        self.assertEqual(self.model.inertia_angle(0.0), 0.0)
        self.assertAlmostEqual(self.model.inertia_angle(5.0), 2.0)
        self.assertAlmostEqual(self.model.inertia_angle(-5.0), 2.0)
        negative_intercept = make_model(
            inertia_intercept_deg=-5.0,
            inertia_slope_sec=0.1,
        )
        self.assertEqual(negative_intercept.inertia_angle(10.0), 0.0)

    def test_total_angle_is_monotonic(self):
        holds = [index / 20.0 for index in range(0, 201)]
        angles = [self.model.total_angle(hold) for hold in holds]
        self.assertTrue(all(a <= b for a, b in zip(angles, angles[1:])))

    def test_invalid_runtime_inputs_are_rejected(self):
        for value in (-0.1,):
            with self.assertRaises(ValueError):
                self.model.total_angle(value)
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.model.command_duration(value)
        with self.assertRaises(TypeError):
            self.model.command_duration(10.0, allow_extrapolation=1)  # type: ignore[arg-type]


class InverseModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()

    def test_zero_target_needs_no_direction_or_command(self):
        plan = self.model.command_duration(0.0)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.sign, 0)
        self.assertEqual(plan.direction, "NONE")
        self.assertEqual(plan.hold_sec, 0.0)
        self.assertEqual(plan.predicted_total_angle_deg, 0.0)

    def test_inverse_recovers_acceleration_phase_hold(self):
        target = self.model.total_angle(2.0)
        self.assertAlmostEqual(target, 4.5)
        plan = self.model.command_duration(target)
        self.assertTrue(plan.feasible)
        self.assertFalse(plan.extrapolated)
        self.assertEqual(plan.direction, "RIGHT")
        self.assertEqual(plan.sign, 1)
        self.assertAlmostEqual(plan.hold_sec, 2.0, places=9)
        self.assertAlmostEqual(plan.predicted_total_angle_deg, target, places=9)

    def test_inverse_recovers_cruise_phase_hold_and_left_sign(self):
        target = self.model.total_angle(4.0)
        self.assertAlmostEqual(target, 23.0)
        plan = self.model.command_duration(-target)
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.direction, "LEFT")
        self.assertEqual(plan.sign, -1)
        self.assertAlmostEqual(plan.hold_sec, 4.0, places=9)
        self.assertAlmostEqual(plan.signed_predicted_total_deg, -target, places=9)

    def test_left_and_right_share_one_magnitude_model(self):
        right = self.model.command_duration(23.0)
        left = self.model.command_duration(-23.0)
        self.assertEqual(right.hold_sec, left.hold_sec)
        self.assertEqual(
            right.predicted_total_angle_deg,
            left.predicted_total_angle_deg,
        )
        self.assertEqual((right.direction, left.direction), ("RIGHT", "LEFT"))

    def test_default_mode_refuses_outside_fitted_angle_range(self):
        too_small = self.model.command_duration(1.0)
        self.assertFalse(too_small.feasible)
        self.assertEqual(too_small.hold_sec, 0.0)
        self.assertIn("below fitted", too_small.note)

        too_large = self.model.command_duration(50.0)
        self.assertFalse(too_large.feasible)
        self.assertEqual(too_large.hold_sec, self.model.fitted_max_hold_sec)
        self.assertIn("above fitted", too_large.note)

    def test_explicit_extrapolation_and_hard_hold_cap(self):
        target = self.model.total_angle(8.0)
        plan = self.model.command_duration(target, allow_extrapolation=True)
        self.assertTrue(plan.feasible)
        self.assertTrue(plan.extrapolated)
        self.assertAlmostEqual(plan.hold_sec, 8.0, places=9)

        capped = self.model.command_duration(
            target, allow_extrapolation=True, max_hold_sec=7.0,
        )
        self.assertFalse(capped.feasible)
        self.assertEqual(capped.hold_sec, 7.0)
        self.assertLess(capped.predicted_total_angle_deg, target)
        self.assertIn("max_hold_sec", capped.note)

    def test_positive_inertia_intercept_exposes_minimum_angle_gap(self):
        plan = self.model.command_duration(0.5, allow_extrapolation=True)
        self.assertFalse(plan.feasible)
        self.assertEqual(plan.hold_sec, 0.0)
        self.assertIn("minimum positive angle", plan.note)


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = make_model()

    def test_dict_shape_and_json_round_trip(self):
        artifact = self.model.to_dict()
        self.assertEqual(
            set(artifact),
            {"schema_version", "model_type", "parameters", "fit_metadata"},
        )
        self.assertEqual(artifact["parameters"]["command_strength"], 30)
        self.assertEqual(LogRotationModel.from_dict(artifact), self.model)
        self.assertEqual(LogRotationModel.from_json(self.model.to_json()), self.model)

    def test_fit_report_is_tolerated_but_other_top_level_keys_are_not(self):
        artifact = self.model.to_dict()
        artifact["fit_report"] = {
            "n_segments": 12,
            "metrics": {"rmse_deg": 0.4},
            "future_diagnostic": [1, 2, 3],
        }
        self.assertEqual(LogRotationModel.from_dict(artifact), self.model)
        artifact["unknown"] = True
        with self.assertRaisesRegex(ValueError, "unexpected unknown"):
            LogRotationModel.from_dict(artifact)

    def test_schema_model_strength_and_parameter_keys_are_strict(self):
        cases = []
        wrong_schema = self.model.to_dict()
        wrong_schema["schema_version"] = 2
        cases.append(wrong_schema)

        wrong_type = self.model.to_dict()
        wrong_type["model_type"] = "some_other_model"
        cases.append(wrong_type)

        wrong_strength = self.model.to_dict()
        wrong_strength["parameters"]["command_strength"] = 29
        cases.append(wrong_strength)

        missing_parameter = self.model.to_dict()
        del missing_parameter["parameters"]["dead_time_sec"]
        cases.append(missing_parameter)

        extra_parameter = self.model.to_dict()
        extra_parameter["parameters"]["unused"] = 1.0
        cases.append(extra_parameter)

        boolean_parameter = self.model.to_dict()
        boolean_parameter["parameters"]["dead_time_sec"] = False
        cases.append(boolean_parameter)

        for artifact in cases:
            with self.subTest(artifact=artifact):
                with self.assertRaises(ValueError):
                    LogRotationModel.from_dict(artifact)

    def test_non_finite_and_duplicate_json_values_are_rejected(self):
        artifact = self.model.to_dict()
        artifact["parameters"]["dead_time_sec"] = math.nan
        with self.assertRaises(ValueError):
            LogRotationModel.from_dict(artifact)

        payload = self.model.to_json(indent=None)
        duplicate = payload.replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            LogRotationModel.from_json(duplicate)

    def test_constructor_rejects_discontinuous_or_wrong_strength_model(self):
        with self.assertRaisesRegex(ValueError, "continuous piecewise"):
            replace(self.model, max_rate_deg_s=9.0)
        with self.assertRaisesRegex(ValueError, "command_strength=30"):
            replace(self.model, command_strength=31)


if __name__ == "__main__":
    unittest.main()

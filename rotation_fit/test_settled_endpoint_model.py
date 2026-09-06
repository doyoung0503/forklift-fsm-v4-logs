import unittest
import numpy as np
from rotation_fit.fit_settled_endpoint_model import fit_knots, predict, earliest_duration


class EndpointFunctionTests(unittest.TestCase):
    def test_monotone_zero_anchor_with_nonmonotone_observations(self):
        knots=fit_knots([1,2,3,4],[2,7,5,10],['a','b','c','d'])
        self.assertEqual(predict(knots,0),0)
        self.assertTrue(np.all(np.diff([predict(knots,t) for t in np.linspace(0,4,101)])>=0))

    def test_recording_weight_unchanged_when_all_its_rows_are_duplicated(self):
        first=fit_knots([1,1],[1,9],['a','b'])
        duplicated=fit_knots([1,1,1],[1,1,9],['a','a','b'])
        self.assertAlmostEqual(predict(first,1),predict(duplicated,1))

    def test_inverse_returns_first_point_on_plateau_and_round_trips(self):
        knots=([0,1,2,3,4],[0,4,4,8,10])
        self.assertEqual(earliest_duration(knots,4),1)
        for angle in (0,1,3,4,6,9,10):
            self.assertAlmostEqual(predict(knots,earliest_duration(knots,angle)),angle)

    def test_out_of_range_is_rejected_in_public_function(self):
        with self.assertRaises(ValueError): predict(([0,1],[0,2]),2)
        with self.assertRaises(ValueError): earliest_duration(([0,1],[0,2]),3)


if __name__=='__main__': unittest.main()

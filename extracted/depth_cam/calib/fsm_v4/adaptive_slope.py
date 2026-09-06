"""In-memory endpoint-slope learning; no file/log access or hardware calls."""
import math


class SessionSlope:
    def __init__(self, alpha=.25, minimum=.5, maximum=2., min_active_sec=.1):
        if not (all(math.isfinite(v) for v in (alpha, minimum, maximum, min_active_sec))
                and 0 < alpha <= 1 and 0 < minimum <= 1 <= maximum and min_active_sec > 0):
            raise ValueError('Invalid session slope learning limits')
        self.alpha = alpha
        self.minimum = minimum
        self.maximum = maximum
        self.min_active_sec = min_active_sec
        self.multipliers = {}

    def multiplier(self, command):
        return self.multipliers.get(command, 1.)

    def observe(self, command, base_slope, delay, actual_hold, settled_angle, min_active_sec=None):
        values = (base_slope, delay, actual_hold, settled_angle)
        if not all(math.isfinite(v) for v in values) or base_slope <= 0 or delay < 0:
            return {'slope_update_accepted': False, 'slope_skip_reason': 'invalid sample'}
        active = actual_hold - delay
        threshold = self.min_active_sec if min_active_sec is None else min_active_sec
        if active < threshold or settled_angle <= 0:
            return {'slope_update_accepted': False, 'slope_skip_reason': 'insufficient directed motion/time'}
        observed = settled_angle / active
        sample = min(self.maximum, max(self.minimum, observed / base_slope))
        previous = self.multiplier(command)
        updated = min(self.maximum, max(self.minimum,
                      previous + self.alpha * (sample - previous)))
        self.multipliers[command] = updated
        return dict(slope_update_accepted=True, slope_command=command,
                    slope_actual_hold_sec=actual_hold, slope_active_sec=active,
                    slope_settled_angle_deg=settled_angle,
                    slope_observed_deg_s=observed, slope_previous_deg_s=base_slope*previous,
                    slope_next_deg_s=base_slope*updated, slope_next_multiplier=updated)

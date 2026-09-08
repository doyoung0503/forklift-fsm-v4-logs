"""Synthetic model results, without rendering or inference.

Replace this sampler with an empirical log distribution later. The world and
CAN protocol do not depend on the distribution family.
"""
import math
import random


class GaussianResultModel:
    def __init__(self,options):
        self.options=options
        self.random=random.Random(options["seed"]^0xBAD5EED)

    def sample(self):
        o=self.options
        # Means are absolute errors (estimate minus ground truth), not jitter
        # about a running estimate. X/Y/Z use metres, yaw uses degrees.
        error={axis:self.random.gauss(o[f"{axis}_bias"],math.hypot(o[f"{axis}_noise"],
                    o["position_noise"] if axis in ("x","z") else 0.))
               for axis in ("x","y","z","yaw")}
        interval=max(.001,1/o["model_hz"]+self.random.gauss(0,o["model_interval_sd"]))
        latency=max(0.,self.random.gauss(o["latency"],o["latency_sd"]))
        return dict(error=error,interval=interval,latency=latency,
                    missed=self.random.random()<o["dropout"])

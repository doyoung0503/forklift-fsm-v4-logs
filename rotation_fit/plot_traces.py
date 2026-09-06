import io, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rot_common.py"), encoding="utf-8").read())

segs = load_segments()
n = len(segs)
cols, rows = 4, int(np.ceil(n / 4))
fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), sharex=False)
axes = np.atleast_1d(axes).ravel()
for ax, s in zip(axes, segs):
    tt = s.t - s.t_cmd
    ax.plot(tt, s.angle, ".-", ms=3, lw=1, color="#1f77b4", label="angle(bearing)")
    ax.axvline(0, color="g", lw=1)
    ax.axvline(s.t_stop - s.t_cmd, color="r", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax2 = ax.twinx()
    ax2.plot(tt, s.rng, lw=0.8, color="#bbbbbb")
    ax2.set_ylim(s.rng.mean() - 0.6, s.rng.mean() + 0.6)
    ax.set_title(f"{s.tag[-6:]} s{s.step} {s.cmd} dur={s.cmd_duration:.2f}s r={s.rng.mean():.1f}m", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
for ax in axes[n:]:
    ax.axis("off")
fig.suptitle("rotation segments: directed angle from face-centre bearing (green=ROT cmd, red=STOP cmd)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "01_segment_traces.png")
fig.savefig(out, dpi=110)
print("saved", out)

"""Reproduce line-level diagnostics for the bundled SHC 13 ring-4 data.

Run from the repository root after `pip install -e '.[examples]'`:
    python scripts/analyze_SHC13_rotational.py

The alternative models are diagnostic checks, not replacements for radiative
transfer or a vetted rotational/vibrational excitation analysis.
"""
from pathlib import Path
import contextlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2
from astropy import constants as const, units as u

from astrothesispy.utiles import utiles_molecules as molecules

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/NGC253_HR/Results"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

lines = molecules.HC3N_line_dict_builder(
    "SHC_13", str(RESULTS) + "/", str(RESULTS / "Tables/SLIM_HC3N_info.xlsx"),
    .020, .022,
)
with contextlib.redirect_stdout(io.StringIO()):
    df, result = molecules.rotational_diagram(lines, Jselect=24, bootstrap=False)

x = df["EUP_K"].to_numpy()
y = df["logNu/gu"].to_numpy()
sigma = df["logNu/gu_relerr"].to_numpy()
old_sigma = np.log10(df["Nu/gu_err"].to_numpy()) / (np.log(10)*y)
old_fit, _ = curve_fit(molecules.linfit, x, y, sigma=old_sigma)
new_fit = result["NoBootstrap"]["popt"]

# Each model is nested only as an algebraic diagnostic. J and observing band
# coincide in this dataset, so the extra J term cannot identify a unique cause.
J = df["qn1"].to_numpy()
E_rot = (const.h * 4.55*u.GHz / const.k_B).to_value(u.K)*J*(J+1)
E_vib = x-E_rot
ground = (df["vib_state"] == "v=0").to_numpy().astype(float)
models = {
    "one_slope": np.column_stack([np.ones(len(df)), x]),
    "two_slopes": np.column_stack([np.ones(len(df)), E_vib, E_rot]),
    "two_slopes_ground_offset": np.column_stack([np.ones(len(df)), E_vib, E_rot, ground]),
}
fits = {}
for name, matrix in models.items():
    weighted_matrix = matrix/sigma[:, None]
    parameters = np.linalg.lstsq(weighted_matrix, y/sigma, rcond=None)[0]
    residual = y-matrix@parameters
    chisq = np.sum((residual/sigma)**2)
    fits[name] = (parameters, chisq, len(df)-len(parameters))

out = df[["vib_state", "qn1", "Frequency_GHz", "EUP_K", "Nu/gu", "Nu/gu_err", "logNu/gu"]].copy()
out["fractional_error"] = df["Nu/gu_err"] / df["Nu/gu"]
out["old_sigma_dex"] = old_sigma
out["correct_sigma_dex"] = sigma
out["old_weight_percent"] = (old_sigma**-2)/sum(old_sigma**-2)*100
out["correct_weight_percent"] = (sigma**-2)/sum(sigma**-2)*100
out["old_fit_residual_dex"] = y-molecules.linfit(x,*old_fit)
out["correct_fit_residual_dex"] = y-molecules.linfit(x,*new_fit)
out["standardized_residual"] = out["correct_fit_residual_dex"] / sigma
out["chi2_contribution"] = out["standardized_residual"]**2
out["chi2_percent"] = out["chi2_contribution"]/out["chi2_contribution"].sum()*100
leave_out = []
for i in range(len(df)):
    keep = np.arange(len(df)) != i
    p, _ = curve_fit(molecules.linfit, x[keep], y[keep], sigma=sigma[keep], absolute_sigma=True)
    leave_out.append(-1/(np.log(10)*p[0]))
out["temperature_without_line_K"] = leave_out
out["delta_temperature_without_line_K"] = np.asarray(leave_out)-result["NoBootstrap"]["Tex"]
csv_path = OUT/"SHC13_line_diagnostics.csv"
out.to_csv(csv_path, index=False, float_format="%.10g")

plt.rcParams["text.usetex"] = False  # Legacy plotting module enables TeX globally.
fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
colors = {24: "#227f95", 26: "#b24d32"}
for j in [24,26]:
    selected = J == j
    ax.errorbar(x[selected],y[selected],yerr=sigma[selected], fmt="o",ms=6,
                color=colors[j], capsize=2, label=f"J={j} measured lines")
    residual_ax.scatter(x[selected],out.loc[selected,"standardized_residual"],
                        color=colors[j],s=45,label=f"J={j}")
xx = np.linspace(min(x)-30,max(x)+30,200)
ax.plot(xx,molecules.linfit(xx,*old_fit),"--",color="#777777",label="Original weighting: 462 K")
ax.plot(xx,molecules.linfit(xx,*new_fit),"-",color="#252525",label="Correct weighting: 608 K")
ax.set_ylabel(r"$\log_{10}(N_u/g_u)$ [cm$^{-2}$]")
ax.legend(loc="upper right",frameon=False)
ax.grid(alpha=.2)
for val in [-3,0,3]:
    residual_ax.axhline(val,color="#666666",linewidth=.8,linestyle="-" if val==0 else ":")
for _,row in out.iterrows():
    if abs(row["standardized_residual"]) >= 3:
        residual_ax.annotate(f"{row.vib_state}, J={row.qn1}",
                             (row.EUP_K,row.standardized_residual),
                             xytext=(4,5),textcoords="offset points",fontsize=7)
residual_ax.set_xlabel(r"$E_u/k$ [K]")
residual_ax.set_ylabel("Residual / stated error")
residual_ax.set_ylim(-10,11)
residual_ax.grid(alpha=.2)
fig.suptitle("SHC 13, ring 4: single-slope fit and line residuals")
fig.tight_layout()
fig.savefig(OUT/"SHC13_fit_diagnostics.png", dpi=180)
plt.close(fig)

print("Old single-slope T:", -1/(np.log(10)*old_fit[0]))
for name, (p,chisq,dof) in fits.items():
    print(name,"T_vib_or_single_K",-1/(np.log(10)*p[1]),
          "chi2",chisq,"dof",dof,"reduced_chi2",chisq/dof,"p",chi2.sf(chisq,dof))
    if name == "two_slopes_ground_offset":
        print("diagnostic T_rot_K",-1/(np.log(10)*p[2]),
              "ground_to_excited_offset_dex",p[3])
print("Line table:", csv_path)

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import colorsys
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import normal_ad, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy.builtins import Q


#---------------------
data = pd.read_csv("CO2 Emissions_Canada.csv", header=0, encoding="utf-8")

if data.shape[1] >= 2:
    data.drop(columns=data.columns[-2], inplace=True)

data.columns = [
    "Markė",
    "Modelis",
    "Kėbulo tipas",
    "Variklio tūris (l)",
    "Cilindrų skaičius",
    "Pavarų dėžės tipas",
    "Kuro tipas",
    "Kuro sąnaudos mieste (l/100km)",
    "Kuro sąnaudos greitkelyje (l/100km)",
    "Vidutinės kuro sąnaudos (l/100km)",
    "CO2 emisijos (g/km)"
]
body_map = {
    "SUV - SMALL": "SUV", "SUV - STANDARD": "SUV",
    "MID-SIZE": "Lengvieji automobiliai", "COMPACT": "Lengvieji automobiliai",
    "SUBCOMPACT": "Lengvieji automobiliai", "MINICOMPACT": "Lengvieji automobiliai",
    "FULL-SIZE": "Lengvieji automobiliai", "TWO-SEATER": "Lengvieji automobiliai",
    "STATION WAGON - SMALL": "Universalai", "STATION WAGON - MID-SIZE": "Universalai",
    "PICKUP TRUCK - SMALL": "Pikapai", "PICKUP TRUCK - STANDARD": "Pikapai",
    "MINIVAN": "Vienatūriai", "VAN - PASSENGER": "Vienatūriai", "VAN - CARGO": "Vienatūriai",
    "SPECIAL PURPOSE VEHICLE": "Specialieji"
}
data["Kėbulo tipas"] = data["Kėbulo tipas"].map(body_map).astype("category")

trans = data["Pavarų dėžės tipas"].astype(str)

conds = [
    trans.str.match(r"^AM", na=False),
    trans.str.match(r"^AS", na=False),
    trans.str.match(r"^AV", na=False),
    trans.str.match(r"^A",  na=False),
    trans.str.match(r"^M",  na=False),
]
choices = [
    "Automatizuota mechaninė",
    "Automatinė su rankiniu perjungimu",
    "Bepakopė (CVT)",
    "Automatinė",
    "Mechaninė",
]

mapped = np.select(conds, choices, default=None)   
s = pd.Series(mapped, index=data.index, dtype="object").replace({None: pd.NA})
data.loc[:, "Pavarų dėžės tipas"] = s.astype("category")

fuel_map = {
    "X": "Įprastas benzinas",
    "Z": "Aukštos kokybės benzinas",
    "D": "Dyzelinas",
    "E": "Etanolis (E85)",
    "N": "Gamtinės dujos",
}
data["Kuro tipas"] = data["Kuro tipas"].map(fuel_map).astype("category")


data = data.loc[data["Variklio tūris (l)"] == 2.0].copy()

for col in ["Kėbulo tipas", "Pavarų dėžės tipas", "Kuro tipas"]:
    if col in data.columns and isinstance(data[col].dtype, CategoricalDtype):
        data[col] = data[col].cat.remove_unused_categories()

#---------------
numeric_vars = [
    "Vidutinės kuro sąnaudos (l/100km)",
    "Kuro sąnaudos greitkelyje (l/100km)",
    "Kuro sąnaudos mieste (l/100km)",
    "Cilindrų skaičius",
    "CO2 emisijos (g/km)"
]

for col in numeric_vars:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")


na_counts = data.isna().sum()
print(na_counts)

def check_outliers(x: pd.Series, coef: float = 1.5) -> int:
    s = x.dropna()
    if s.empty:
        return 0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - coef * iqr
    upper = q3 + coef * iqr
    mask = (x < lower) | (x > upper)
    return int(mask.fillna(False).sum())

def is_outlier(x: pd.Series, coef: float = 1.5) -> pd.Series:
    s = x.dropna()
    if s.empty:
        return pd.Series(False, index=x.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - coef * iqr
    upper = q3 + coef * iqr
    mask = (x < lower) | (x > upper)
    return mask.fillna(False)

out_15 = {col: check_outliers(data[col], coef=1.5) for col in numeric_vars if col in data.columns}
out_3  = {col: check_outliers(data[col], coef=3.0) for col in numeric_vars if col in data.columns}
outlier_summary = pd.DataFrame({
    "Outliers_1.5IQR": pd.Series(out_15),
    "Outliers_3IQR": pd.Series(out_3)
})
print(outlier_summary)

print(f"Pradinė imtis: {len(data)} eilučių")

outlier_list = [is_outlier(data[col], coef=1.5) for col in numeric_vars if col in data.columns]
if outlier_list:
    rows_to_remove = pd.concat(outlier_list, axis=1).any(axis=1)
else:
    rows_to_remove = pd.Series(False, index=data.index)

removed = int(rows_to_remove.sum())
data = data.loc[~rows_to_remove].copy()

print(f"Pašalinta išskirčių (1.5 IQR): {removed} eilučių")
print(f"Likusi imtis: {len(data)} eilučių")


#-----------------------------------
s = pd.to_numeric(data["CO2 emisijos (g/km)"], errors="coerce").dropna()

plt.figure()
plt.hist(s, bins=40, density=True, edgecolor="black", color="lightblue", label="CO2 emisijų pasiskirstymas")

x = np.linspace(s.min(), s.max(), 200)
m = s.mean()
sd = s.std(ddof=1)
y = stats.norm.pdf(x, loc=m, scale=sd)
plt.plot(x, y, linewidth=2, color="red", label="Normalumo kreivė")

plt.title("Automobilių CO2 emisijų pasiskirstymo histograma")
plt.xlabel("CO2 emisijos (g/km)")
plt.ylabel("Tankis")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()

plt.figure()
stats.probplot(s, dist="norm", plot=plt)

plt.title("QQ CO2 emisijų grafikas")
plt.xlabel("Teoriniai kvantiliai")
plt.ylabel("Stebėtos CO2 emisijos (g/km)")

plt.tight_layout()
plt.show()

#-------------------

s = pd.to_numeric(data["CO2 emisijos (g/km)"], errors="coerce").dropna()

A, p = normal_ad(s)

print("Anderson-Darling normality test")
print(f"A = {A:.6f}, p-value = {p:.6g}")


#----------------

num = data[[
    "Kuro sąnaudos mieste (l/100km)",
    "Kuro sąnaudos greitkelyje (l/100km)",
    "Vidutinės kuro sąnaudos (l/100km)",
    "CO2 emisijos (g/km)"
]].copy()

num.columns = [
    "Kuro sąnaudos\nmieste (l/100km)",
    "Kuro sąnaudos\ngreitkelyje (l/100km)",
    "Vidutinės kuro\nsąnaudos (l/100km)",
    "CO2 emisijos (g/km)"
]
for c in num.columns:
    num[c] = pd.to_numeric(num[c], errors="coerce")

labels = list(num.columns)
cor = num.corr(numeric_only=True).values
n = len(labels)

plot_vals = cor.copy()
plot_vals[np.tril_indices(n, -1)] = np.nan

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(plot_vals, vmin=-1, vmax=1, cmap="bwr_r")

for i in range(n):
    for j in range(n):
        if i <= j:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                       edgecolor="lightgray", linewidth=1))
            val = plot_vals[i, j]
            if not np.isnan(val):
                txt_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color=txt_color)

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=90, color="black")
ax.set_yticklabels(labels, color="black")
ax.set_title("Koreliacijų matrica (−1 raudona, 1 mėlyna)", pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

cols = labels
n = len(cols)
fig, axes = plt.subplots(n, n, figsize=(12, 12))
point_color = "#009ACD"
line_color = "red"

for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            ax.axis("off")
            ax.text(0.5, 0.5, cols[i], ha="center", va="center",
                    fontweight="bold", transform=ax.transAxes)
        elif i < j:
            x = num[cols[j]]
            y = num[cols[i]]
            m = x.notna() & y.notna()
            xv = x[m].values
            yv = y[m].values

            ax.scatter(xv, yv, alpha=0.5, s=15, edgecolors="none", color=point_color)

            if xv.size >= 2:
                b1, b0 = np.polyfit(xv, yv, 1)
                xline = np.linspace(xv.min(), xv.max(), 100)
                yline = b1 * xline + b0
                ax.plot(xline, yline, linewidth=1.0, color=line_color)

            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(cols[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(cols[i])
        else:
            ax.axis("off")

fig.suptitle("Sklaidos diagramų matrica", fontweight="bold", y=0.92)
plt.tight_layout()
plt.show()



#-----------------

dep_var = "CO2 emisijos (g/km)"
data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")

def rainbow_hcl_py(n, c=60, l=80):
    if n <= 0:
        return []
    s = float(np.clip(c/100.0, 0, 1))
    L = float(np.clip(l/100.0, 0, 1))
    cols = []
    for k in range(n):
        h = (k / n) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, L, s)
        cols.append(f'#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}')
    return cols

def plot_box_rainbow(df, var_name, title, x_angle=0):
    plt.close('all')

    cats = pd.Index(df[var_name].dropna().astype(str).unique())
    n = len(cats)
    if n == 0:
        print(f"[{var_name}] nėra kategorijų po valymo. Nieko nepaišau.")
        return

    palette = rainbow_hcl_py(n, c=60, l=80)
    groups = [df.loc[df[var_name].astype(str) == cat, dep_var].dropna().values for cat in cats]

    fig, ax = plt.subplots(figsize=(max(8, n*0.5), 6))

    bp = ax.boxplot(groups, patch_artist=True, widths=0.6)

    for patch, color in zip(bp['boxes'], palette):
        patch.set(facecolor=color, edgecolor="black", linewidth=1.0)
    for comp in ('whiskers', 'caps', 'medians'):
        for el in bp[comp]:
            el.set(color="black", linewidth=1.0)
    for el in bp['medians']:
        el.set(linewidth=1.5)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(cats.tolist(), rotation=x_angle)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(var_name)
    ax.set_ylabel(dep_var)

    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    plt.close(fig)

plot_box_rainbow(data, "Markė", "CO2 emisijos pagal markes", x_angle=90)
plot_box_rainbow(data, "Kėbulo tipas", "CO2 emisijos pagal kėbulo tipą")
plot_box_rainbow(data, "Pavarų dėžės tipas", "CO2 emisijos pagal pavarų dėžę")
plot_box_rainbow(
    data.loc[data["Kuro tipas"] != "Gamtinės dujos"],
    "Kuro tipas", "CO2 emisijos pagal kuro tipą", x_angle=0
)

#----------------
def freq_table_df(df: pd.DataFrame, var_name: str) -> pd.DataFrame:
    """
    Dažnių lentelė kaip dplyr::count + arrange(desc(n)).
    NaN/NA NEIŠMETAMI (dropna=False).
    """
    out = (
        df.groupby(var_name, dropna=False, observed=True)
          .size()
          .reset_index(name="Dažnis")
          .sort_values("Dažnis", ascending=False, kind="mergesort")
          .reset_index(drop=True)
    )
    out.columns = [var_name, "Dažnis"]
    return out

freq_marke = freq_table_df(data, "Markė")

freq_kebulas = freq_table_df(data, "Kėbulo tipas")

freq_pavaros = freq_table_df(data, "Pavarų dėžės tipas")

freq_kuras = freq_table_df(data, "Kuro tipas")

print(freq_marke)
print(freq_kebulas)
print(freq_pavaros)
print(freq_kuras)

#------------------
dep_var = "CO2 emisijos (g/km)"
data[dep_var] = pd.to_numeric(data[dep_var], errors="coerce")

def mean_point_plot(df, var_name, filter_values=None, color="black", title=None, x_angle=0):
    d = df
    if filter_values is not None:
        d = d[d[var_name].isin(filter_values)]

    summ = (
        d.groupby(var_name, dropna=False, observed=True)[dep_var]
         .agg(mean_CO2=lambda s: s.mean(skipna=True),
              sd_CO2=lambda s: s.std(ddof=1, skipna=True),
              n="size")
         .reset_index()
    )
    summ = summ.dropna(subset=["mean_CO2", "sd_CO2"])

    cats = summ[var_name].astype(str).tolist()
    x = np.arange(len(cats)) + 1 

    y = summ["mean_CO2"].to_numpy()
    sd = summ["sd_CO2"].to_numpy()
    y_min = y - sd
    y_max = y + sd

    fig_w = max(8, 0.6 * len(cats))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    ax.scatter(x, y, s=30, color=color, zorder=3)

    ax.vlines(x, y_min, y_max, colors=color, linewidth=1.5, zorder=2)

    cap_w = 0.2
    for xi, ymin_i, ymax_i in zip(x, y_min, y_max):
        ax.hlines([ymin_i, ymax_i], xi - cap_w, xi + cap_w, colors=color, linewidth=1.5, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=x_angle, ha="center")
    ax.set_xlabel(var_name)
    ax.set_ylabel(dep_var)
    if title:
        ax.set_title(title, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="both", width=0.5)

    ax.grid(False)
    plt.tight_layout()
    plt.show()
    plt.close(fig)

mean_point_plot(
    data, "Markė",
    color="navy",
    title="Vidutinės CO2 emisijos pagal markes",
    x_angle=90
)

mean_point_plot(
    data, "Kėbulo tipas",
    color="black",
    title="Vidutinės CO2 emisijos pagal kėbulo tipą"
)

mean_point_plot(
    data, "Pavarų dėžės tipas",
    color="darkred",
    title="Vidutinės CO2 emisijos pagal pavarų dėžės tipą"
)

mean_point_plot(
    data, "Kuro tipas",
    color="darkgreen",
    title="Vidutinės CO2 emisijos pagal kuro tipą"
)



#----------------

rng = np.random.default_rng(123)

n = len(data)
perm = rng.permutation(n)   
split = int(0.8 * n)             

train_idx = perm[:split]
test_idx  = perm[split:]

train_data = data.iloc[train_idx].copy()
test_data  = data.iloc[test_idx].copy()

print(f"Eilučių skaičius pradiniuose duomenyse: {n}")
print(f"Eilučių skaičius treniravimo aibėje: {len(train_data)}")
print(f"Eilučių skaičius testavimo aibėje: {len(test_data)}")


#-------------------------------------
for col in ["Pavarų dėžės tipas", "Kuro tipas", "Markė", "Kėbulo tipas"]:
    if col in train_data.columns:
        train_data[col] = train_data[col].astype("category")

formula = (
    "Q('CO2 emisijos (g/km)') ~ "
    "Q('Vidutinės kuro sąnaudos (l/100km)') + "
    "Q('Kuro sąnaudos mieste (l/100km)') + "
    "Q('Kuro sąnaudos greitkelyje (l/100km)') + "
    "C(Q('Pavarų dėžės tipas')) + "
    "C(Q('Kuro tipas')) + "
    "C(Q('Markė')) + "
    "C(Q('Kėbulo tipas'))"
)

model1 = ols(formula=formula, data=train_data).fit()

influence = model1.get_influence()

fitted = model1.fittedvalues
resid = model1.resid

rstd_internal = influence.resid_studentized_internal    
rstd_external = influence.resid_studentized_external   

hat = influence.hat_matrix_diag

cooks = influence.cooks_distance[0]

train_data = train_data.copy()
train_data["fitted"] = fitted.values
train_data["resid"] = resid.values
train_data["rstd"] = rstd_internal
train_data["rstud"] = rstd_external
train_data["hat"] = hat
train_data["cooks"] = cooks
train_data["obs"] = np.arange(1, len(train_data) + 1)

print(model1.summary())


#----------------------------
fitted = model1.fittedvalues
resid = model1.resid
rstd  = train_data["rstd"].to_numpy()  
cooks = train_data["cooks"].to_numpy()
n = len(resid)

color_pts = "#009ACD" 
color_line = "red"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axes

ax1.scatter(fitted, resid, s=30, color=color_pts)
ax1.axhline(0, linestyle="--", color=color_line)
ax1.set_xlabel("Prognozuotos reikšmės")
ax1.set_ylabel("Liekanos")
ax1.set_title("Liekanos ir prognozuotos reikšmės")

stats.probplot(resid, dist="norm", plot=ax2)
ax2.lines[0].set_color(color_pts)  
ax2.lines[1].set_color(color_line) 
ax2.lines[1].set_linewidth(2)
ax2.set_title("Liekanų QQ grafikas")
ax2.set_xlabel("Teoriniai kvantiliai")
ax2.set_ylabel("Stebėtos liekanos")

y_sl = np.sqrt(np.abs(rstd))
ax3.scatter(fitted, y_sl, s=30, color=color_pts)
mask = np.isfinite(fitted) & np.isfinite(y_sl)
if mask.sum() > 2:
    smth = lowess(y_sl[mask], fitted[mask], frac=0.6, return_sorted=True)
    ax3.plot(smth[:, 0], smth[:, 1], color=color_line, linewidth=2)
ax3.set_xlabel("Prognozuotos reikšmės")
ax3.set_ylabel("Standartizuotos liekanos")
ax3.set_title("Scale-Location grafikas")

x_idx = np.arange(1, n + 1)
ax4.vlines(x_idx, 0, cooks, color=color_pts, linewidth=2)
ax4.set_xlim(0, n + 1)
ax4.set_xlabel("Stebėjimo indeksas")
ax4.set_ylabel("Cook's atstumas")
ax4.set_title("Įtakingi taškai (Cook's distance)")
ax4.axhline(4 / n, color=color_line, linestyle="--")

plt.tight_layout()
plt.show()


#----------------------
resid = model1.resid
cooks = train_data["cooks"].to_numpy()
rstd = train_data["rstd"].to_numpy()    
rstud = train_data["rstud"].to_numpy()  
n = len(resid)

color_pts = "#009ACD" 

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axes
x = np.arange(1, n + 1)

ax1.scatter(x, resid, s=30, color="red")
ax1.axhline(0, color="black", linestyle="--", linewidth=2)
ax1.set_xlabel("Stebėjimo indeksas")
ax1.set_ylabel("Liekanos")
ax1.set_title("Liekanos pagal stebėjimus")

ax2.scatter(x, cooks, s=30, color=color_pts)
ax2.axhline(4 / n, color="black", linestyle="--", linewidth=2)
ax2.set_xlabel("Stebėjimo indeksas")
ax2.set_ylabel("Cook's atstumas")
ax2.set_title("Įtakingi taškai (Cook's distance)")

ax3.scatter(x, rstd, s=30, color=color_pts)
for h in [-2, 0, 2]:
    ax3.axhline(h, color="black", linestyle="--")
ax3.set_xlabel("Stebėjimo indeksas")
ax3.set_ylabel("Standartizuotos liekanos")
ax3.set_title("Standartizuotos liekanos")

ax4.scatter(x, rstud, s=30, color=color_pts)
for h in [-3, 0, 3]:
    ax4.axhline(h, color="black", linestyle="--")
ax4.set_xlabel("Stebėjimo indeksas")
ax4.set_ylabel("Studentizuotos liekanos")
ax4.set_title("Studentizuotos liekanos")

plt.tight_layout()
plt.show()


#----------------------------
res = model1.resid.dropna().to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(res, dist="norm", plot=axes[0])
axes[0].lines[0].set_color("#009ACD")
axes[0].lines[1].set_color("red")    
axes[0].lines[1].set_linewidth(2)
axes[0].set_title("Liekanų QQ grafikas")
axes[0].set_xlabel("Teoriniai kvantiliai")
axes[0].set_ylabel("Stebėtos liekanos")

axes[1].hist(res, bins=30, density=True, color="#009ACD", edgecolor="black")
axes[1].set_title("Liekanų pasiskirstymas")
axes[1].set_xlabel("Liekanos")
axes[1].set_ylabel("Tankis")

mu = np.mean(res)
sd = np.std(res, ddof=1)
x = np.linspace(res.min(), res.max(), 200)
axes[1].plot(x, stats.norm.pdf(x, loc=mu, scale=sd), color="red", linewidth=2)

plt.tight_layout()
plt.show()

A, p = normal_ad(res)
print("Anderson-Darling normality test")
print(f"A = {A:.6f}, p-value = {p:.6g}")


#-----------------------
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model1.resid, model1.model.exog)
print("Breusch–Pagan testas (H0: homoskedastiškos liekanos)")
print(f"LM statistika = {bp_lm:.4f}, p-reikšmė = {bp_lm_p:.6g}")
print(f"F statistika  = {bp_f:.4f}, p-reikšmė = {bp_f_p:.6g}")

fitted = model1.fittedvalues.to_numpy()
resid = model1.resid.to_numpy()

plt.figure(figsize=(7, 5))
plt.scatter(fitted, resid, s=30, color="#009ACD") 
plt.axhline(0, color="black", linestyle="--", linewidth=2)

mask = np.isfinite(fitted) & np.isfinite(resid)
if mask.sum() > 2:
    smth = lowess(resid[mask], fitted[mask], frac=0.6, return_sorted=True)
    plt.plot(smth[:, 0], smth[:, 1], color="red", linewidth=2)

plt.xlabel("Prognozuotos reikšmės")
plt.ylabel("Liekanos")
plt.title("Liekanos ir prognozuotos reikšmės")
plt.tight_layout()
plt.show()

rob = model1.get_robustcov_results(cov_type="HC3")

ci = rob.conf_int()
robust_table = pd.DataFrame({
    "coef": rob.params,
    "std err (HC3)": rob.bse,
    "t": rob.tvalues,
    "P>|t|": rob.pvalues,
    "CI low": ci[:, 0],
    "CI high": ci[:, 1],
})
print("\nKoeficientai su HC3 robust SE:")
print(robust_table)


#--------------------------
def bonferroni_outlier_test(model, rstud_series):
    """
    Analogiška car::outlierTest:
    - Skaičiuoja dvišales p reikšmes iš rstudent
    - Bonferroni korekcija
    - Išveda mažiausią p ir lentelę surikiuotą pagal |rstud|
    """
    rstud = np.asarray(rstud_series)
    n = model.nobs
    df = int(model.df_resid) 
    p_two = 2 * stats.t.sf(np.abs(rstud), df=df)
    p_bonf = np.minimum(p_two * n, 1.0)

    out = pd.DataFrame({
        "index": np.arange(1, n + 1),
        "rstudent": rstud,
        "p_two_sided": p_two,
        "p_bonferroni": p_bonf
    }).sort_values("p_two_sided", ascending=True, ignore_index=True)

    best = out.iloc[0]
    print("Bonferroni outlier test (pagal rstudent)")
    print(f"Mažiausia p: indeksas {int(best['index'])}, rstudent = {best['rstudent']:.4f}, "
          f"p = {best['p_two_sided']:.6g}, p_bonf = {best['p_bonferroni']:.6g}")
    return out

out_tbl = bonferroni_outlier_test(model1, train_data["rstud"])

print("\nTOP 10 mažiausių p reikšmių:")
print(out_tbl.head(10))

idx = np.arange(1, len(train_data) + 1)
cook = train_data["cooks"].to_numpy()
rstud = train_data["rstud"].to_numpy()
hat   = train_data["hat"].to_numpy()

def annotate_top3(ax, x, y, title, horiz=None):
    ax.scatter(x, y, s=18, color="#009ACD")
    if horiz is not None:
        ax.axhline(horiz, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Stebėjimo indeksas")
    ax.set_title(title)

    top3_idx = np.argsort(np.abs(y))[-3:][::-1]
    for i in top3_idx:
        ax.scatter(x[i], y[i], s=35, color="red", zorder=3)
        ax.annotate(str(x[i]), (x[i], y[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color="red")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

annotate_top3(
    axes[0], idx, cook,
    "Įtakingi taškai (Cook's distance)",
    horiz=4/len(train_data)
)
axes[0].set_ylabel("Cook's atstumas")

annotate_top3(
    axes[1], idx, rstud,
    "Studentizuotos liekanos"
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
axes[1].axhline(3, color="black", linestyle="--", linewidth=1.0)
axes[1].axhline(-3, color="black", linestyle="--", linewidth=1.0)
axes[1].set_ylabel("rstudent")

annotate_top3(
    axes[2], idx, hat,
    "Leverage (hat reikšmės)"
)
axes[2].set_ylabel("hat")
axes[2].set_xlabel("Stebėjimo indeksas")

plt.tight_layout()
plt.show()


#----------------
infl  = model1.get_influence()
rstud = infl.resid_studentized_external      
hatv  = infl.hat_matrix_diag               
cooks = infl.cooks_distance[0]              

n = int(model1.nobs)
suspect_mask = (np.abs(rstud) > 3) | (hatv > 0.5) | (cooks > (4 / n))
suspect_idx  = np.flatnonzero(suspect_mask)  

print("Įtartinų taškų skaičius:", suspect_idx.size)

data_clean = train_data.drop(train_data.index[suspect_idx]).copy()

model2 = ols(formula=formula, data=data_clean).fit()

#--------------
X = pd.DataFrame(model2.model.exog, columns=model2.model.exog_names)

if "Intercept" in X.columns:
    X = X.drop(columns=["Intercept"])

nzv_cols = []
for c in X.columns:
    s = X[c].to_numpy()
    if not np.isfinite(s).all():
        nzv_cols.append(c)
    else:
        if np.nanstd(s) == 0: 
            nzv_cols.append(c)
        elif (np.nanmax(s) - np.nanmin(s)) < 1e-12: 
            nzv_cols.append(c)
if nzv_cols:
    print("VIF: išmetu nevaryuojančius stulpelius:", nzv_cols)
    X = X.drop(columns=nzv_cols)

if X.shape[1] == 0:
    print("VIF: po valymo neliko stulpelių.")
else:
    vif_vals = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for i in range(X.shape[1]):
            try:
                vif_vals.append(variance_inflation_factor(X.values, i))
            except Exception:
                vif_vals.append(np.inf)

    vif_df = (pd.DataFrame({"Kintamasis": X.columns, "VIF": vif_vals})
                .sort_values("VIF", ascending=False, kind="mergesort")
                .reset_index(drop=True))
    print("\nVIF (po apsivalymo):")
    print(vif_df)



#---------------
formula2a = (
    "Q('CO2 emisijos (g/km)') ~ "
    "Q('Vidutinės kuro sąnaudos (l/100km)') + "
    "C(Q('Pavarų dėžės tipas')) + "
    "C(Q('Markė')) + "
    "C(Q('Kuro tipas')) + "
    "C(Q('Kėbulo tipas'))"
)

need2a = [
    "CO2 emisijos (g/km)",
    "Vidutinės kuro sąnaudos (l/100km)",
    "Pavarų dėžės tipas", "Markė", "Kuro tipas", "Kėbulo tipas"
]
df2a = data_clean.dropna(subset=need2a).copy()

for c in ["Pavarų dėžės tipas", "Markė", "Kuro tipas", "Kėbulo tipas"]:
    df2a[c] = df2a[c].astype("category")

model2a = ols(formula=formula2a, data=df2a).fit()
print(model2a.summary().tables[1]) 

X = pd.DataFrame(model2a.model.exog, columns=model2a.model.exog_names)

if "Intercept" in X.columns:
    X = X.drop(columns=["Intercept"])

drop_cols = [c for c in X.columns if not np.isfinite(X[c]).all() or np.nanstd(X[c]) == 0 or (X[c].max() - X[c].min()) < 1e-12]
if drop_cols:
    X = X.drop(columns=drop_cols)

if X.shape[1] == 0:
    print("VIF: po apsivalymo neliko stulpelių skaičiavimui.")
else:
    vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = pd.DataFrame({"Kintamasis": X.columns, "VIF": vif_vals}) \
             .sort_values("VIF", ascending=False, kind="mergesort") \
             .reset_index(drop=True)
    print("\nVIF (model2a):")
    print(vif_df)





#----------------
need2a = [
    "CO2 emisijos (g/km)",
    "Vidutinės kuro sąnaudos (l/100km)",
    "Pavarų dėžės tipas", "Markė", "Kuro tipas", "Kėbulo tipas"
]
df2a = data_clean.dropna(subset=need2a).copy()
for c in ["Pavarų dėžės tipas", "Markė", "Kuro tipas", "Kėbulo tipas"]:
    df2a[c] = df2a[c].astype("category")

y_term = "Q('CO2 emisijos (g/km)')"

terms_all = [
    "Q('Vidutinės kuro sąnaudos (l/100km)')",
    "C(Q('Pavarų dėžės tipas'))",
    "C(Q('Markė'))",
    "C(Q('Kuro tipas'))",
    "C(Q('Kėbulo tipas'))",
]

def fit_aic(terms):
    rhs = " + ".join(terms) if terms else "1"
    fml = f"{y_term} ~ {rhs}"
    m = ols(fml, data=df2a).fit()
    return m, m.aic, fml

current_terms = terms_all.copy()
model_current, aic_current, fml_current = fit_aic(current_terms)

improved = True
while improved and len(current_terms) > 1:
    improved = False
    best_drop = None
    best_model = None
    best_aic = aic_current

    for t in current_terms:
        cand_terms = [x for x in current_terms if x != t]
        try:
            m_cand, aic_cand, _ = fit_aic(cand_terms)
        except Exception:
            continue
        if aic_cand + 1e-9 < best_aic:
            best_aic = aic_cand
            best_drop = t
            best_model = m_cand

    if best_drop is not None:
        current_terms.remove(best_drop)
        model_current = best_model
        aic_current = best_aic
        improved = True
        print(f"StepAIC: šalinam {best_drop} → AIC {aic_current:.3f}")

step_model2 = model_current
print("\nGalutinė formulė po step-AIC:")
print(step_model2.model.formula)
print(f"AIC: {aic_current:.3f}, nobs: {int(step_model2.nobs)}")

X = pd.DataFrame(step_model2.model.exog, columns=step_model2.model.exog_names)
if "Intercept" in X.columns:
    X = X.drop(columns=["Intercept"])

drop_cols = [c for c in X.columns
             if not np.isfinite(X[c]).all() or np.nanstd(X[c]) == 0 or (np.nanmax(X[c]) - np.nanmin(X[c])) < 1e-12]
if drop_cols:
    X = X.drop(columns=drop_cols)

if X.shape[1] == 0:
    print("\nVIF: po apsivalymo neliko stulpelių skaičiavimui.")
else:
    vif_vals = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = (pd.DataFrame({"Kintamasis": X.columns, "VIF": vif_vals})
                .sort_values("VIF", ascending=False, kind="mergesort")
                .reset_index(drop=True))
    print("\nVIF (step_model2):")
    print(vif_df)





#----------------
fitted = model2a.fittedvalues.to_numpy()
resid  = model2a.resid.to_numpy()

rstd = model2a.get_influence().resid_studentized_internal
sl_y = np.sqrt(np.abs(rstd))

color_pts = "#7A378B"
color_line = "red"

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axes

ax1.scatter(fitted, resid, s=20, color=color_pts)
ax1.axhline(0, color=color_line, linestyle="--")
ax1.set_xlabel("Prognozuotos reikšmės")
ax1.set_ylabel("Liekanos")
ax1.set_title("Liekanos ir prognozuotos")

stats.probplot(resid, dist="norm", plot=ax2)
ax2.lines[0].set_color(color_pts)   
ax2.lines[1].set_color(color_line)  
ax2.lines[1].set_linewidth(2)
ax2.set_title("Liekanų QQ grafikas")
ax2.set_xlabel("Teoriniai kvantiliai")
ax2.set_ylabel("Stebėtos liekanos")

ax3.scatter(fitted, sl_y, s=20, color=color_pts)
mask = np.isfinite(fitted) & np.isfinite(sl_y)
if mask.sum() > 2:
    sm = lowess(sl_y[mask], fitted[mask], frac=0.6, return_sorted=True)
    ax3.plot(sm[:, 0], sm[:, 1], color=color_line, linewidth=2)
ax3.set_xlabel("Prognozuotos reikšmės")
ax3.set_ylabel("√|Standartizuotos liekanos|")
ax3.set_title("Scale-Location")

cooks = model2a.get_influence().cooks_distance[0]
idx = np.arange(1, len(cooks) + 1)
ax4.vlines(idx, 0, cooks, color=color_pts, linewidth=2)
ax4.axhline(4 / len(resid), color=color_line, linestyle="--")
ax4.set_xlabel("Stebėjimo indeksas")
ax4.set_ylabel("Cook's atstumas")
ax4.set_title("Įtakingi taškai (Cook's distance)")

plt.tight_layout()
plt.show()




#------------
infl2a = model2a.get_influence()
resid  = model2a.resid.to_numpy()
rstd   = infl2a.resid_studentized_internal
rstud  = infl2a.resid_studentized_external
cooks  = infl2a.cooks_distance[0]
n      = len(resid)
idx    = np.arange(1, n + 1)

color_pts = "#7A378B" 

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axes

ax1.scatter(idx, resid, s=20, color=color_pts)
ax1.axhline(0, color="black", linestyle="--")
ax1.set_xlabel("Stebėjimo indeksas")
ax1.set_ylabel("Liekanos")
ax1.set_title("Liekanos pagal stebėjimus")

ax2.scatter(idx, cooks, s=20, color=color_pts)
ax2.axhline(4 / n, color="black", linestyle="--")
ax2.set_xlabel("Stebėjimo indeksas")
ax2.set_ylabel("Cook'o atstumas")
ax2.set_title("Įtakingi taškai (Cook's distance)")

ax3.scatter(idx, rstd, s=20, color=color_pts)
for h in (-2, 0, 2):
    ax3.axhline(h, color="black", linestyle="--")
ax3.set_xlabel("Stebėjimo indeksas")
ax3.set_ylabel("Standartizuotos liekanos")
ax3.set_title("Standartizuotos liekanos")

ax4.scatter(idx, rstud, s=20, color=color_pts)
for h in (-3, 0, 3):
    ax4.axhline(h, color="black", linestyle="--")
ax4.set_xlabel("Stebėjimo indeksas")
ax4.set_ylabel("Studentizuotos liekanos")
ax4.set_title("Studentizuotos liekanos")

plt.tight_layout()
plt.show()




#------------
res2 = model2a.resid.dropna().to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(res2, dist="norm", plot=axes[0])
axes[0].lines[0].set_color("#7A378B") 
axes[0].lines[1].set_color("red")   
axes[0].lines[1].set_linewidth(2)
axes[0].set_title("Liekanų QQ grafikas")
axes[0].set_xlabel("Teoriniai kvantiliai")
axes[0].set_ylabel("Stebėtos liekanos")

axes[1].hist(res2, bins=30, density=True, color="#7A378B", edgecolor="black")
mu, sd = np.mean(res2), np.std(res2, ddof=1)
x = np.linspace(res2.min(), res2.max(), 200)
axes[1].plot(x, stats.norm.pdf(x, loc=mu, scale=sd), color="red", linewidth=2)
axes[1].set_title("Liekanų pasiskirstymas")
axes[1].set_xlabel("Liekanos")
axes[1].set_ylabel("Tankis")

plt.tight_layout()
plt.show()

print("\n=== MODELIS 2a ===")
A, p_ad = normal_ad(res2)
print("Anderson–Darling testas:")
print(f"A = {A:.6f}, p-reikšmė = {p_ad:.6g}")

W, p_sh = stats.shapiro(res2)
print("\nShapiro–Wilk testas:")
print(f"W = {W:.6f}, p-reikšmė = {p_sh:.6g}")


#-------------
bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model2a.resid, model2a.model.exog)
print("Breusch–Pagan testas (H0: homoskedastiškos liekanos)")
print(f"LM statistika = {bp_lm:.4f}, p-reikšmė = {bp_lm_p:.6g}")
print(f"F  statistika = {bp_f:.4f},  p-reikšmė = {bp_f_p:.6g}")

fitted = model2a.fittedvalues.to_numpy()
resid  = model2a.resid.to_numpy()

plt.figure(figsize=(7, 5))
plt.scatter(fitted, resid, s=25, color="#7A378B")
plt.axhline(0, color="black", linestyle="--", linewidth=2)

mask = np.isfinite(fitted) & np.isfinite(resid)
if mask.sum() > 3:
    sm = lowess(resid[mask], fitted[mask], frac=0.6, return_sorted=True)
    plt.plot(sm[:, 0], sm[:, 1], color="red", linewidth=2)

plt.xlabel("Prognozuotos reikšmės")
plt.ylabel("Liekanos")
plt.title("Liekanos ir prognozuotos reikšmės")
plt.tight_layout()
plt.show()

rob = model2a.get_robustcov_results(cov_type="HC3")
ci = rob.conf_int()
robust_table = pd.DataFrame({
    "coef": rob.params,
    "std err (HC3)": rob.bse,
    "t": rob.tvalues,
    "P>|t|": rob.pvalues,
    "CI low": ci[:, 0],
    "CI high": ci[:, 1],
})
print("\nKoeficientai su HC3 robust SE (model2a):")
print(robust_table)




#-------------

def outlierTest_py(model, df, alpha=0.05):
    """
    car::outlierTest analogas:
    - naudoja išorinai studentizuotas liekanas (rstudent)
    - skaičiuoja dvišales p ir Bonferroni p
    - grąžina lentelę ir atspausdina „mažiausią p“
    """
    infl = model.get_influence()
    rstud = infl.resid_studentized_external
    n = int(model.nobs)
    dof = int(model.df_resid)

    p_two = 2 * stats.t.sf(np.abs(rstud), df=dof)
    p_bonf = np.minimum(p_two * n, 1.0)

    out = (pd.DataFrame({
            "index": np.arange(1, n + 1),
            "rstudent": rstud,
            "p_two_sided": p_two,
            "p_bonferroni": p_bonf
        })
        .sort_values("p_two_sided", ascending=True, ignore_index=True))

    best = out.iloc[0]
    print("Bonferroni outlier test (pagal rstudent)")
    print(f"Mažiausia p: indeksas {int(best['index'])}, rstudent = {best['rstudent']:.4f}, "
          f"p = {best['p_two_sided']:.6g}, p_bonf = {best['p_bonferroni']:.6g}")
    return out

def influenceIndexPlot_py(model, top_n=3):
    """
    car::influenceIndexPlot analogas trims paneliams: Cook, Studentized, Hat.
    Pažymi TOP-N pagal |reikšmę|.
    """
    infl = model.get_influence()
    cooks = infl.cooks_distance[0]
    rstud = infl.resid_studentized_external
    hat   = infl.hat_matrix_diag
    n     = len(cooks)
    idx   = np.arange(1, n + 1)

    def _panel(ax, y, title, ylines=None):
        ax.scatter(idx, y, s=20, color="#7A378B")
        if ylines is not None:
            for y0 in ylines:
                ax.axhline(y0, color="black", linestyle="--", linewidth=1.0)
        top = np.argsort(np.abs(y))[-top_n:][::-1]
        for i in top:
            ax.scatter(idx[i], y[i], s=35, color="red", zorder=3)
            ax.annotate(str(idx[i]), (idx[i], y[i]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="red")
        ax.set_xlabel("Stebėjimo indeksas")
        ax.set_title(title)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    _panel(axes[0], cooks, "Įtakingi taškai (Cook's distance)", ylines=[4/n])
    axes[0].set_ylabel("Cook's atstumas")

    _panel(axes[1], rstud, "Studentizuotos liekanos", ylines=[-3, 0, 3])
    axes[1].set_ylabel("rstudent")

    _panel(axes[2], hat, "Leverage (hat reikšmės)")
    axes[2].set_ylabel("hat")
    axes[2].set_xlabel("Stebėjimo indeksas")

    plt.tight_layout()
    plt.show()

df2a_used = model2a.model.data.frame

out_tbl_2a = outlierTest_py(model2a, df2a_used)

influenceIndexPlot_py(model2a, top_n=3)



#-------------
print(model2a.summary())

rob = model2a.get_robustcov_results(cov_type="HC3")
ci = rob.conf_int()
coef_tbl = pd.DataFrame({
    "Terminas": rob.model.exog_names,
    "Koeficientas": rob.params,
    "Std. paklaida (HC3)": rob.bse,
    "t": rob.tvalues,
    "p": rob.pvalues,
    "CI low": ci[:, 0],
    "CI high": ci[:, 1],
})

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

coef_tbl["Žv."] = coef_tbl["p"].apply(stars)

order = list(coef_tbl.index)
if "Intercept" in coef_tbl["Terminas"].values:
    intercept_idx = coef_tbl.index[coef_tbl["Terminas"] == "Intercept"][0]
    order.remove(intercept_idx)
    order = [intercept_idx] + order

coef_tbl = coef_tbl.loc[order].reset_index(drop=True)
print("\nKoeficientai su HC3 robust SE (žvaigždutės pagal p):")
print(coef_tbl)
 
final_model = model2a





#--------------
formula_final = (
    "Q('CO2 emisijos (g/km)') ~ "
    "Q('Vidutinės kuro sąnaudos (l/100km)') * C(Q('Kuro tipas')) + "
    "Q('Vidutinės kuro sąnaudos (l/100km)') * C(Q('Kėbulo tipas')) + "
    "C(Q('Kuro tipas')) * C(Q('Pavarų dėžės tipas')) + "
    "C(Q('Markė'))"
)

need_cols = [
    "CO2 emisijos (g/km)",
    "Vidutinės kuro sąnaudos (l/100km)",
    "Kuro tipas", "Kėbulo tipas", "Pavarų dėžės tipas", "Markė"
]
df_final = data_clean.dropna(subset=need_cols).copy()

for c in ["Kuro tipas", "Kėbulo tipas", "Pavarų dėžės tipas", "Markė"]:
    df_final[c] = df_final[c].astype("category")

model_final = ols(formula=formula_final, data=df_final).fit()
print(model_final.summary())



#-------------
def anova_compare(restricted, full):
    """
    ANOVA F-testas tarp dviejų įdėtinių OLS modelių:
    - jei skirtingas nobs, perfitina ant bendrų eilučių
    - grąžina anova lentelę
    """
    n1, n2 = int(restricted.nobs), int(full.nobs)
    if n1 != n2:
        df_r = restricted.model.data.frame.copy()
        df_f = full.model.data.frame.copy()
        common_idx = df_r.index.intersection(df_f.index)

        fml_r = restricted.model.formula
        fml_f = full.model.formula

        r_refit = sm.OLS.from_formula(fml_r, data=df_r.loc[common_idx]).fit()
        f_refit = sm.OLS.from_formula(fml_f, data=df_f.loc[common_idx]).fit()
        return anova_lm(r_refit, f_refit)
    else:
        return anova_lm(restricted, full)

anova_res = anova_compare(model2a, model_final)
print(anova_res)




#------------
test_data_clean = test_data.loc[~test_data["Markė"].isin(["GMC", "SCION"])].copy()

for col in ["Markė", "Kuro tipas", "Pavarų dėžės tipas", "Kėbulo tipas"]:
    if col in test_data_clean.columns and col in data_clean.columns:
        train_cats = data_clean[col].astype("category").cat.categories
        test_data_clean[col] = pd.Categorical(test_data_clean[col], categories=train_cats)

test_pred = model_final.predict(test_data_clean)

results = pd.DataFrame({
    "Tikros": pd.to_numeric(test_data_clean["CO2 emisijos (g/km)"], errors="coerce"),
    "Prognozuotos": test_pred
})

n_before = len(results)
results_eval = results.dropna(subset=["Tikros", "Prognozuotos"]).copy()
n_dropped = n_before - len(results_eval)
if n_dropped > 0:
    print(f"Įspėjimas: {n_dropped} test eil. praleistos dėl trūkstamų reikšmių ar nepažintų kategorijų.")

print(results_eval.head())


#------------
if "results_eval" not in locals():
    results_eval = results.dropna(subset=["Tikros", "Prognozuotos"]).copy()

y_true = pd.to_numeric(results_eval["Tikros"], errors="coerce")
y_pred = pd.to_numeric(results_eval["Prognozuotos"], errors="coerce")

mask = np.isfinite(y_true) & np.isfinite(y_pred)
yt = y_true[mask].to_numpy()
yp = y_pred[mask].to_numpy()

if yt.size == 0:
    raise ValueError("Po NaN filtravimo neliko nei vienos eilutės. Patikrink `results` turinį.")


rmse_val = float(np.sqrt(np.mean((yt - yp) ** 2)))
mae_val  = float(np.mean(np.abs(yt - yp)))

if yt.size >= 2 and np.std(yt, ddof=1) > 0 and np.std(yp, ddof=1) > 0:
    r = float(np.corrcoef(yt, yp)[0, 1])
    r2_corr = r ** 2
else:
    r2_corr = np.nan

if yt.size >= 2 and np.var(yt, ddof=1) > 0:
    sse = float(np.sum((yt - yp) ** 2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    r2_reg = 1 - sse / sst
else:
    r2_reg = np.nan

print(f"RMSE: {rmse_val:.4f}")
print(f"MAE:  {mae_val:.4f}")
print(f"R² (cor^2): {r2_corr if np.isfinite(r2_corr) else 'n/a'}")
print(f"R² (1 - SSE/SST): {r2_reg if np.isfinite(r2_reg) else 'n/a'}")





#---------

df = results.dropna(subset=["Tikros", "Prognozuotos"]).copy()

x = df["Prognozuotos"].to_numpy()
y = df["Tikros"].to_numpy()

plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=40, facecolors="lightblue", edgecolors="darkblue", linewidths=0.8)
mn = np.nanmin([x.min(), y.min()])
mx = np.nanmax([x.max(), y.max()])
plt.plot([mn, mx], [mn, mx], color="red", linewidth=2)

plt.xlabel("Prognozuotos reikšmės")
plt.ylabel("Stebėtos reikšmės")
plt.title("Prognozuotos ir stebėtos reikšmės")
plt.tight_layout()
plt.show()



#------------
df_fit = model_final.model.data.frame.copy()

naujas = pd.DataFrame({
    "Vidutinės kuro sąnaudos (l/100km)": [8.5],
    "Pavarų dėžės tipas": ["Automatinė su rankiniu perjungimu"],
    "Kuro tipas": ["Įprastas benzinas"],
    "Kėbulo tipas": ["Lengvieji automobiliai"],
    "Markė": ["TOYOTA"],
})

for col in ["Pavarų dėžės tipas", "Kuro tipas", "Kėbulo tipas", "Markė"]:
    if col in df_fit.columns:
        cats = df_fit[col].astype("category").cat.categories
        naujas[col] = pd.Categorical(naujas[col], categories=cats)

pred = model_final.get_prediction(naujas)
sf = pred.summary_frame(alpha=0.05) 

rez = pd.DataFrame({
    "Prognozė": sf["mean"],
    "PI apatinė (95%)": sf["obs_ci_lower"],
    "PI viršutinė (95%)": sf["obs_ci_upper"],
})
print(rez.round(3))

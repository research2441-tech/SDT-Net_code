"""Spectral CT physics for the kidney-stone simulator and the spectral twin.

Polychromatic tungsten spectra (Kramers bremsstrahlung + K lines), Al/Sn filtration,
water-equivalent patient filtration, energy-integrating detector weighting and
NIST/Elam mass attenuation coefficients (xraydb). All attenuation is reported as
water-referenced Hounsfield units for a given protocol channel.
"""
import numpy as np
import xraydb

E_KEV = np.arange(15.0, 151.0, 1.0)

# mass fractions (ICRU-44 style compositions for soft tissues)
TISSUES = {
    "water":   ({"H": .111894, "O": .888106}, 1.000),
    "urine":   ({"H": .111894, "O": .888106}, 1.012),
    "adipose": ({"H": .114, "C": .598, "N": .007, "O": .278, "Na": .001, "S": .001, "Cl": .001}, 0.95),
    "kidney":  ({"H": .103, "C": .132, "N": .030, "O": .724, "Na": .002, "P": .002, "S": .002,
                 "Cl": .002, "K": .002, "Ca": .001}, 1.05),
    "blood":   ({"H": .102, "C": .110, "N": .033, "O": .745, "Na": .001, "P": .001, "S": .002,
                 "Cl": .003, "K": .002, "Fe": .001}, 1.06),
    "muscle":  ({"H": .102, "C": .143, "N": .034, "O": .710, "Na": .001, "P": .002, "S": .003,
                 "Cl": .001, "K": .004}, 1.05),
}

# stone crystal phases: chemical formula, crystal density (g/cm^3)
STONES = {
    "UA":  ("C5H4N4O3", 1.87),        # uric acid (anhydrous)
    "COM": ("CaC2O5H2", 2.21),        # calcium oxalate monohydrate
    "COD": ("CaC2O6H4", 1.94),        # calcium oxalate dihydrate
    "CAP": ("Ca5P3O13H", 3.16),       # (carbonate) apatite approximated by hydroxyapatite
    "STR": ("MgNH16PO10", 1.71),      # struvite MgNH4PO4.6H2O
    "CYS": ("C6H12N2O4S2", 1.68),     # cystine
    "BRU": ("CaPO6H5", 2.33),         # brushite CaHPO4.2H2O   (held out, OOD)
    "XAN": ("C5H4N4O2", 1.60),        # xanthine               (held out, OOD)
}
CLASSES = ["UA", "COM", "COD", "CAP", "STR", "CYS"]
OOD_CLASSES = ["BRU", "XAN"]

_ATOMIC_MASS = {}


def _parse_formula(f):
    import re
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def mass_fractions(formula):
    comp = _parse_formula(formula)
    m = {el: n * xraydb.atomic_mass(el) for el, n in comp.items()}
    tot = sum(m.values())
    return {el: v / tot for el, v in m.items()}


def mass_atten(fracs, E=E_KEV):
    """mass attenuation (cm^2/g) of a mixture given element mass fractions."""
    mu = np.zeros_like(E)
    for el, w in fracs.items():
        mu += w * xraydb.mu_elam(el, E * 1000.0)
    return mu


_CACHE = {}


def mu_spectrum(name):
    """linear attenuation (1/cm) versus energy for a named tissue or stone crystal."""
    if name in _CACHE:
        return _CACHE[name]
    if name in TISSUES:
        fr, rho = TISSUES[name]
    else:
        f, rho = STONES[name]
        fr = mass_fractions(f)
    v = mass_atten(fr) * rho
    _CACHE[name] = v
    return v


def mu_mass(name):
    """mass attenuation (cm^2/g)."""
    if name in TISSUES:
        return mu_spectrum(name) / TISSUES[name][1]
    return mu_spectrum(name) / STONES[name][1]


_FILT = {}


def _filt(el):
    if el not in _FILT:
        _FILT[el] = mass_atten({el: 1.0})
    return _FILT[el]


def tube_spectrum(kvp, al_mm=3.5, sn_mm=0.0):
    """Unfiltered Kramers + tungsten K lines, then Al/Sn filtration (photon fluence)."""
    E = E_KEV
    phi = np.where(E < kvp, (kvp - E) / E, 0.0)
    if kvp > 69.5:
        k = 0.06 * (kvp - 69.5) / 69.5
        for e0, w in ((59.3, 1.0), (58.0, 0.58), (67.2, 0.33)):
            phi[np.argmin(np.abs(E - e0))] += k * w * phi.sum() / 10.0
    phi = phi * np.exp(-_filt("Al") * 2.70 * al_mm / 10.0)
    if sn_mm > 0:
        phi = phi * np.exp(-_filt("Sn") * 7.31 * sn_mm / 10.0)
    return phi


def detector_weights(kvp, sn_mm, wed_cm):
    """energy-integrating weights after water-equivalent patient filtration."""
    phi = tube_spectrum(kvp, sn_mm=sn_mm)
    phi = phi * np.exp(-mu_spectrum("water") * wed_cm)
    w = phi * E_KEV
    return w / w.sum()


def mu_eff(mu_E, w):
    return float((mu_E * w).sum())


def hu_of(mu_E, w):
    muw = mu_eff(mu_spectrum("water"), w)
    return 1000.0 * (mu_eff(mu_E, w) - muw) / muw


def stone_mu(phases, fracs, packing):
    """linear attenuation (1/cm vs E) of a porous stone: solid phases mixed by volume
    fraction, remaining pore volume filled with urine."""
    v = np.zeros_like(E_KEV)
    for p, f in zip(phases, fracs):
        v += packing * f * mu_spectrum(p)
    v += (1.0 - packing) * mu_spectrum("urine")
    return v


# acquisition protocols (sites). channels: list of (kVp, Sn mm)
PROTOCOLS = {
    "A": dict(ch=[(80, 0.0), (140, 0.4)], fwhm=0.70, slice=0.75, noise=[22.0, 14.0], dev=True),
    "B": dict(ch=[(100, 0.0), (150, 0.6)], fwhm=0.80, slice=1.00, noise=[18.0, 15.0], dev=True),
    "C": dict(ch=[(120, 0.0)], fwhm=0.90, slice=2.00, noise=[16.0], dev=True),
    "D": dict(ch=[(90, 0.0), (150, 0.4)], fwhm=0.60, slice=1.50, noise=[30.0, 24.0], dev=False),
    "E": dict(ch=[(120, 0.0)], fwhm=1.10, slice=3.00, noise=[30.0], dev=False),
}


def protocol_weights(site, wed_cm):
    return [detector_weights(k, s, wed_cm) for k, s in PROTOCOLS[site]["ch"]]


if __name__ == "__main__":
    for site in "ABCDE":
        ws = protocol_weights(site, 28.0)
        row = []
        for st in CLASSES + OOD_CLASSES:
            h = [hu_of(stone_mu([st], [1.0], 0.80), w) for w in ws]
            row.append(f"{st}:" + "/".join(f"{x:.0f}" for x in h))
        print(site, " ".join(row))

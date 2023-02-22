import uproot
import numpy as np
from scipy.special import erfc
import iminuit



def collect_data(sources, ideal=False, Nbins=np.linspace(1, 2400, 600), normed=False):
    totalPEs = []
    source_counts = []
    source_bins = []
    nHits_array = []
    edepScint_MeV_array = []
    for source in sources:
        if ideal:
            file = uproot.open(f'data/lnl{source}I.root')
        else:
            file = uproot.open(f'data/lnl{source}.root')
        totalPE = np.array(file['evt']['totalPE'].array())
        nHits = np.array(file['evt']['nHits'].array())
        edepScint_MeV = np.array(file['evt']['edepScint_MeV'].array())

        print(f"{source} totalPE mean: {totalPE.mean()}")
        totalPEs.append(totalPE)
        nHits_array.append(nHits)
        edepScint_MeV_array.append(edepScint_MeV)

        counts, bins = np.histogram(totalPE, bins=Nbins, normed=normed)
        bins = 0.5 * (bins[:-1] + bins[1:])
        source_counts.append(counts)
        source_bins.append(bins)
    
    pmt_x = np.array(file['pmt_pos']['pmtX_cm'].array())
    pmt_y = np.array(file['pmt_pos']['pmtY_cm'].array())
    pmt_z = np.array(file['pmt_pos']['pmtZ_cm'].array())
        
    return source_counts, source_bins, totalPEs, nHits_array, edepScint_MeV_array


def get_pmt_coors():
    
    file = uproot.open(f'data/lnlCo60I.root')
    pmt_x = np.array(file['pmt_pos']['pmtX_cm'].array())
    pmt_y = np.array(file['pmt_pos']['pmtY_cm'].array())
    pmt_z = np.array(file['pmt_pos']['pmtZ_cm'].array())
    return pmt_x, pmt_y, pmt_z


def cylinder(r, h):
    theta = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(-20, h-20, 100)
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z


def comp_multiplicity(nHits, naxis=2, Nbins=48):
    source_counts = []
    source_bins = []

    multiplicities = np.count_nonzero(nHits, axis=naxis)
    for multiplicity in multiplicities:
        counts, bins = np.histogram(multiplicity, bins=np.linspace(1, max(multiplicity), max(multiplicity)))
        bins = 0.5 * (bins[:-1] + bins[1:])
        
        source_counts.append(counts)
        source_bins.append(bins)

    return source_counts, source_bins

def comp_corrected_mult(sources, factor=20/257, ideal=False):
    if ideal:
        factor = 20/1213
    source_counts, source_bins, totalPEs, nHits_array = collect_data(sources, ideal=ideal)
    nHits_corrected = nHits_array[0] * factor
    nHits_poisson = np.random.poisson(nHits_corrected)
    corrected_multiplicity = np.count_nonzero(nHits_poisson, axis=1)
    counts, bins = np.histogram(corrected_multiplicity, bins=np.linspace(
        1, max(corrected_multiplicity), max(corrected_multiplicity)
    ))
    bins = 0.5 * (bins[:-1] + bins[1:])
    
    return counts, bins

def fit_data(source_counts, source_bins, x_mins, x_maxs, mu0s, 
             sigmas, exps, labels, colors, param_names):

    def cost_function(theta):
        mu, sigma, rate, bkg = theta
        y = fit_function(x_data, mu, sigma, rate, bkg)
        return chi2(y_data[mask_fit], y[mask_fit])

    fit_outputs = []
    ys_fit = []
    masks_fit = []
    for i in range(len(source_counts)):
        x_data = source_bins[i]
        y_data = source_counts[i]
        
        x_min = x_mins[i]
        x_max = x_data.max()
        mask_fit = np.logical_and(x_data>=x_mins[i], x_data<=x_maxs[i])
        masks_fit.append(mask_fit)
        theta0 = mu0s[i], sigmas[i], 0.005, 100

        
        m = iminuit.Minuit(cost_function, theta0, name=param_names)
        m.errordef = m.LEAST_SQUARES
        m.limits['mu'] = (0, None)
        m.limits['sigma'] = (0, None)
        m.limits['rate'] = (0, None)
        m.limits['bkg'] = (0, None)
        m.migrad()

        fit_output = {}
        fit_output["mu"] = m.params['mu'].value
        fit_output['mu_std'] = m.errors[0]
        fit_output["sigma"] = m.params['sigma'].value
        fit_output["rate"] = m.params['rate'].value
        fit_output["bkg"] = m.params['bkg'].value
        fit_output["label"] = labels[i]
        fit_output["exp"] = exps[i]
        fit_outputs.append(fit_output)

        y_fit = fit_function(x_data, *m.values)
        ys_fit.append(y_fit)
        
    return fit_outputs, ys_fit, masks_fit


def fit_function(x, mu, sigma, rate, bkg):
    y = rate*erfc((x-mu)/sigma) + bkg
    return y


def chi2(x, mu):
    return ((x-mu)**2/mu).sum()



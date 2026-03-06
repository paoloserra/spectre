# Import modules on top
import numpy as np
import argparse, sys
from astropy.io import fits
from matplotlib import pyplot as plt
import shutil

# Define functions

# read arguments
def create_parser():
  p = argparse.ArgumentParser()
  p.add_argument("-c", "--cube", type=str, required=True,
                 help="Input FITS cube.")
  p.add_argument("-m", "--mask", type=str,
                 help="Optional FITS detection mask. Spectra including detected voxels"
                 " are excluded from the analysis.")
  p.add_argument("-nspec", "--nr-spec", type=int, default=1000,
                 help="Number of unique random spectra to be extracted from the input FITS"
                 " cube. Default = 1000.")
  p.add_argument("-nchan", "--nr-chan", type=int, default=0,
                 help="Number of channels per spectrum. Default = 0 = all channels.")
  p.add_argument("-o", "--output", type=str, default=None,
                 help="Name of the output plot including the extension. Default = None"
                 " = system interactive backend.")
  p.add_argument("-sinc", "--sinc-kernel", type=float, nargs='+', required = False,
                 help="Space-separated list of scales for comparison Sinc kernels.")
  p.add_argument("-gauss", "--gauss-kernel", type=float, nargs='+', required = False,
                 help="Space-separated list of widths for comparison Gaussian kernels.")
  p.add_argument("-hann", "--hanning-kernel", type=int, nargs='+', required = False,
                 help="Space-separated list of widths for comparison Hanning kernels."
                 " Only odd numbers will be considered.")
  p.add_argument("-box", "--boxcar-kernel", type=int, nargs='+', required = False,
                 help="Space-separated list of widths for comparison Box kernels."
                 " Only odd numbers will be considered.")
  p.add_argument("-bin", "--binomial-kernel", type=int, nargs='+', required = False,
                 help="Space-separated list of widths for comparison Binomial kernels."
                 " Only odd numbers > 1 will be considered.")
  # --------------------------------- ADVANCED OPTIONS --------------------------------- #
  p.add_argument("-artlen", "--artefacts-autocorr-length", type=int, default=-1,
                 help="*** ADVANCED OPTION ***"
                 " Autocorrelation coefficients from this length on are assumed to be"
                 " due to artefacts (e.g., from continuum subtraction). We fit them with a"
                 " polynomial of order set by -artord, and subtract the result from"
                 " <A_F> before calculating K. Default = 0 (or any value < 0) means that"
                 " the data cube is assumed to have no artefatcs.")
  p.add_argument("-artord", "--artefacts-autocorr-order", type=int, default=2,
                 help="*** ADVANCED OPTION ***"
                 " Polynomial order used when fitting <A_F> beyond the length set by"
                 " -artlen in order to model the autocorrelation caused by artefacts."
                 " Default - 2.")
  p.add_argument("-notrack", "--no-track-sign-change", action="store_true",
                 help="*** ADVANCED OPTION ***"
                 " Skip the sign tracking of +/- sqrt[FT(<A_F>)]. This means always taking"
                 " the + sign unless the user forces the sign change at specific points"
                 " with -force.")
  p.add_argument("-trackpar", "--track-sign-change-params", type=float, nargs=5,
                 default=[15, 0.7, 0.5, 0.1, 5], help="*** ADVANCED OPTION ***"
                 " Space-separated parameters of the algorithm that tracks the sign of"
                 "  +/- sqrt[FT(<A_F>)]. The algorithm looks for local minima sufficiently"
                 " close to zero. The five parameters are:"
                 " 1) Half-width of the window used to check whether a point is a local"
                 " minimum, to be rounded to the nearest integer. Default = 15."
                 " 2) Minimum fraction of points with a larger value than the central point"
                 " inside the above window. Must be between 0 and 1, with higher values"
                 " giving fewer sign changes. Default = 0.7."
                 " 3) Minumum fraction of points with a negative (positive) 1st derivative"
                 " on the side closer to (farther from) the zero-frequency term. Must be"
                 " between 0 and 1, with higher values giving fewer sign changes. Default"
                 " = 0.5."
                 " 4) Maximum ratio between the point and the peak. Must be between 0 and 1,"
                 " with lower values giving fewer sign changes. Default = 0.1."
                 " 5) Minimum distance from another local minimum candidate, to be rounded"
                 " to the nearest integer. Points closer than this limit are grouped"
                 " together in a friends-of-friends way, and only the median point of"
                 " each group is retained). Default = 5.")
  p.add_argument("-force", "--force-sign-change", type=int, nargs='+', required = False,
                 help="*** ADVANCED OPTION *** Space-separated list of points where to"
                 " force a sign change of sqrt[FT(<A_F>)]. Can only be used with -notrack,"
                 " because the sign tracking algorithm would generally give a conflicting"
                 " list of points.")
  p.add_argument("-interp", "--interp-sign-change", type=int, nargs=3,
                 default=[10, 2, 10], help="*** ADVANCED OPTION ***"
                 " Space-separated parameters used for interpolating sqrt[FT(<A_F>)]"
                 " across the sign-changing points, whether found by the sign tracking"
                 " algorithm or given by the user. Interpolation avoids abrupt jumps"
                 " across the zero line in case of significan noise floor. The three"
                 " parameters are:"
                 " 1) Number of points to be included in the interpolation on either side"
                 " of the point. Default = 10. Set to 0 for no interpolation."
                 " 2) Order of the fit. Default = 2."
                 " 3) Number of points to be excluded from the interpolation on either"
                 " side of the point. If > 0, the above number of points included in the"
                 " interpolation does not change, and the points included move outward."
                 " Default = 10.")
  p.add_argument("-floor", "--noise-floor", type=int, default=-1,
                 help="*** ADVANCED OPTION ***"
                 " Calculate the noise floor of FT(<A_F>) as the selected percentile"
                 " (0 to 100), and subtract it before reconstructing the kernel K."
                 " Default = -1 = no noise floor removal.")
  return(p)

# make a spectrum symmetric about mid point
def symmetrize(x):
  for ii in range(int(np.around(x.shape[0]/2,2))):
    x[ii], x[x.shape[0]-ii-1] = (x[ii]+x[x.shape[0]-ii-1])/2, (x[ii]+x[x.shape[0]-ii-1])/2
  return(x)

# sinc
def sinc_kern(z,scale):
  kern = np.sinc(z*scale)
  return(kern / np.nanmax(kern))

# gaussian
def gauss_kern(z,sig):
  kern = np.exp(-z**2 / 2 / sig**2)
  return(kern / np.nanmax(kern))

# hanning
def hann_kern(z,width):
  kern = np.zeros(z.shape)
  nch = int((z.shape[0] - 1) / 2)
  kern[nch-int((width+1)/2):nch+int((width+1)/2)+1] = np.hanning(width+2)
  return(kern / np.nanmax(kern))

# box
def box_kern(z,width):
  kern = np.zeros(z.shape)
  nch = int((z.shape[0] - 1) / 2)
  for ww in range(int((width-1)/2)+1):
    kern[nch-ww] = 1.00
    kern[nch+ww] = 1.00
  return(kern / np.nanmax(kern))

# binomial
def binomial_kern(z,width):
  bin2 = np.pad(np.ones(2), 2*width)
  kern = bin2.copy()
  ll = 2
  while ll < width:
    kern = convolve_fft(kern, bin2)
    kern[kern < 1e-6] = 0
    ll += 1
  kern = kern[kern > 0]
  kern = np.pad(kern, max(0,(z.shape[0]-kern.shape[0])//2))
  return(kern / np.nanmax(kern))


# DFT-based autocorrelation
def autocorrelate_fft(signal):
  min_fft_length = 2 * signal.shape[0] - 1
  fft_length = 1
  while fft_length < min_fft_length:
    fft_length *= 2
  signal_fft     = np.fft.fft(signal,       n=fft_length)
  inv_signal_fft = np.fft.fft(signal[::-1], n=fft_length)
  signal_psd = signal_fft * inv_signal_fft
  signal_autocorr = np.real(np.fft.ifft(signal_psd))
  signal_autocorr = signal_autocorr[(signal.shape[0]-1) // 2 : (signal.shape[0]-1) // 2 + signal.shape[0]]
  signal_autocorr /= np.nanmax(np.abs(signal_autocorr))
  return(signal_autocorr)

# DFT-based convolution
def convolve_fft(signal, kern):
  # tested that this retunrs a convolved signal centred as the input signal
  min_fft_length = signal.shape[0] + kern.shape[0] - 1
  fft_length = 1
  while fft_length < min_fft_length:
    fft_length *= 2
  signal_fft = np.fft.fft(signal, n=fft_length)
  kern_fft   = np.fft.fft(kern,   n=fft_length)
  convolved_signal = np.real(np.fft.ifft(signal_fft * kern_fft))
  convolved_signal = convolved_signal[(kern.shape[0]-1) // 2 : (kern.shape[0]-1) // 2 + signal.shape[0]]
  return(convolved_signal)

# function to change the sign at the position of the selected local minima, with optional interpolation to smooth over jumps
def change_sign(uval, upos, interp_incl, interp_order, interp_excl):
  uval_new = uval.copy()
  uneg = uval_new.shape[0]-upos-1
  uval_new[upos:] *= -1
  uval_new[:uneg+1] *= -1
  if interp_incl:
    ux0 = np.arange(upos-interp_excl,upos+interp_excl)
    ux1 = np.arange(upos-interp_excl-interp_incl,upos-interp_excl)
    ux2 = np.arange(upos+interp_excl,upos+interp_excl+interp_incl)
    uy1 = uval_new[upos-interp_excl-interp_incl:upos-interp_excl]
    uy2 = uval_new[upos+interp_excl:upos+interp_excl+interp_incl]
    coeffs = np.polyfit(np.concatenate((ux1,ux2)), np.concatenate((uy1,uy2)), interp_order)[::-1]
    uval_new[upos-interp_excl:upos+interp_excl] = np.zeros(ux0.shape)
    for oo in range(coeffs.shape[0]):
      uval_new[upos-interp_excl:upos+interp_excl] += coeffs[oo] * ux0**oo
    ux0 = np.arange(uneg-interp_excl+1,uneg+interp_excl+1)
    ux1 = np.arange(uneg-interp_excl-interp_incl+1,uneg-interp_excl+1)
    ux2 = np.arange(uneg+interp_excl+1,uneg+interp_excl+interp_incl+1)
    uy1 = uval_new[uneg-interp_excl-interp_incl+1:uneg-interp_excl+1]
    uy2 = uval_new[uneg+interp_excl+1:uneg+interp_excl+interp_incl+1]
    coeffs = np.polyfit(np.concatenate((ux1,ux2)), np.concatenate((uy1,uy2)), interp_order)[::-1]
    uval_new[uneg-interp_excl+1:uneg+interp_excl+1] = np.zeros(ux0.shape)
    for oo in range(coeffs.shape[0]):
      uval_new[uneg-interp_excl+1:uneg+interp_excl+1] += coeffs[oo] * ux0**oo
  return(uval_new)

# sign tracking function
def track_ft_sign_smooth(x, track_sign_par, inter_sign_change, pos_sign_change, zlab, max_sign_change=1000, verbose=0):
  # Starting from the centre, find a point that meets these requirements:
  # 1) most neighbours within a symmetric window have a larger value
  #    (fraction neighbours > local_min_frac);
  # 2) most neighbours within a symmetric window have negative fist derivative on one side,
  #    and positive on the other side (fraction neighbours > local_d1_frac);
  # 3) the point is close to zero (< peak_frac * peak);
  # 4) if the sign of this point is changed, most neighbours within a symmetric window have
  #    negative fist derivative on both sides (fraction neighbours > local_d1_frac);
  # 5) if the sign of this point is changed, condition 1 is no longer satisfied.
  # To avoid frequent sign changes caused by noise, group sign-changing points based on
  #    min_gap and select only the median point of each group.
  # Stop when the number of sign changes reaches max_sign_change.
  # For the final selection of points, smooth across the sign-changing region with
  #    interp_excl, interp_incl and interp_order.

  [window, local_min_frac, local_d1_frac, peak_frac, min_gap] = track_sign_par
  [interp_incl, interp_order, interp_excl] = inter_sign_change

  x = np.fft.fftshift(x)
  dx = x[1:] - x[:-1]
  dxmed = np.nanmedian(dx)
  dxstd = 1.4826 * np.nanmedian(np.abs(dx - dxmed))
  
  if not pos_sign_change:
    pos_sign_change = []
  else:
    pos_sign_change = [sc + x.shape[0]//2 for sc in pos_sign_change]
  if not len(pos_sign_change):
    print('#   - track sign change of sqrt[FT(<A_F>{0:s})]'.format(zlab))
    # start loop, moving from the centre towards high channels
    for jj in np.arange(x.shape[0]//2, x.shape[0]-max(window, interp_excl+interp_incl)):
      # condition 1
      if (x[jj-window:jj+window+1] > x[jj]).sum() >= local_min_frac * (2 * window + 1):
        if verbose:
          print('###      .',jj)
        # condition 2
        if (dx[jj-window:jj] < 0).sum() >= local_d1_frac * window and (dx[jj+1:jj+1+window] > 0).sum() >= local_d1_frac * window:
          if verbose:
            print('###      ..')
          # condition 3
          if np.abs(x[jj]) < peak_frac * np.nanmax(np.abs(x)):
            if verbose:
              print('###      ...')
            xtemp = change_sign(x, jj, 0, 0, 0) # no sign-change interpolation when going through the 5 conditions
            dxtemp = xtemp[1:] - xtemp[:-1]
            # condition 4
            if (dxtemp[jj-window:jj] < 0).sum() >= local_d1_frac * window and (dxtemp[jj+1:jj+1+window] < 0).sum() >= local_d1_frac * window:
              if verbose:
                print('###      ....')
              # condition 5
              if (xtemp[jj-window:jj+window+1] > xtemp[jj]).sum() < local_min_frac * (2 * window + 1):
                if verbose:
                  print('###      .....')
                pos_sign_change.append(jj)
              if len(pos_sign_change) == max_sign_change:
                print('###      !!! maximum number of sign changes reached !!!')
                break
    if len(pos_sign_change):
      jj=1
      groups_sign_change = [[pos_sign_change[0],],]
      while jj < len(pos_sign_change):
        if pos_sign_change[jj] - pos_sign_change[jj-1] < min_gap:
          groups_sign_change[-1].append(pos_sign_change[jj])
        else:
          groups_sign_change.append([pos_sign_change[jj],])
        jj += 1
      pos_sign_change = [int(np.around(np.median(np.array(jj)),0)) for jj in groups_sign_change]
  
  if len(pos_sign_change):
    print('#   - change sign of sqrt[FT(<A_F>)] at {0}'.format(np.array(pos_sign_change)-x.shape[0]//2))
    if interp_incl:
      print('#   - interpolate across sign changes')
  else:
    print('#   - no sign changes found')
  for jj in pos_sign_change:
    x = change_sign(x, jj, interp_incl, interp_order, interp_excl)

  x = np.fft.ifftshift(x)
  return(x)


def main():


  # Read settings from command line
  args       = create_parser().parse_args([a for a in sys.argv[1:]])
  cubef      = args.cube
  maskf      = args.mask
  nr_chan    = args.nr_chan
  nr_spec    = args.nr_spec
  output     = args.output
  sinc       = args.sinc_kernel
  gauss      = args.gauss_kernel
  hann       = args.hanning_kernel
  box        = args.boxcar_kernel
  binom      = args.binomial_kernel
  # --------- ADVANCED OPTIONS --------- #
  art_len    = args.artefacts_autocorr_length
  art_ord    = args.artefacts_autocorr_order
  noise_f    = args.noise_floor
  track_sign = not args.no_track_sign_change
  track_par  = args.track_sign_change_params
  force_sign = args.force_sign_change
  inter_sign = args.interp_sign_change

  
  plt.rcParams['text.usetex']= True if shutil.which('latex') else False
  plt.rcParams.update({'font.size': 16})
  legend_font_size = 10

  # Check that the settings are valid
  if noise_f < -1 or noise_f > 100:
    print('# ERROR: invalid value for -floor/--noise-floor: {0:d}. Only integers between -1 and +100 are allowed.'.format(noise_f))
    sys.exit()

  if track_par[1] < 0 or track_par[1] > 1:
    print('# ERROR: invalid value for the second parameter value in -trackpar/--track-sign-change-params: {0:.f}. Only floats between 0 and 1 are allowed.'.format(track_par[1]))
    sys.exit()
  elif track_par[2] < 0 or track_par[2] > 1:
    print('# ERROR: invalid value for the third parameter value in -trackpar/--track-sign-change-params: {0:.f}. Only floats between 0 and 1 are allowed.'.format(track_par[2]))
    sys.exit()
  elif track_par[3] < 0 or track_par[3] > 1:
    print('# ERROR: invalid value for the fourth parameter value in -trackpar/--track-sign-change-params: {0:.f}. Only floats between 0 and 1 are allowed.'.format(track_par[3]))
    sys.exit()
  else:
    track_par[0] = int(np.around(track_par[0]))
    track_par[4] = int(np.around(track_par[4]))

  if force_sign and track_sign:
    print('# ERROR: can only use -force/--force-sign-change together with -notrack/--no-track-sign-change.'.format(noise_f))
    sys.exit()

  # Load the input FITS cube and, if requested, the FITS detection mask
  print('# Loading FITS cube {0:s}'.format(cubef))
  with fits.open(cubef) as f:
      cube = f[0].data
      if len(cube.shape) == 4 and cube.shape[0] == 1:
        cube = cube[0]
  if maskf:
    print('# Loading FITS detection mask {0:s} as boolean array'.format(maskf))
    with fits.open(maskf) as f:
      msk = f[0].data.astype(bool)
      if len(msk.shape) == 4 and msk.shape[0] == 1:
        msk = msk[0]
  else:
    print('# WARNING: No FITS detection mask given. Will assume the cube is pure noise.')
    msk = np.full(cube.shape, False, dtype=bool)

  # Initialise a few things
  if nr_spec <= cube.shape[1]*cube.shape[2]:
    print('# Will extract {0:d} unique random spectra from the input cube ({1:d} available).'.format(nr_spec, cube.shape[1]*cube.shape[2]))
  else:
    print('# ERROR: You are requesting more spectra ({0:d}) than available in cube ({1:d}). Please change this with the -ns option.'.format(nr_spec, cube.shape[1]*cube.shape[2]))
    sys.exit()
  if nr_chan:
    nr_chan = 2 * (nr_chan // 2) + 1
    print('# Will take {0:d} channels per spectrum ({1:d} available).'.format(nr_chan, cube.shape[0]))
    if cube.shape[0] < nr_chan:
      nr_chan = 2 * ((cube.shape[0]-1) // 2) + 1
      print('# WARNING: Number of channels per spectrum modified to {0:d} to fit within the spectral axis of the input cube.'.format(nr_chan))
  else:
    nr_chan = 2 * ((cube.shape[0]-1) // 2) + 1
    print('# Will take {0:d} channels per spectrum ({1:d} available).'.format(nr_chan, cube.shape[0]))
  spec_z = np.arange(-nr_chan//2+1,nr_chan//2+1)
  spec_autocorr_all = np.zeros((nr_spec,nr_chan))

  # Extract random spectra from cube, with some constraints:
  # - exclude spectra already extracted
  # - exclude spectra included in the mask
  # - exclude spectra with NaN's
  print('# Extracting {0:d} unique random spectra F and calculating autocorrelation A_F.'.format(nr_spec))
  ii, skipped = 0, 0
  while ii < nr_spec:
    if skipped > 10 * nr_spec:
      print('ERROR: Cannot find enough unique random spectra sufficiently quickly. Try to lower your request with the -ns option.')
      sys.exit()
    x0 = np.random.randint(0,high=cube.shape[2])
    y0 = np.random.randint(0,high=cube.shape[1])
    z0 = np.random.randint(0,high=cube.shape[0]-nr_chan+1)
    spec = cube[z0:z0+nr_chan,y0,x0]
    if not msk[z0:z0+nr_chan,y0,x0].sum() and not np.isnan(spec).sum():
      spec_autocorr_all[ii] = autocorrelate_fft(spec) # peak = 1 at centre of array
      msk[z0:z0+nr_chan,y0,x0] = True
      ii += 1
    else:
      skipped += 1
  print('# Calculating mean autocorrelation <A_F>.')
  spec_autocorr_mean = np.nanmean(spec_autocorr_all, axis=0)
  spec_autocorr_std  = np.nanstd(spec_autocorr_all, axis=0)

  # Fit <A_F> beyond art_len with a polynomial of order art_ord. The result will be later subtracted from <A_F>
  art_autocorr = np.zeros(spec_z.shape)
  if art_len > 0:
    if art_ord == 1:
      art_ord_lab = 'st'
    if art_ord == 2:
      art_ord_lab = 'nd'
    if art_ord == 3:
      art_ord_lab = 'rd'
    else:
      art_ord_lab = 'th'
    print('# Estimating autocorrelation Z caused by artefacts with a {0:d}{1:s} order polynomial fit to <A_F> beyond scale {2:d}'.format(art_ord, art_ord_lab, art_len))
    art_coeffs = np.polyfit(spec_z[nr_chan//2+art_len:], spec_autocorr_mean[nr_chan//2+art_len:], art_ord, w=1./spec_autocorr_std[nr_chan//2+art_len:])[::-1]
    for oo in range(art_coeffs.shape[0]):
      art_autocorr[nr_chan//2:] += art_coeffs[oo] * spec_z[nr_chan//2:]**oo
    art_autocorr[:nr_chan//2] = art_autocorr[nr_chan//2+1:][::-1]
  Z_label = '-Z' if art_len > 0 else ''

  # Calculate maximum autocorrelation length with coefficients significantly above zero (or above the artefacts fit)
  max_nonzero_autocorr = max(3,np.max(np.abs(np.where((spec_autocorr_mean - art_autocorr) > spec_autocorr_std)[0] - nr_chan//2)))
    
  # Compare mean autocorrelation to autocorrelation of requested kernels
  kernels, kern_autocorr, knames, deltas = {}, {}, [], []
  delta_tol = 3.
  if sinc or gauss or hann or box or binom:
    print('# Comparing mean autocorrelation <A_F>{0:s} with autocorrelation A_K of known convolution kernels:'.format(Z_label))
    print('#   delta calculated with the first {0:d} elements of <A_F>{1:s} after the peak'.format(2*max_nonzero_autocorr, Z_label))
    if sinc:
      for ss in sinc:
        kernels['sinc-{0:.2f}'.format(ss)] = sinc_kern(spec_z, ss)
    if gauss:
      for gg in gauss:
        kernels['gaussian-{0:.2f}'.format(gg)] = gauss_kern(spec_z, gg)
    if hann:
      for hh in hann:
        if hh // 2 * 2 != hh:
          kernels['hanning-{0:d}'.format(hh)] = hann_kern(spec_z, hh)
    if box:
      for bb in box:
        if bb // 2 * 2 != bb:
          kernels['boxcar-{0:d}'.format(bb)] = box_kern(spec_z, bb)
    if binom:
      for nn in binom:
        if nn > 1 and nn // 2 * 2 != nn:
          kernels['binomial-{0:d}'.format(nn)] = binomial_kern(spec_z, nn)
    for kk in kernels:
      kern_autocorr[kk] = autocorrelate_fft(kernels[kk])
      knames.append(kk)
      deltas.append(np.nanmean(((spec_autocorr_mean-art_autocorr)[nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1] - kern_autocorr[kk][nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1])**2 / spec_autocorr_std[nr_chan//2+1:nr_chan//2+2*max_nonzero_autocorr+1]**2))
    knames, deltas = np.array(knames), np.array(deltas)
    knames, deltas = knames[np.argsort(deltas)], deltas[np.argsort(deltas)]
    print('#   {0:15s} {1:8s}   {2}'.format('kernel-name','delta', 'area'))
    for kk in range(len(knames)):
      if kk < 3 and deltas[kk] < delta_tol * deltas.min():
        print('#   {0:15s} {1:8.2e}   {2:.2f}  (plotted)'.format(knames[kk], deltas[kk], kernels[knames[kk]].sum()))
      else:
        print('#   {0:15s} {1:8.2e}'.format(knames[kk], deltas[kk]))
    # Select kernels to plot (delta within a factor of delta_tol of best delta, and no more than 3 kernels)
    knames = knames[deltas < delta_tol * deltas.min()][:3]


  # Core calculation: from mean autocorrelation to kernel
  print('# Reconstructing spectral convolution kernel K from mean autocorrelation <A_F>{0:s}:'.format(Z_label))
  print('#   - FT(<A_F>{0:s})'.format(Z_label))
  rec_kernel_psd = np.real(np.fft.fft(np.fft.ifftshift(spec_autocorr_mean-art_autocorr))) # note that the kernel is reordered before taking its FT
  if noise_f != -1:
    print('#   - remove noise floor from FT(<A_F>{0:s}) defined as {1:d}-th percentile'.format(Z_label, noise_f))
    rec_kernel_psd -= np.percentile(rec_kernel_psd, noise_f) # remove noise floor
  rec_kernel_psd[rec_kernel_psd<0] = 0 # the power spectrum is >=0 by definition
  rec_kernel_fft = np.sqrt(rec_kernel_psd)
  rec_kernel_fft /= np.nanmax(np.abs(rec_kernel_fft))
  if track_sign or force_sign:
    rec_kernel_fft_sign = track_ft_sign_smooth(rec_kernel_fft, track_par, inter_sign, force_sign, Z_label)
  print('#   - K = IFT{{+/- sqrt[FT(<A_F>{0:s})]}}'.format(Z_label))
  if track_sign or force_sign:
    rec_kernel = np.real(np.fft.ifft(rec_kernel_fft_sign))
  else:
    rec_kernel = np.real(np.fft.ifft(rec_kernel_fft))
  rec_kernel = np.fft.ifftshift(rec_kernel)
  rec_kernel /= np.nanmax(rec_kernel)
  rec_kernel = np.roll(rec_kernel, -1)

  # Find asymptotic area
  rec_area = np.array([rec_kernel[nr_chan//2-aa:nr_chan//2+aa+1].sum() for aa in range(nr_chan//2)])
  ii_area = max(9,nr_chan//20)
  rec_area_std = np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:])))
  while ii_area <= nr_chan//2 and np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:]))) < 1.1 * rec_area_std:
    ii_area += 1
  ii_area -= 1
  rec_area_std = np.median(np.abs(rec_area[-ii_area:] - np.median(rec_area[-ii_area:])))
  rec_area_med = np.median(rec_area[-ii_area:])
  print('# Kernel area = integral(K/K_max) = {0:.2f} channels (best guess of asymptotic value = median of last {1:d} channels).'.format(rec_area_med, ii_area))

  # Additional variables for plotting
  spec_autocorr_p16  = np.nanpercentile(spec_autocorr_all, 16, axis=0)
  spec_autocorr_p84  = np.nanpercentile(spec_autocorr_all, 84, axis=0)
  kcolors = ['orange', 'green', 'blue']

  # Plotting
  fig = plt.figure(figsize=(9,8))
  gs = fig.add_gridspec(3, 2, height_ratios=[1,1,1])
  ax0 = plt.subplot(gs[0,:])
  ax1 = plt.subplot(gs[1,0])
  ax2 = plt.subplot(gs[1,1])
  ax3 = plt.subplot(gs[2,0])
  ax4 = plt.subplot(gs[2,1])

  ax0.axhline(y=0, color='k', ls=':')
  ax0.plot(spec_z, spec_autocorr_mean, 'k-', ds='steps-mid', label='$\\langle A_F \\rangle $ from {0:d} spectra'.format(nr_spec), lw=3)
  ax0.fill_between(spec_z, spec_autocorr_p16, spec_autocorr_p84, color='k', alpha=0.3, step='mid', label='$16^\\mathrm{th}$ - $84^\\mathrm{th}$ perc.')
  ymin = np.nanmin(spec_autocorr_p16[nr_chan//2+min(nr_chan//2,2*max_nonzero_autocorr):])
  ymax = np.nanmax(spec_autocorr_p84[nr_chan//2+min(nr_chan//2,2*max_nonzero_autocorr):])
  if art_len > 0:
    ax0.plot(spec_z[nr_chan//2+art_len:], art_autocorr[nr_chan//2+art_len:], 'r-', label='$Z$ = artefacts (order {0:d})'.format(art_ord), lw=1)
    ax0.plot(spec_z[:nr_chan//2+art_len+1], art_autocorr[:nr_chan//2+art_len+1], 'r--', lw=1)
    ax0.plot([art_len, art_len], [art_autocorr[nr_chan//2+art_len] - 0.1*(ymax-ymin), art_autocorr[nr_chan//2+art_len] + 0.1*(ymax-ymin)], 'r--', lw=1)
  ax0.legend(fontsize=legend_font_size, ncols=3)
  ax0.set_xlim(0,nr_chan//2)
  ax0.set_ylim(ymin, ymax)
  ax0.set_xlabel('$\\Delta$ channel')
  ax0.set_ylabel('$A$')

  ax1.axhline(y=0, color='k', ls=':')
  ax1.plot(spec_z, spec_autocorr_mean, 'k-', ds='steps-mid', label='$\\langle A_F \\rangle $ from {0:d} spectra'.format(nr_spec), lw=3)
  ax1.fill_between(spec_z, spec_autocorr_p16, spec_autocorr_p84, color='k', alpha=0.3, step='mid', label='$16^\\mathrm{th}$ - $84^\\mathrm{th}$ perc.')
  if art_len > 0:
    ax1.plot(spec_z[nr_chan//2+art_len:], art_autocorr[nr_chan//2+art_len:], 'r-', label='$Z$ = artefacts (order {0:d})'.format(art_ord), lw=1)
    ax1.plot(spec_z[:nr_chan//2+art_len+1], art_autocorr[:nr_chan//2+art_len+1], 'r--', lw=1)
    ax1.plot([art_len, art_len], [art_autocorr[nr_chan//2+art_len] - 0.1, art_autocorr[nr_chan//2+art_len] + 0.1], 'r--', lw=1)
  colind = 0
  for kk in knames:
    ax1.plot(spec_z, kern_autocorr[kk], c=kcolors[colind], marker='o', ls='', alpha=0.5, label='$A_K$({0:s})'.format(kk))
    colind += 1
  ax1.legend(fontsize=legend_font_size)
  ax1.set_xlim(0, min(nr_chan//2,3*max_nonzero_autocorr))
  ax1.set_xlabel('$\\Delta$ channel')
  ax1.set_ylabel('$A$')

  ax2.axhline(y=0, color='k', ls=':')
  if art_len > 0:
    lab2_1 = '$+\\sqrt{\\mathcal{F}\\langle A_F\\rangle - Z }$'
    lab2_2 = '$\\Lambda\\left(\\sqrt{\\mathcal{F}\\langle A_F\\rangle - Z }\\right)$'
  else:
    lab2_1 = '$+\\sqrt{\\mathcal{F}\\langle A_F\\rangle }$'
    lab2_2 = '$\\Lambda\\left(\\sqrt{\\mathcal{F}\\langle A_F\\rangle }\\right)$'
  if track_sign or force_sign:
    ax2.plot(spec_z, np.real(np.fft.fftshift(rec_kernel_fft)), 'k-', ds='steps-mid', lw=8, alpha=0.3, label=lab2_1)
    ax2.plot(spec_z, np.real(np.fft.fftshift(rec_kernel_fft_sign)), 'k-', ds='steps-mid', lw=3, alpha=1, label=lab2_2)
  else:
    ax2.plot(spec_z, np.real(np.fft.fftshift(rec_kernel_fft)), 'k-', ds='steps-mid', lw=3, alpha=1, label=lab2_1)
  ax2.legend(fontsize=legend_font_size)
  ax2.set_xlim(0,nr_chan//2)
  ax2.set_xlabel('conjugate channel')
  ax2.set_ylabel('$\\mathcal{F}K$')

  ax3.axhline(y=0, color='k', ls=':')
  if art_len > 0:
    lab3_1 = '$\\mathcal{F}^{-1}\\Lambda\\left(\\sqrt{\\mathcal{F}\\langle A_F\\rangle - Z }\\right)$'
    lab3_2 = '$\\mathcal{F}^{-1}\\sqrt{\\mathcal{F}\\langle A_F\\rangle - Z }$'
  else:
    lab3_1 = '$\\mathcal{F}^{-1}\\Lambda\\left(\\sqrt{\\mathcal{F}\\langle A_F\\rangle }\\right)$'
    lab3_2 = '$\\mathcal{F}^{-1}\\sqrt{\\mathcal{F}\\langle A_F\\rangle }$'
  if track_sign or force_sign:
    ax3.plot(spec_z, rec_kernel, 'k-', ds='steps-mid', alpha=1, lw=3, label=lab3_1)
  else:
    ax3.plot(spec_z, rec_kernel, 'k-', ds='steps-mid', alpha=1, lw=3, label=lab3_2)
  colind = 0
  for kk in knames:
    ax3.plot(spec_z, kernels[kk], c=kcolors[colind], marker='o', ls='', alpha=0.5, label='$K$({0:s})'.format(kk))
    colind += 1
  ax3.legend(fontsize=legend_font_size)
  ax3.set_xlim(0, min(nr_chan//2,3*max_nonzero_autocorr))
  ax3.set_xlabel('channel')
  ax3.set_ylabel('$K$')

  ax4.plot(np.arange(nr_chan//2), rec_area, 'k-', ds='steps-post', alpha=0.3, lw=3)
  ax4.plot(np.arange(nr_chan//2-ii_area,nr_chan//2), rec_area[-ii_area:], 'k-', ds='steps-post', alpha=1, lw=3)
  ax4.plot([nr_chan//2-ii_area, nr_chan//2-ii_area], [0.9 * rec_area_med, 1.1 * rec_area_med], 'k:')
  ax4.axhline(y=rec_area_med, color='k', ls=':', label='$\\int{{ K / K_\\mathrm{{max}}}} \\to$ {0:.2f} channels'.format(rec_area_med))
  ax4.legend(fontsize=legend_font_size)
  ax4.set_xlim(0,nr_chan//2)
  ax4.set_ylim(0,1.1*rec_area.max())
  ax4.set_xlabel('channel')
  ax4.set_ylabel('cumul. $\\int{K / K_\\mathrm{max}}$')

  plt.tight_layout()
  if output:
    plt.savefig(output)
  else:
    plt.show()

# Run the program is called from command line
if __name__ == "__main__":  
  main()
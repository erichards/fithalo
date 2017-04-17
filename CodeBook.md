# Code Book for fithalo.py v.1.1
4/17/2017

The distribution of mass in galaxies can be constrained through decomposition
of the observed rotation curve, which traces the total gravitational potential
of the galaxy. Following standard physics for objects in circular orbit, the
observed rotation curve can be decomposed as:

(V_obs)^2 = (V_gas)^2 + (M/L)(V_star)^2 + (V_halo)^2,

where V_obs is the observed rotation curve, V_gas is the model
rotation curve of the gas distribution, V_star is the stellar
component model rotation curve, M/L is the stellar mass-to-light ratio, and
V_halo is the estimated dark matter halo model rotation curve.

This script takes observed and model baryon (gas + stars) component circular
rotational velocities as a function of radius (rotation curves) for an
individual galaxy as input and fits a pseudo-isothermal spherical dark matter
halo model rotation curve to the residuals between the observed and summed
baryon model components using non-linear least-squares fitting. The
pseudo-isothermal spherical dark matter halo model circular rotational
velocities as a function of radius are described by the functional form:

```python
def halo(r, *p):
        V_H, R_C = p
	return V_H*np.sqrt(1-((R_C/r)*(np.arctan(r/R_C))))
```

where V_H and R_C are free parameters to be solved for during
the fitting routine. Mass-to-light ratio (M/L) scalings of the model stellar
components (disk and/or bulge) are 
also derived initially through non-linear least-squares fitting along with 
the dark matter halo component. These fitting results are displayed to the 
user upon which he or she is given the option of manually specifying the 
disk and/or bulge M/L. There is an additional option for the user to define 
the M/L(s) and the dark matter halo model completely through interaction 
with the plot.

## Input File

fithalo.py expects an input text file with the following data and format:

Galaxy Name  
disk magnitude bulge magnitude  
D (Mpc)  
R_(V_HI) (kpc)  
h_R (arcsec) D_25 (arcsec)

RAD | VROT | V_ERR | V_GAS | V_DISK | V_BULGE
--- | ---- | ----- | ----- | ------ | -------
0.00 | 0.00 | 0.00 | 0.000000 | 0.000000 | 0.000000
...

See example file 'galaxy.rot'.

## Variables

### Metadata

- Galaxy Name: whatever object/catalog name you prefer; used for plot title
and output text file

- disk & bulge magnitude: measured apparent magnitudes at 3.6 micron; used in
luminosity calculations for proper scaling of the rotation curves

- D (Mpc): distance; used in various calculations and to convert between
angular and linear scales (i.e. arcsec to kpc)

- R_(V_HI) (kpc): first radial point in the observed rotation curve
where the circular velocities have been derived from neutral gas (HI) rather
than ionized gas; used for changing plot symbols to indicate source of measured
circular rotational velocities

- h_R (arcsec): disk exponential scale length which describes slope
of the decrease in the surfrace brightness profiles of stellar disks modeled as
exponentials; used to mark 2.2*h_R on the plot, which is the radius
where a point source orbiting in a self-gravitating infinitely thin disk
would experience maximum circular rotational velocity

- D_25 (arcsec): diameter of the stellar disk measured at the point where
the surface brightness in the B-band (blue) decreases to 25 mag/arcsec^2;
used to mark this radius on the plot

### Vectors

- RAD (kpc): radial sampling for rotation curves, starting at 0 kpc

- VROT (km/s): observed circular rotational velocity at each radius,
typically measured either from a 2D map of the velocity field, or using the
full 3D data cube

- V_ERR (km/s): error on the observed circular rotational velocities; main
sources of uncertainty include resolution effects, non-circular motions of
the gas, and asymmetry between the approaching and receding sides of the
rotating gas disk

- V_GAS (km/s): modeled contribution of the gas disk to the total observed
circular rotational velocities; determined from the HI mass surface density
radial distribution and converted into circular rotational velocities
assuming an infintely thin disk potential

- V_DISK (km/s): modeled contribution of the stellar disk to the total
observed circular rotational velocities; determined from the 3.6 mircon surface
brightness profile and converted into circular rotational velocities using
the gravitational potential of a disk with some vertical thickness and a mass
of 10^9 Msun

- V_BULGE (km/s): modeled contribution of the stellar bulge (if present) to
the total observed circular rotational velocities; determined from the
3.6 micron surface brightness profile decomposed into an r^(1/4)
de Vaucouleur's bulge component and exponential disk component and translated
into circular rotational velocities assuming a spherical potential and mass
of 10^9 Msun

## Running the Code

1. Start by typing the following while in the directory containing fithalo.py:

   `$ python fithalo.py path_to_files/filename`

2. A plot will appear showing the initial fit results for the dark matter
halo model rotation curve and best-fitting stellar disk and/or bulge M/L in
the range 0.3-0.8. The fit parameters are additionally printed to the terminal
window. See Output for an explanation of the various rotation curves.

3. From here, the user can choose an option from the list printed to the
terminal window:

#### Menu Options

Plot options | (see Interacting with the Plot)
------------ | -------------------------------
 left-mouse-click | adjust disk M/L
 middle-mouse-click | adjust bulge M/L
 shift+left-mouse-click | also adjust bulge M/L
 right-mouse-click | adjust halo fit parameters $R_{\rm C}$ & $V_{\rm H}$
 e | toggle +/- 0.1 M/L error bands on/off
 h | re-fit halo using current M/L
 m | mouse-click scaling enabled (default, interactive mode)
 n | no mouse-click scaling (activate label mode)
 c | center selected label
 d | shift selected label down
 l | shift selected label to the left
 r | shift selected label to the right
 u | shift selected label up

Command line options | (see Fitting Details and Output)
-------------------- | --------------------------------
 f |                provide fixed M/L and re-fit halo
 r |                re-fit halo & M/L using non-linear least-squares
 p |                save figure to file
 s |                write text file
 q |                quit

### Interacting with the Plot

The user may interact with the plot in two modes: interactive and label. The
default interactive mode is used for manipulation of the data and fit, whereas
label mode is used for the finishing touches to create a publication-quality
plot. 

#### Interactive Mode

The user has the option of using the mouse while the plot window is activated
to manually alter the free parameters in the fit:

- disk M/L             --> left-mouse-click
- bulge M/L            --> middle-mouse-click (or shift + left-mouse-click)
- halo model R_C & V_H --> right-mouse-click

The stellar rotation curve components are scaled by setting the circular
rotational velocity at the radius closest where the mouse click occurred to
the y-axis value at the location of the mouse click. The dark matter halo
model parameters R_C and V_H are set to the x- and y-axis
values, respectively, at the location of the right-mouse-click.

The user may at any time press the 'h' key on the keyboard to re-fit for the
dark matter halo model parameters using the currently specified M/L(s), as
long as the following conditions are satisfied: 0 <= M/L <= 10 and the
residual vector formed by subtracting the total baryon contribution from the
observed rotation curve contains more than 2 points.

Pressing the 'e' key on the keyboard will toggle error bands on and off. The
error bands are determined by adding and subtracting 0.1 to the current M/L(s)
and re-computing the stellar disk and/or bulge and total baryon rotation
curves followed by re-fitting for the dark matter halo parameters. The error
bands will only be computed in the range 0.3 <= M/L <= 0.8. Only the
lower or upper error bands will be displayed if the M/L(s) = 0.8 or 0.3,
respectively. A known issue with the toggle feature on some systems is having
to click on the plot before the toggle off is registered. Therefore, it is
recommended that the errorbands are only toggled on once the fit and plot are
finalized.

#### Label Mode

Label mode functionality was created to quickly and easily shift the component
rotation curve and radii labels to prevent crowded or overlapping text.
Pressing the 'n' key disconnects the mouse-click events so that the user may
interact with the text labels without triggering scaling of the disk, bulge,
or halo model rotation curves. To move a text label position, simply click on
it then use the arrow keys to shift it up, down, right, or left. Alternatively,
the 'u', 'd', 'r', and 'l' keys may be used to shift the text up, down, right,
or left, respectively. Please note that the text is shifted by changing its
vertical and horizontal alignment only, so it cannot be shifted by an
arbitrary amount. Press 'm' to return to the interactive mode and re-enable
mouse-click scaling.

### Fitting Details

The stellar M/L(s) may be included as free parameters to be fit, or fixed to
user specified values. Both fitting options are described in detail below.

#### Free M/L(s)

Upon starting the program, a non-linear least-squares fit is performed on the
data to minimize the function:

```python
def total(X, *p):
        V_H, R_C, dML, bML = p
        r, vgas, vdisk, vbulge = X
        return np.sqrt((halo(r, V_H, R_C)**2.) + (vgas**2.)
                       + (dML * (vdisk**2.)) + (bML * (vbulge**2.)))
```

where the disk and bulge M/L (dML and bML, respectively) and dark matter halo
parameters V_H and R_C are free parameters. The `curve_fit` function from the
scipy optimize package is used for the non-linear least-squares fitting. To
avoid non-real and non-physical values in the fitting, the dark matter halo
free parameters are constrained to the range 0-1000, and the M/L(s) are
forced to stay between 0.3 and 0.8. For bound problems such as this,
`curve_fit` employs the Trust Region Reflective algorithm for optimization.
The fit is also weighted by the errors on the observed rotation. This
original fit with the M/L(s) are free parameters may be returned to at any
time by entering 'r' on the terminal window command line.

#### Fixed M/L(s)

After the initial fit, the user may specify the disk and/or bulge M/L, either
through interaction with the plot (see Interacting with the Plot) or by typing
in the M/L(s) at the prompt after entering 'f' on the command line. In both
cases, the specified M/L(s) will be used to calculate the stellar disk and/or
bulge circular rotational velocities and sum them in quadrature with the gas
rotational velocities to compute a total baryon contribution to the observed
rotation curve. The baryon contribution is then subtracted (in quadrature)
from the total observed circular rotational velocities to create a residual
rotation curve, which is then fit with the dark matter halo model from the
function above described by the free parameters V_H and R_C.
The `curve_fit` function is once again used for the non-linear least-squares
fitting with bounds, and is weighted by the observed uncertainties.

### Output

Results of the current decomposition may be saved at any time as a figure by
entering 'p' at the command line and a text file by entering 's' containing
the fit parameters and component rotation curves (see the example output
files contained in this repository).

![](https://github.com/erichards/fithalo/blob/master/galaxy.png)

To exit the program, type 'q' at the command prompt.

## Future Improvements

There are a number of developments that would greatly improve the flexibility
and functionality of this program, most notably of which is less stringent
constraints on the input file format. Short-term goals include allowing the
user to choose whether or not to use error weighting in the fits.
Long-term goal is to overhaul the UI so that much of the metadata
and user options can be specified through input options, allowing the program
to be run interactively and as a non-interactive script for batch mode
processing. It would also be nice to remove the assumed mass scaling for the
stellar components to enable this own program's output data to be re-fit.
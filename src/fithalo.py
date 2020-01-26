"""
fithalo.py v.1.1
ER 4/17/2017
________________________________________________________________________

This script takes observed and model baryon component circular
rotational velocities as a function of radius (rotation curves) for an
individual galaxy as input and fits a pseudo-isothermal spherical dark
matter halo model rotation curve to the residuals between the observed
and summed baryon model components using non-linear least-squares
fitting. Mass-to-light ratio (M/L) scalings* of the model stellar
components (disk and/or bulge) are also derived initially through
non-linear least-squares fitting along with the dark matter halo
component. These fitting results are displayed to the user upon which
he or she is given the option of manually specifying the disk and/or
bulge M/L. There is an additional option for the user to define the
M/L(s) and the dark matter halo model completely through interaction
with the plot.

* It is assumed that the provided model stellar component rotation
curves have been normalized to masses of 10^9 Msun and M/L = 1.
"""

# =========================================================================== #


import sys
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from draw_plot import draw_plot
from halo import halo, do_fit, do_fixed_ml_fit
from interactive_plot import InteractivePlot
from menus import print_menu, print_fit_results

logger = logging.getLogger('fithalo.src.fithalo')

ABSMAG_SUN = 3.24  # at 3.6 microns, used in luminosity calculations


def ml_prompt(v_disk, v_bulge):
    # ran into issues with floating point precision
    x = np.linspace(0.0, 10.0, 101)
    accept_ml = [float('{0:.1f}'.format(num)) for num in x]
    if not all(v_disk == 0.):
        d_ml = float(input('Enter disk M/L: '))
        while d_ml not in accept_ml:
            print('Please enter a disk M/L between 0 and 10 rounded to the nearest tenth place.')
            d_ml = float(input('Enter disk M/L: '))
    else:
        d_ml = 0.
    if not all(v_bulge == 0.):
        b_ml = float(input('Enter bulge M/L: '))
        while b_ml not in accept_ml:
            print('Please enter a bulge M/L between 0 and 10 rounded to the nearest tenth place.')
            b_ml = float(input('Enter bulge M/L: '))
    else:
        b_ml = 0.

    return d_ml, b_ml


def read_file(filename):
    with open(filename, 'r') as f:
        params = {}
        params['galaxy_name'] = f.readline().strip()
        magnitudes = f.readline().split()
        params['disk_mag'] = float(magnitudes[0])
        params['bulge_mag'] = float(magnitudes[1])
        params['d_mpc'] = float(f.readline().strip())
        params['v_hi_rad'] = float(f.readline().strip())  # new addition to files
        radii = f.readline().split()
        params['h_r'] = float(radii[0])  # new addition to files
        params['d25'] = float(radii[1])  # new addition to files
    rcdf = pd.read_table(filename, skiprows=5, delim_whitespace=True)
    return params, rcdf


def magnitude_to_luminosity(magnitude, d_mpc):
    return 10**(0.4 * (ABSMAG_SUN - magnitude + 5 * np.log10((d_mpc * 10**6.) / 10.)))


def calculate_rotation_curves(rcdf, halo_fit_params):
    # calculate velocities based on fit parameters
    if 'd_ml' in halo_fit_params:
        rcdf['V_disk_fit'] = rcdf.V_disk_norm * np.sqrt(halo_fit_params['d_ml'])
        rcdf['V_bulge_fit'] = rcdf.V_bulge_norm * np.sqrt(halo_fit_params['b_ml'])
    rcdf['V_bary_fit'] = np.sqrt((rcdf.V_gas_scaled ** 2.) + (rcdf.V_disk_fit ** 2.) + (rcdf.V_bulge_fit ** 2.))
    rcdf['V_halo_fit'] = halo(rcdf.RAD, halo_fit_params['v_h'], halo_fit_params['r_c'])
    rcdf.loc[0, 'V_halo_fit'] = 0  # replace first row NaN with 0
    rcdf['V_tot_fit'] = np.sqrt((rcdf.V_bary_fit ** 2.) + (rcdf.V_halo_fit ** 2.))
    return rcdf


def initialize_data(rcdf, params):
    rcdf['V_gas_scaled'] = rcdf.V_GAS * np.sqrt(1.4)  # scale by 1.4 for He
    ldisk = magnitude_to_luminosity(params['disk_mag'], params['d_mpc'])
    lbulge = magnitude_to_luminosity(params['bulge_mag'], params['d_mpc'])
    # calculate disk & bulge rotation curves assuming M/L = 1
    rcdf['V_disk_norm'] = rcdf.V_DISK * np.sqrt(ldisk / 10**9.)
    rcdf['V_bulge_norm'] = rcdf.V_BULGE * np.sqrt(lbulge / 10**9.)

    # initial fit for halo & M/L's
    v = [rcdf.VROT, rcdf.V_ERR, rcdf.V_gas_scaled, rcdf.V_disk_norm, rcdf.V_bulge_norm]
    halo_fit_params = do_fit(rcdf.RAD, v)

    rcdf = calculate_rotation_curves(rcdf, halo_fit_params)

    return rcdf, halo_fit_params


def write_results(ip, halo_fit_params, galaxy_name):
    if hasattr(ip, 'v_h'):
        v_h, v_h_err, r_c, r_c_err = ip.get_current_halo()
        chi, redchi = ip.get_current_chi()
    else:
        v_h, v_h_err = halo_fit_params['v_h'], halo_fit_params['v_h_err']
        r_c, r_c_err = halo_fit_params['r_c'], halo_fit_params['r_c_err']
        chi, redchi = halo_fit_params['chi_sq'], halo_fit_params['red_chi_sq']
    d_ml, b_ml = ip.get_current_ml()
    wdf = ip.write_df()
    # TODO: column names in df are different
    # cols = ['Rad', 'V_Rot', 'V_err', 'V_gas', 'V_disk', 'V_bulge', 'V_bary', 'V_halo', 'V_tot']
    txtfile = input('Enter text file name: ')
    with open(txtfile, 'w') as wf:
        wf.write(galaxy_name + '\n')
        wf.write(f'Disk M/L = {d_ml:.1f}, Bulge M/L = {b_ml:.1f}\n')
        wf.write(f'R_C = {r_c:.1f} +/- {r_c_err:.2f} kpc\n')
        wf.write(f'V_H = {v_h:.0f} +/- {v_h_err:.0f} km/s\n')
        wf.write(f'chi squared = {chi:.3f}\n')
        wf.write(f'reduced chi squared = {redchi:.3f}\n')
        # TODO: look into better way of writing out results
        wdf.to_csv(wf, sep='\t', float_format='%.2f', index=False)


def handle_choice(choice, ip, rcdf, galaxy_params, halo_fit_params):
    while choice not in ['f', 'r', 'p', 's', 'q']:
        print(f"'{choice}' is not a valid entry. Please try again.")
        choice = input()
    if choice == 'q':
        sys.exit(0)
    while choice != 'q':
        if choice == 'f': # user input M/L
            ip.disconnect()
            plt.close()
            # calculate V_disk & V_bulge based on user-input M/L
            d_ml, b_ml = ml_prompt(rcdf.V_disk_norm, rcdf.V_bulge_norm)
            rcdf.V_disk_fit = rcdf.V_disk_norm * np.sqrt(d_ml)
            rcdf.V_bulge_fit = rcdf.V_bulge_norm * np.sqrt(b_ml)
            # fit halo with fixed M/L
            vels = [rcdf.VROT, rcdf.V_ERR, rcdf.V_gas_scaled, rcdf.V_disk_fit, rcdf.V_bulge_fit]
            halo_fit_params = do_fixed_ml_fit(rcdf.RAD, vels)
            halo_fit_params['d_ml'] = d_ml
            halo_fit_params['b_ml'] = b_ml
            print_fit_results(halo_fit_params)
            # update rotation curves (minus stellar ones which were calculated above)
            rcdf = calculate_rotation_curves(rcdf, halo_fit_params)
            # update plot
            fig, rc_plots = draw_plot(rcdf, galaxy_params, halo_fit_params)
            ip = InteractivePlot(fig, rc_plots, rcdf, halo_fit_params)
            plt.show(block=False)
            choice = input()
        elif choice == 'r': # fit halo & M/L
            ip.disconnect()
            plt.close()
            # fit halo & M/L
            vels = [rcdf.VROT, rcdf.V_ERR, rcdf.V_gas_scaled, rcdf.V_disk_norm, rcdf.V_bulge_norm]
            halo_fit_params = do_fit(rcdf.RAD, vels)
            print_fit_results(halo_fit_params)
            # update rotation curves
            rcdf = calculate_rotation_curves(rcdf, halo_fit_params)
            # update plot
            fig, rc_plots = draw_plot(rcdf, galaxy_params, halo_fit_params)
            ip = InteractivePlot(fig, rc_plots, rcdf, halo_fit_params)
            plt.show(block=False)
            choice = input()
        elif choice == 'p':
            figfile = input('Enter figure file name: ')
            plt.savefig(figfile)
            choice = input()
        elif choice == 's': # save results to file
            write_results(ip, halo_fit_params, galaxy_params['galaxy_name'])
            choice = input()
        else:
            return


def main():
    # start by reading in the file
    try:
        filename = sys.argv[1]
    except IndexError:
        logger.error('Please provide a rotation curve file. Usage: "$ python fithalo.py path/filename.rot"')
        return

    if not os.path.exists(filename):
        logger.error(f'File {filename} does not exist.')
    else:
        # read data file
        galaxy_params, rcdf = read_file(filename)
        # do an initial fit with free M/L & record resulting RCs in data frame
        rcdf, halo_fit_params = initialize_data(rcdf, galaxy_params)
        # plot initial fit
        fig, rc_plots = draw_plot(rcdf, galaxy_params, halo_fit_params)
        ip = InteractivePlot(fig, rc_plots, rcdf, halo_fit_params)
        plt.show(block=False)
        print_menu(galaxy_params['galaxy_name'])
        print_fit_results(halo_fit_params)
        choice = input()
        handle_choice(choice, ip, rcdf, galaxy_params, halo_fit_params)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main()

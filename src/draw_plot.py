import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams
from halo import halo

plt.rc('font', family='serif')
rcParams['axes.labelsize'] = 38
rcParams['xtick.labelsize'] = 38
rcParams['ytick.labelsize'] = 38
rcParams['legend.fontsize'] = 32
rcParams['axes.titlesize'] = 42
# disable some annoying default keymaps
rcParams['keymap.yscale'] = ''
rcParams['keymap.fullscreen'] = ''
rcParams['keymap.back'] = ''


def initialize_plot():
    fig = plt.figure(figsize=(25, 15), dpi=30)
    ax1 = fig.add_subplot(111)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.tick_params(which='major', width=3, length=20)
    ax1.tick_params(which='minor', width=3, length=10)
    ax1.set_xlabel('Radius (arcsec)')
    ax1.set_ylabel('Velocity (km s$^{-1}$)')
    ax2 = ax1.twiny()
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.tick_params(which='major', width=3, length=20)
    ax2.tick_params(which='minor', width=3, length=10)
    ax2.set_xlabel('Radius (kpc)')
    return fig, ax1, ax2


def annotate(text, x_loc, arrow_height):
    return plt.annotate(
        text, xy=(x_loc, 0.), xycoords='data', xytext=(x_loc, arrow_height), textcoords='data', 
        arrowprops=dict(arrowstyle='simple', fc='k'), fontsize=38, ha='center', va='center', picker=True)


def draw_observed(df, v_hi_rad):
    first_hi = df.index[df['RAD'] == v_hi_rad].tolist()[0]
    for i in range(0, first_hi):
        plt.plot(df.RAD[i], df.VROT[i], 'ok', ms=20, ls='None', fillstyle='none', mew=5)
    for j in range(first_hi, len(df.RAD)):
        plt.plot(df.RAD[j], df.VROT[j], 'ok', ms=20, ls='None')
    plt.errorbar(df.RAD, df.VROT, yerr=df.V_ERR, color='k', lw=5, ls='None')
    return


def draw_gas(df, x_label_pos, last):
    plt.plot(df.RAD, df.V_gas_scaled, 'g-', ls='--', lw=5, dashes=(20, 20))
    gas_label = plt.text(x_label_pos, df.V_gas_scaled[last], 'Gas', fontsize=38, ha='center', va='center', picker=True)
    return gas_label


def draw_stellar_disk(df, x_label_pos, last, d_ml):
    if all(df.V_disk_fit == 0.):
        d_ml_txt = plt.figtext(0., 0., '')
        disk_rc = plt.plot(df.RAD, df.V_disk_fit, '', ls='')
        disk_label = plt.text(0., 0., '')
    else:
        d_ml_txt = plt.figtext(0.4, 0.85, 'disk M/L = %.1f' % d_ml, fontsize=38)
        disk_rc = plt.plot(df.RAD, df.V_disk_fit, 'm-', ls=':', lw=5, dashes=(5, 15))
        disk_label = plt.text(
            x_label_pos, df.V_disk_fit[last], 'Disk', fontsize=38, ha='center', va='center', picker=True)
    return disk_rc, disk_label, d_ml_txt


def draw_stellar_bulge(df, x_label_pos, last, b_ml):
    if all(df.V_bulge_fit == 0.):
        b_ml_txt = plt.figtext(0., 0., '')
        bulge_rc = plt.plot(df.RAD, df.V_bulge_fit, '', ls='')
        bulge_label = plt.text(0., 0., '')
    else:
        b_ml_txt = plt.figtext(0.4, 0.85, 'bulge M/L = %.1f' % b_ml, fontsize=38)
        bulge_rc = plt.plot(df.RAD, df.V_bulge_fit, 'c-', ls='-.', lw=5, dashes=[20, 20, 5, 20])
        bulge_label = plt.text(
            x_label_pos, df.V_bulge_fit[last], 'Bulge', fontsize=38, ha='center', va='center', picker=True)
    return bulge_rc, bulge_label, b_ml_txt


def draw_baryons(df, x_label_pos, last):
    bary_rc = plt.plot(df.RAD, df.V_bary_fit, 'b-', ls='--', lw=5, dashes=[20, 10, 20, 10, 5, 10])
    bary_label = plt.text(
        x_label_pos, df.V_bary_fit[last], 'Bary', fontsize=38, ha='center', va='center', picker=True)
    return bary_rc, bary_label


def draw_halo(df, v_h, r_c, x_label_pos, last):
    r_high_res = np.linspace(df.RAD.min(), df.RAD.max())
    halo_rc = plt.plot(r_high_res, halo(r_high_res, v_h, r_c), 'r-', lw=5)
    halo_label = plt.text(
        x_label_pos, df.V_halo_fit[last], 'Halo', fontsize=38, ha='center', va='center', picker=True)
    return halo_rc, halo_label


def draw_total(df, x_label_pos, last):
    total_rc = plt.plot(df.RAD, df.V_tot_fit, 'k-', ls='-', lw=5)
    total_label = plt.text(
        x_label_pos, df.V_tot_fit[last], 'Total', fontsize=38, ha='center', va='center', picker=True)
    return total_rc, total_label


def plot_rotation_curves(df, v_hi_rad, halo_fit_params):
    # observed RC
    draw_observed(df, v_hi_rad)

    last = len(df.RAD) - 1
    x_label_pos = df.RAD.max() * 1.08

    # model gas RC
    gas_label = draw_gas(df, x_label_pos, last)
    # model stellar disk & bulge RC
    disk_rc, disk_label, d_ml_txt = draw_stellar_disk(df, x_label_pos, last, halo_fit_params['d_ml'])
    bulge_rc, bulge_label, b_ml_txt = draw_stellar_bulge(df, x_label_pos, last, halo_fit_params['b_ml'])
    # model total baryon RC
    bary_rc, bary_label = draw_baryons(df, x_label_pos, last)
    # model DM halo RC
    halo_rc, halo_label = draw_halo(df, halo_fit_params['v_h'], halo_fit_params['r_c'], x_label_pos, last)
    # best fitting total
    total_rc, total_label = draw_total(df, x_label_pos, last)

    rc_plots = [disk_rc[0], bulge_rc[0], bary_rc[0], halo_rc[0], total_rc[0]]
    labels = [gas_label, disk_label, bulge_label, bary_label, halo_label, total_label]
    return rc_plots, labels


def draw_plot(df, galaxy_params, halo_fit_params):
    # calculate radii in arcseconds for alt axis
    arcrad = (df.RAD / (galaxy_params['d_mpc'] * 1000.)) * 206265.
    # setup the plot text & axes
    fig, ax1, ax2 = initialize_plot()
    ax1.set_xlim(0, arcrad.max() * 1.15)
    ax1.set_ylim(0, df.VROT.max() * 1.6)
    ax2.set_xlim(0, df.RAD.max() * 1.15)
    ax2.set_ylim(0, df.VROT.max() * 1.6)
    plt.figtext(0.1, 1, galaxy_params['galaxy_name'], fontsize=48, va='top')  # galaxy title
    plt.figtext(0.15, 0.85, 'D = {0:.1f} Mpc'.format(galaxy_params['d_mpc']), fontsize=38)
    # plt.figtext(0.15, 0.8, 'max. disk', fontsize=38)
    r_c_txt = plt.figtext(0.65, 0.85, 'R$_\mathrm{C}$ = %.1f $\pm$ %.1f kpc'
                          % (halo_fit_params['r_c'], halo_fit_params['r_c_err']), fontsize=38)
    v_h_txt = plt.figtext(0.65, 0.8, 'V$_\mathrm{H}$ = %0.f $\pm$ %0.f km s$^{-1}$'
                          % (halo_fit_params['v_h'], halo_fit_params['v_h_err']), fontsize=38)

    # mark radii
    arrow_height = (df.VROT.max() * 1.4) * 0.095
    rbary = df.RAD[df['V_bary_fit'].idxmax()]
    rb_txt = annotate('R$_\mathrm{bary}$', rbary, arrow_height)
    rdyn = 2.2 * ((galaxy_params['h_r'] / 206.265) * galaxy_params['d_mpc'])
    rh_txt = annotate('2.2h$_\mathrm{R}$', rdyn, arrow_height)
    r25 = ((galaxy_params['d25'] / 2.) / 206.265) * galaxy_params['d_mpc']
    if r25 < df.RAD.max() * 1.15:
        r25_txt = annotate('R$_{25}$', r25, arrow_height)
    else:
        r25_txt = plt.annotate('', xy=(0, 0))

    # plot the observed rotation curve using radii in arcseconds
    ax1.plot(arcrad, df.VROT, 'ok', marker='None', ls='None')
    # plot the rest of the rotation curves
    rc_plots, labels = plot_rotation_curves(df, galaxy_params['v_hi_rad'], halo_fit_params)
    labels.extend([r_c_txt, v_h_txt, rb_txt, rh_txt, r25_txt])

    return fig, rc_plots, labels

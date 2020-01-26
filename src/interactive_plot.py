import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text
import logging

from menus import print_fit_results
from halo import halo, do_fixed_ml_fit, fit_dm_halo_fixed_ml

logger = logging.getLogger('fithalo.src.interactive_plot')


class InteractivePlot(object):
    def __init__(self, fig, rc_plots, df, halo_fit_params):
        self.fig = fig
        self.rc_plots = rc_plots
        self.r = df.RAD
        self.vels = {
            'obs': df.VROT, 'obs_err': df.V_ERR, 'gas': df.V_gas_scaled, 'disk': df.V_disk_fit,
            'bulge': df.V_bulge_fit, 'bary': df.V_bary_fit, 'halo': df.V_halo_fit, 'total': df.V_tot_fit}
        self.d_ml = halo_fit_params['d_ml']
        self.b_ml = halo_fit_params['b_ml']
        self.r_c = halo_fit_params['r_c']
        self.r_c_err = halo_fit_params['r_c_err']
        self.v_h = halo_fit_params['v_h']
        self.v_h_err = halo_fit_params['v_h_err']
        self.chi_sq = halo_fit_params['chi_sq']
        self.red_chi_sq = halo_fit_params['red_chi_sq']
        self.disk_bands = None
        self.bulge_bands = None
        self.bary_bands = None
        self.halo_bands = None
        self.total_bands = None
        self.toggle_on = False
        self.shift_is_held = False
        self.picked = None
        # connect event keys
        self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cidpick = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidkey = self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.cidkey_release = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidclick)
        self.fig.canvas.mpl_disconnect(self.cidpick)
        self.fig.canvas.mpl_disconnect(self.cidkey)
        self.fig.canvas.mpl_disconnect(self.cidkey_release)

    def update_rc_label(self, component):
        last = len(self.r) - 1
        x_label_pos = self.r.max() * 1.08
        self.rc_plots[component]['label'].remove()
        self.rc_plots[component]['label'] = plt.text(
            x_label_pos, self.vels[component][last], component, fontsize=38, ha='center', va='center', picker=True)

    def update_rc_plot(self, component, r=None):
        if r is None:
            r = self.r
        self.rc_plots[component]['plot'].set_data(r, self.vels[component])
        self.update_rc_label(component)
        self.rc_plots[component]['plot'].figure.canvas.draw()

    def change_bary(self):
        self.vels['bary'] = np.sqrt((self.vels['gas'] ** 2.) + (self.vels['disk'] ** 2.) + (self.vels['bulge'] ** 2.))
        self.update_rc_plot('bary')

    def change_total(self):
        self.vels['total'] = np.sqrt((self.vels['bary'] ** 2.) + (self.vels['halo'] ** 2.))
        self.update_rc_plot('total')

    def change_disk(self):
        self.rc_plots['disk']['text']['d_ml'].remove()
        self.rc_plots['disk']['text']['d_ml'] = plt.figtext(0.4, 0.85, 'disk M/L = %.1f' % self.d_ml, fontsize=38)
        self.update_rc_plot('disk')
        self.fig.canvas.draw()

    def change_bulge(self):
        self.rc_plots['bulge']['text']['b_ml'].remove()
        if all(self.vels['disk'] == 0.):
            self.rc_plots['bulge']['text']['b_ml'] = plt.figtext(0.4, 0.85, 'bulge M/L = %.1f' % self.b_ml, fontsize=38)
        else:
            self.rc_plots['bulge']['text']['b_ml'] = plt.figtext(0.4, 0.8, 'bulge M/L = %.1f' % self.b_ml, fontsize=38)
        self.update_rc_plot('bulge')
        self.fig.canvas.draw()

    def change_halo(self):
        self.rc_plots['halo']['text']['r_c'].remove()
        self.rc_plots['halo']['text']['v_h'].remove()
        self.rc_plots['halo']['text']['r_c'] = plt.figtext(
            0.65, 0.85, 'R$_\mathrm{C}$ = %.1f kpc' % self.r_c, fontsize=38)
        self.rc_plots['halo']['text']['v_h'] = plt.figtext(
            0.65, 0.8, 'V$_\mathrm{H}$ = %0.f km s$^{-1}$' % self.v_h, fontsize=38)
        r_high_res = np.linspace(self.r.min(), self.r.max())
        self.vels['halo'] = halo(r_high_res, self.v_h, self.r_c)
        self.vels['halo'][0] = 0  # replace first row NaN with 0
        self.update_rc_plot('halo', r=r_high_res)
        self.fig.canvas.draw()
        # reset halo rotation curve to use original radii
        self.vels['halo'] = halo(self.r, self.v_h, self.r_c)
        self.vels['halo'][0] = 0  # replace first row NaN with 0

    def fit_halo(self):
        vels = [self.vels['obs'], self.vels['obs_err'], self.vels['gas'], self.vels['disk'], self.vels['bulge']]
        halo_fit_params = do_fixed_ml_fit(self.r, vels)
        if halo_fit_params is not None:
            self.r_c = halo_fit_params['r_c']
            self.r_c_err = halo_fit_params['r_c_err']
            self.v_h = halo_fit_params['v_h']
            self.v_h_err = halo_fit_params['v_h_err']
            self.chi_sq = halo_fit_params['chi_sq']
            self.red_chi_sq = halo_fit_params['red_chi_sq']
            self.change_halo()
            halo_fit_params['d_ml'] = self.d_ml
            halo_fit_params['b_ml'] = self.b_ml
            print_fit_results(halo_fit_params)
        else:
            self.rc_plots['halo']['text']['r_c'] = plt.figtext(0, 0, '')
            self.rc_plots['halo']['text']['v_h'] = plt.figtext(0, 0, '')
            self.rc_plots['halo']['label'] = plt.text(0, 0, '')
            return

    def get_current_halo(self):
        if hasattr(self, 'v_h_err'):
            return self.v_h, self.v_h_err, self.r_c, self.r_c_err
        else:
            return self.v_h, 10.0, self.r_c, 1.0

    def get_current_ml(self):
        return self.d_ml, self.b_ml

    def get_current_chi(self):
        if hasattr(self, 'chi_sq'):
            return self.chi_sq, self.red_chi_sq
        else:
            return 0.0, 0.0

    def write_df(self):
        d = {'Rad': self.r}
        d.update(self.vels)
        df = pd.DataFrame(d)
        print(df.head())
        return df

    def on_press(self, event):
        if event.key == 'shift':
            self.shift_is_held = True
        if 0. >= self.d_ml >= 10. or 0. >= self.b_ml >= 10.:
            logger.warning('0 <= M/L <= 10 necessary for halo fitting!')
            return
        if event.key == 'e':
            # get upper/lower limits from +/- 0.1 range in M/L
            vdisk_range, vbulge_range, vbary_range, vhalo_range, vtot_range = ml_range(
                self.r, self.vels, self.d_ml, self.b_ml)
            if self.toggle_on is False:
                self.toggle_on = True
                self.disk_bands = plt.fill_between(self.r, vdisk_range[0], vdisk_range[1], color='m', alpha=0.3)
                self.bulge_bands = plt.fill_between(self.r, vbulge_range[0], vbulge_range[1], color='c', alpha=0.3)
                self.bary_bands = plt.fill_between(self.r, vbary_range[0], vbary_range[1], color='b', alpha=0.3)
                self.halo_bands = plt.fill_between(self.r, vhalo_range[0], vhalo_range[1], color='r', alpha=0.3)
                self.total_bands = plt.fill_between(self.r, vtot_range[0], vtot_range[1], color='k', alpha=0.3)
                self.fig.canvas.draw()
            else:
                self.disk_bands.remove()
                self.bulge_bands.remove()
                self.bary_bands.remove()
                self.halo_bands.remove()
                self.total_bands.remove()
                self.toggle_on = False
        elif event.key == 'h':
            self.fit_halo()
            self.change_total()
        elif event.key == 'n':
            self.fig.canvas.mpl_disconnect(self.cidclick)
            logger.info('Now entering label mode. Mouse-click scaling disabled.')
        elif event.key == 'm':
            self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            logger.info('Returning to interactive mode. Mouse-click scaling enabled.')
        elif event.key == 'u' or event.key == 'up':
            if self.picked is not None:
                self.picked.set_va('bottom')
                self.fig.canvas.draw()
            else:
                return
        elif event.key == 'd' or event.key == 'down':
            if self.picked is not None:
                self.picked.set_va('top')
                self.fig.canvas.draw()
            else:
                return
        elif event.key == 'c':
            if self.picked is not None:
                self.picked.set_va('center')
                self.picked.set_ha('center')
                self.fig.canvas.draw()
            else:
                return
        elif event.key == 'r' or event.key == 'right':
            if self.picked is not None:
                self.picked.set_ha('left')
                self.fig.canvas.draw()
            else:
                return
        elif event.key == 'l' or event.key == 'left':
            if self.picked is not None:
                self.picked.set_ha('right')
                self.fig.canvas.draw()
            else:
                return
        else:
            return

    def on_pick(self, event):
        if isinstance(event.artist, Text):
            self.picked = event.artist
            logger.info(f'{self.picked.get_text()} label selected')

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def on_click(self, event):
        if event.inaxes != self.rc_plots['disk']['plot'].axes:
            return
        idx = find_nearest(self.r, event.xdata)
        if event.button == 1:
            if self.shift_is_held:
                if all(self.vels['bulge'][1:]) == 0.:
                    return
                scale = event.ydata / self.vels['bulge'][idx]
                self.vels['bulge'] = scale * self.vels['bulge']
                self.b_ml = (scale ** 2.) * self.b_ml
                self.change_bulge()
                self.change_bary()
                self.change_total()
            else:
                if all(self.vels['disk'][1:]) == 0.:
                    return
                scale = event.ydata / self.vels['disk'][idx]
                self.vels['disk'] = scale * self.vels['disk']
                self.d_ml = (scale ** 2.) * self.d_ml
                self.change_disk()
                self.change_bary()
                self.change_total()
        elif event.button == 2:
            if all(self.vels['bulge'][1:]) == 0.:
                return
            scale = event.ydata / self.vels['bulge'][idx]
            self.vels['bulge'] = scale * self.vels['bulge']
            self.b_ml = (scale ** 2.) * self.b_ml
            self.change_bulge()
            self.change_bary()
            self.change_total()
        else:
            self.r_c = event.xdata
            self.v_h = event.ydata
            self.change_halo()
            self.change_total()


def scale_stellar_by_ml(v, ml):
    # format so that 0.79999099123 doesn't get past
    if 0.3 < float('{0:.1f}'.format(ml)) < 0.8:
        upper = v * np.sqrt((ml + 0.1) / ml)
        lower = v * np.sqrt((ml - 0.1) / ml)
    else:
        upper = v
        lower = v
    return upper, lower


def fit_halo_to_resid(r, v, vbary):
    # TODO: remove redundancy with fithalo.do_fixed_ml_fit
    vfit = np.sqrt((v['obs'] ** 2.) - (vbary ** 2.)).dropna()[1:]
    rfit = r.reindex(vfit.index)
    vfit_err = ((v['obs'] * v['obs_err']) / np.sqrt((v['obs'] ** 2.) - (vbary ** 2.))).reindex(vfit.index)
    popt, perr, chi_sq, red_chi_sq = fit_dm_halo_fixed_ml(rfit, vfit, vfit_err)
    halo_fit_params = {'v_h': popt[0], 'v_h_err': perr[0], 'r_c': popt[1], 'r_c_err': perr[1],
                       'chi_sq': chi_sq, 'red_chi_sq': red_chi_sq}
    vhalo = halo(r, halo_fit_params['v_h'], halo_fit_params['r_c'])
    vhalo[0] = 0.  # replace first row NaN with 0
    vtot = np.sqrt((vbary**2.) + (vhalo**2.))
    return vhalo, vtot


def ml_range(r, v, d_ml, b_ml):
    # add +/- 0.1 to M/L, if within limits 0.3 <= M/L <= 0.8
    vdisk_upper, vdisk_lower = scale_stellar_by_ml(v['disk'], d_ml)
    vbulge_upper, vbulge_lower = scale_stellar_by_ml(v['bulge'], b_ml)
    vbary_upper = np.sqrt((v['gas'] ** 2.) + (vdisk_upper ** 2.) + (vbulge_upper ** 2.))
    vbary_lower = np.sqrt((v['gas'] ** 2.) + (vdisk_lower ** 2.) + (vbulge_lower ** 2.))
    vhalo_upper, vtot_upper = fit_halo_to_resid(r, v, vbary_upper)
    vhalo_lower, vtot_lower = fit_halo_to_resid(r, v, vbary_lower)

    vdisk_range = (vdisk_upper, vdisk_lower)
    vbulge_range = (vbulge_upper, vbulge_lower)
    vbary_range = (vbary_upper, vbary_lower)
    vhalo_range = (vhalo_upper, vhalo_lower)
    vtot_range = (vtot_upper, vtot_lower)

    return vdisk_range, vbulge_range, vbary_range, vhalo_range, vtot_range


def find_nearest(radii, xvalue):
    return (np.abs(radii - xvalue)).idxmin()

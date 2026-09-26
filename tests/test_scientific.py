"""Independent numerical, dimensional, limiting-case and synthetic-data checks."""
import contextlib
import io
import unittest

import numpy as np
from astropy import constants as const, units as u
from astropy.modeling.models import BlackBody
from scipy.integrate import quad

from astrothesispy.utiles import u_conversion as c
from astrothesispy.utiles import utiles_alma as alma
from astrothesispy.utiles import utiles_physical as physical
from astrothesispy.utiles import utiles_molecules as molecules


class ScientificTests(unittest.TestCase):
    def test_planck_integrates_to_stefan_boltzmann(self):
        temperature = 50*u.K
        scale = (const.k_B*temperature/const.h).to_value(u.Hz)
        integral = quad(lambda x: c.planck(x*scale, temperature).value*scale, 1e-8, 80)[0]
        expected = (const.sigma_sb*temperature**4/np.pi).to_value(u.W/u.m**2)
        np.testing.assert_allclose(integral, expected, rtol=1e-7)

    def test_planck_frequency_wavelength_jacobian(self):
        wavelength = np.array([10, 100, 1000])*u.um
        bnu = BlackBody(temperature=40*u.K)(wavelength)
        expected = bnu.to(u.W/(u.m**2*u.m*u.sr), equivalencies=u.spectral_density(wavelength))
        actual = c.planckwave(wavelength, 40)
        np.testing.assert_allclose(actual.value, expected.value, rtol=1e-12)
        self.assertEqual(actual.unit, expected.unit)

    def test_luminosity_flux_round_trip_quantity_distance(self):
        flux = c.lum_to_flux(1e6, 3500*u.kpc, z=False)
        np.testing.assert_allclose(c.flux_to_lum(flux, 3500*u.kpc, z=False).value, 1e6)

    def test_line_flux_frequency_wavelength_and_false_flag(self):
        expected = (1*u.Jy*u.km/u.s * 100*u.GHz / const.c.to(u.km/u.s)).to_value(u.W/u.m**2)
        wavelength = (100*u.GHz).to(u.mm, equivalencies=u.spectral())
        for actual in [c.jykms_to_wm2(1, 100), c.jykms_to_wm2(1, 100, False),
                       c.jykms_to_wm2(1, wavelength, wavelength=True)]:
            self.assertAlmostEqual(actual.value/expected, 1)
        for value, use_wavelength in [(100, False), (wavelength, True)]:
            recovered = c.wm2_to_jykms(expected, value, wavelength=use_wavelength)
            self.assertAlmostEqual(recovered.value, 1)

    def test_flux_density_width_roundtrip(self):
        # This legacy pair uses an equivalent rectangular bandwidth, not a Gaussian FWHM.
        flux = c.jy_to_wm2(2, 0.01, False)
        self.assertAlmostEqual(flux.value/2e-19, 1)
        self.assertAlmostEqual(c.wm2_to_jy(flux, 0.01, wavelength=False).value, 2)

    def test_gaussian_temperature_uncertainty_and_aliases(self):
        # Independent NRAO rounded coefficient and analytical fractional derivatives.
        expected_T = 1.222e6/(100**2*2*1)
        for function in [c.Jybeam_to_T_fromGaussian, c.Jybeam_to_T_fromGaussian_v2,
                         c.Jybeam_to_T_fromGaussian_array]:
            temp, error = function(1, 100, 2, 1, .1, .2, .2)
            self.assertAlmostEqual(temp.to_value(u.K)/expected_T, 1, places=3)
            self.assertAlmostEqual((error/temp).value, np.sqrt(.1**2+.1**2+.2**2))
            temp, error = function(np.array([0, 1]), 100, 2, 1)
            np.testing.assert_array_equal(error.value, [0, 0])
            np.testing.assert_allclose(temp.value, [0, expected_T], rtol=2e-4)

    def test_negative_ra_pixel_scale_does_not_negate_flux(self):
        positive = c.Jybeam_to_Jy(np.ones(4), 1, 1, .1, .1)
        negative = c.Jybeam_to_Jy(np.ones(4), 1, 1, -.1, .1)
        self.assertEqual(positive, negative)
        np.testing.assert_allclose(np.sum(c.Jybeam_to_Jypx(np.ones(4), -.1, .1, 1, 1)), positive)

    def test_alma_angular_scales(self):
        # lambda/D at 100 GHz and 12 m, and lambda/B/5 at 1 km baseline.
        np.testing.assert_allclose(alma.FOV_telescope(100, 12), 51.53, rtol=2e-5)
        np.testing.assert_allclose(alma.cellsize_ALMA(100, 1), .12367, rtol=3e-5)
        self.assertAlmostEqual(alma.HPBW(100*u.GHz, 12).to_value(u.arcsec),
                               1.13*alma.FOV_telescope(100, 12))

    def test_escape_speed_satisfies_energy_equation(self):
        velocity = physical.escape_velocity(1e5, 1)
        recovered_mass = (velocity**2 * u.pc / (2*const.G)).to_value(u.Msun)
        self.assertAlmostEqual(recovered_mass, 1e5, places=6)
        self.assertEqual(velocity.unit, u.km/u.s)

    def test_freefall_has_time_units_and_density_scaling(self):
        first = physical.freefall_time(1e5, 1)
        second = physical.freefall_time(4e5, 1)
        self.assertEqual(first.unit, u.yr)
        self.assertAlmostEqual((first/second).value, 2)
        self.assertAlmostEqual(first.value, 52368.5938654, places=5)

    def test_virial_mass_has_mass_units(self):
        value = physical.virial_mass_mccrady(1, 10)
        expected = (10*u.pc*(10*u.km/u.s)**2/const.G).to(u.Msun)
        self.assertEqual(value.unit, u.Msun)
        self.assertAlmostEqual((value/expected).value, 1)

    def test_dust_optical_depth_recovers_synthetic_slab(self):
        omega = (alma.beam_size(1, 1)*u.arcsec**2).to(u.sr)
        intensity = BlackBody(temperature=50*u.K)(350*u.GHz)
        flux = (intensity * omega * (1-np.exp(-.1))).to_value(u.Jy)
        tau, mass = physical.gas_mass_from_dust_leroy_v2(flux, 1, 1, 50, 350, 3.5)
        self.assertAlmostEqual(tau, .1)
        self.assertGreater(mass, 0)

    def test_column_error_matches_numerical_jacobian(self):
        slope, intercept = -.001, 13.
        sigma = np.array([1e-5, .02])
        covariance = np.array([[sigma[0]**2, -1e-7], [-1e-7, sigma[1]**2]])
        def column(m, b):
            T = -1/(np.log(10)*m)
            return (const.k_B*T*u.K/(const.h*4.55*u.GHz)).to_value(u.one)*10**b
        steps = [1e-8, 1e-5]
        derivative = np.array([(column(slope+steps[0], intercept)-column(slope-steps[0], intercept))/(2*steps[0]),
                               (column(slope, intercept+steps[1])-column(slope, intercept-steps[1]))/(2*steps[1])])
        N, error, T, T_error = molecules.HC3N_rotational_diagram_from_fit(
            slope, sigma[0], intercept, sigma[1], covariance[0, 1])
        np.testing.assert_allclose(error, np.sqrt(derivative@covariance@derivative), rtol=1e-7)
        self.assertAlmostEqual(T_error/T, sigma[0]/abs(slope))

    def test_rotational_fit_honors_J_and_log_uncertainties(self):
        lines = []
        for energy in [100., 300., 500., 700.]:
            N = 10**(13-energy/(200*np.log(10)))
            lines.append({'qn1': 10, 'EUP_K': energy, 'Nu/gu': N, 'Nu/gu_err': .1*N,
                          'logNu/gu': np.log10(N), 'logNu/gu_err': np.log10(.1*N)})
        with contextlib.redirect_stdout(io.StringIO()):
            state = np.random.get_state()
            try:
                np.random.seed(1001)
                table, fit = molecules.rotational_diagram({'synthetic': lines}, Jselect=10, bootstrap=True)
            finally:
                np.random.set_state(state)
        np.testing.assert_allclose(table['logNu/gu_relerr'], .1/np.log(10))
        self.assertAlmostEqual(fit['NoBootstrap']['Tex'], 200, places=6)
        self.assertGreater(fit['NoBootstrap']['Tex_err'], 0)
        self.assertLess(fit['NoBootstrap']['reduced_chi2'], 1e-10)
        np.testing.assert_allclose(fit['Bootstrap']['covariance'],
                                   fit['NoBootstrap']['covariance'], rtol=.2)

    def test_thin_column_against_si_einstein_coefficient(self):
        frequency = 100*u.GHz
        dipole = (3.7*u.Debye).to(u.C*u.m)
        J = 10
        A = (16*np.pi**3*frequency**3*dipole**2*J /
             (3*const.eps0*const.h*const.c**3*(2*J+1))).to(1/u.s)
        omega = (np.pi/(4*np.log(2))*u.arcsec**2).to(u.sr)
        Tb = (1*u.Jy/u.sr/omega.value).to(u.K, equivalencies=u.brightness_temperature(frequency))
        expected = (8*np.pi*const.k_B*frequency**2*Tb*u.km/u.s /
                    (const.h*const.c**3*A*(2*J+1))).to_value(1/u.cm**2)
        actual, _ = molecules.Columndensity_thin(100, 1, .1, J, 3.7, 1, 1)
        # Legacy 1.224e6 beam-temperature constant is rounded (~0.154% high).
        np.testing.assert_allclose(actual, expected, rtol=.002)

    def test_optical_depth_correction_thin_limit(self):
        args = (100, 1, .1, 10, 3.7, 1, 1)
        thin = molecules.Columndensity_thin(*args)
        for tau in [0, 1e-10]:
            thick = molecules.Columndensity_thick(*args, tau=tau)
            np.testing.assert_allclose(thick, thin, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()

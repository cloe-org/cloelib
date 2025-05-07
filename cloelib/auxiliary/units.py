from astropy import units
from astropy.constants import G

# Astronomical constants
SPEED_OF_LIGHT = 2.99792458E8         # Speed of light in m/s
GRAVITATIONAL_CONSTANT = G.to(units.Mpc**3.0 / (units.Msun * units.s**2.0)).value # Gravitational constant in Mpc^3/Msun/s^2
MPC_TO_KM = units.Mpc.to(units.km) # 1 Mpc in km

# Definition of units based on astropy
from astropy.constants import G, M_sun, R_sun, au, pc, c, h, k_B, m_p, m_e # type: ignore
from astropy.units import yr, s, day, km, m # type: ignore

# Astronomical constants
GRAVITATIONAL_CONSTANT = G.value  # Gravitational constant in m^3 kg^-1 s^-2
SOLAR_MASS = M_sun.value         # Solar mass in kilograms
SOLAR_RADIUS = R_sun.value       # Solar radius in meters
ASTRONOMICAL_UNIT = au.value     # Astronomical unit in meters
PARSEC = pc.value                # Parsec in meters
SPEED_OF_LIGHT = c.value         # Speed of light in m/s

# Planck and Boltzmann constants
PLANCK_CONSTANT = h.value        # Planck constant in J·s
BOLTZMANN_CONSTANT = k_B.value   # Boltzmann constant in J/K

# Particle masses
PROTON_MASS = m_p.value          # Proton mass in kilograms
ELECTRON_MASS = m_e.value        # Electron mass in kilograms

# Time conversions
YEAR_IN_SECONDS = yr.to(s)       # Year in seconds
DAY_IN_SECONDS = day.to(s)       # Day in seconds

# Distance conversions
KILOMETER = km.to(m)             # Kilometer in meters

# Export as dictionary (optional for introspection)
CONSTANTS = {
    "G": GRAVITATIONAL_CONSTANT,
    "M_sun": SOLAR_MASS,
    "R_sun": SOLAR_RADIUS,
    "AU": ASTRONOMICAL_UNIT,
    "pc": PARSEC,
    "c": SPEED_OF_LIGHT,
    "h": PLANCK_CONSTANT,
    "k_B": BOLTZMANN_CONSTANT,
    "m_p": PROTON_MASS,
    "m_e": ELECTRON_MASS,
    "yr": YEAR_IN_SECONDS,
    "day": DAY_IN_SECONDS,
    "km": KILOMETER,
}

if __name__ == "__main__":
    for name, value in CONSTANTS.items():
        print(f"{name}: {value}")


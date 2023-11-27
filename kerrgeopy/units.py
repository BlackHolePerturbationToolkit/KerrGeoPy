"""Module containing functions to convert between geometrized units and physical units."""
# https://en.wikipedia.org/wiki/Geometrized_unit_system

# speed of light in m/s
c = 299792458.0
# gravitational constant in m^3 kg^-1 s^-2
G = 6.67408e-11
# solar mass in kg
solar_mass = 1.98847e30


def mass_in_kg(M):
    """
    Converts solar masses to kg

    :param M: mass in solar masses
    :type M: double

    :return: mass in kg
    :rtype: double
    """
    return M * solar_mass


def distance_in_meters(d, M):
    """Converts distance in geometrized units to meters

    Parameters
    ----------
    d : double
        distance in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        distance in meters
    """
    return d * G / c**2 * mass_in_kg(M)


def distance_in_cm(d, M):
    """Converts distance in geometrized units to centimeters

    Parameters
    ----------
    d : double
        distance in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        distance in centimeters
    """
    return distance_in_meters(d, M) * 100


def distance_in_km(d, M):
    """Converts distance in geometrized units to kilometers

    Parameters
    ----------
    d : double
        distance in multiples of M

    Returns
    -------
    double
        distance in kilometers
    """
    return distance_in_meters(d, M) / 1000


def distance_in_lightyears(d, M):
    """Converts distance in geometrized units to lightyears

    Parameters
    ----------
    d : double
        distance in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        distance in lightyears
    """
    return distance_in_meters(d, M) / 9.461e15


def distance_in_au(d, M):
    """Converts distance in geometrized units to astronomical units

    Parameters
    ----------
    d : double
        distance in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        distance in astronomical units
    """
    return distance_in_meters(d, M) / 1.496e11


def time_in_seconds(t, M):
    """Converts time in geometrized units to seconds

    Parameters
    ----------
    t : double
        time in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        time in seconds
    """
    return t * G / c**3 * mass_in_kg(M)


def time2_in_seconds2(t, M):
    """Converts time^2 in geometrized units to seconds^2

    Parameters
    ----------
    t : double
        time^2 in multiples of M^2
    M : double
        mass of the primary body in solar masses
    """
    return t * (G / c**3) ** 2 * mass_in_kg(M) ** 2


def time_in_days(t, M):
    """Converts time in geometrized units to days

    Parameters
    ----------
    t : double
        time in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        time in days
    """
    return time_in_seconds(t, M) / 86400


def energy_in_joules(E, M):
    """Converts energy in geometrized units to joules

    Parameters
    ----------
    E : double
        energy in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        energy in joules
    """
    return E * c**2 * mass_in_kg(M)


def energy_in_ergs(E, M):
    """Converts energy in geometrized units to ergs

    Parameters
    ----------
    E : double
        energy in multiples of M
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        energy in ergs
    """
    return energy_in_joules(E, M) * 1e7


def angular_momentum_in_mks(L, M):
    """Converts angular momentum in geometrized units to kg m^2 s^-1

    Parameters
    ----------
    L : double
        angular momentum in multiples of M^2
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        angular momentum in kg m^2 s^-1
    """
    return L * G / c * mass_in_kg(M) ** 2


def angular_momentum_in_cgs(L, M):
    """Converts angular momentum in geometrized units to g cm^2 s^-1

    Parameters
    ----------
    L : double
        angular momentum in multiples of M^2
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        angular momentum in g cm^2 s^-1
    """
    return angular_momentum_in_mks(L, M) * 1e7


def carter_constant_in_mks(Q, M):
    """Converts Carter constant in geometrized units to kg^2 m^4 s^-2

    Parameters
    ----------
    Q : double
        Carter constant in multiples of M^4
    M : double
        mass of the primary body in solar masses
    """
    return Q * G**2 / c**2 * mass_in_kg(M) ** 4


def carter_constant_in_cgs(Q, M):
    """Converts Carter constant in geometrized units to g^2 cm^4 s^-2

    Parameters
    ----------
    Q : double
        Carter constant in multiples of M^4
    M : double
        mass of the primary body in solar masses
    """
    return carter_constant_in_mks(Q, M) * 1e14


def frequency_in_Hz(f, M):
    """Converts frequency in geometrized units to hertz

    Parameters
    ----------
    f : _type_
        frequency in multiples of M^-1
    M : _type_
        _description_
    """
    return f * c**3 / G / mass_in_kg(M)


def frequency_in_mHz(f, M):
    """Converts frequency in geometrized units to millihertz

    Parameters
    ----------
    f : double
        frequency in multiples of M^-1
    M : double
        mass of the primary body in solar masses

    Returns
    -------
    double
        frequency in millihertz
    """
    return frequency_in_Hz(f, M) * 1e3

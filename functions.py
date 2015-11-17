def yearQuarter2Index(year, quarter, year0=1948, q0=1):
    # Convert year and month to integer index
    return 4 * (year - year0) + quarter - q0 + 1

#######################################################################################################################
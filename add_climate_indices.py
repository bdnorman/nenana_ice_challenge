"""
Add ENSO (Nino 3.4), Arctic Oscillation (AO), and Pacific Decadal Oscillation (PDO)
monthly indices to the Nenana dataset for February and March of each year.

Sources:
  Nino 3.4: https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
  AO:       https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table
  PDO:      https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat
"""

import pandas as pd

# Nino 3.4 anomaly (relative to 1991-2020 base period)
nino34 = {
    1989: {"feb": -1.53, "march": -1.16},
    1990: {"feb":  0.38, "march":  0.31},
    1991: {"feb":  0.54, "march":  0.34},
    1992: {"feb":  0.89, "march":  0.64},
    1993: {"feb":  0.10, "march":  0.05},
    1994: {"feb": -0.06, "march":  0.05},
    1995: {"feb":  0.71, "march":  0.71},
    1996: {"feb": -0.52, "march": -0.40},
    1997: {"feb":  0.19, "march":  0.24},
    1998: {"feb":  0.59, "march":  0.30},
    1999: {"feb": -1.64, "march": -1.35},
    2000: {"feb": -1.47, "march": -1.43},
    2001: {"feb": -0.82, "march": -0.71},
    2002: {"feb":  0.46, "march":  0.25},
    2003: {"feb":  0.69, "march":  0.60},
    2004: {"feb":  0.38, "march":  0.10},
    2005: {"feb":  0.67, "march":  0.61},
    2006: {"feb": -0.67, "march": -0.59},
    2007: {"feb":  0.46, "march":  0.34},
    2008: {"feb": -1.58, "march": -1.35},
    2009: {"feb": -0.85, "march": -0.58},
    2010: {"feb":  0.88, "march":  0.75},
    2011: {"feb": -1.05, "march": -0.81},
    2012: {"feb": -0.91, "march": -0.77},
    2013: {"feb": -0.19, "march": -0.37},
    2014: {"feb":  0.07, "march":  0.28},
    2015: {"feb":  0.87, "march":  0.90},
    2016: {"feb":  1.22, "march":  1.07},
    2017: {"feb": -0.09, "march": -0.03},
    2018: {"feb": -0.35, "march": -0.22},
    2019: {"feb":  0.71, "march":  0.64},
    2020: {"feb":  0.78, "march":  0.75},
    2021: {"feb": -0.96, "march": -0.65},
    2022: {"feb": -0.37, "march": -0.69},
    2023: {"feb": -0.55, "march": -0.25},
    2024: {"feb":  1.26, "march":  0.99},
    2025: {"feb": -0.52, "march": -0.34},
    2026: {"feb":  0.20, "march":  0.35},
}

# Arctic Oscillation index
ao_index = {
    1989: {"feb":  3.106, "march":  3.279},
    1990: {"feb":  1.001, "march":  3.402},
    1991: {"feb":  0.723, "march": -0.876},
    1992: {"feb":  0.550, "march":  1.122},
    1993: {"feb":  3.495, "march":  0.184},
    1994: {"feb": -0.288, "march": -0.862},
    1995: {"feb": -0.154, "march":  1.429},
    1996: {"feb": -1.200, "march":  0.163},
    1997: {"feb": -0.457, "march":  1.889},
    1998: {"feb": -2.081, "march": -0.183},
    1999: {"feb":  0.110, "march":  0.482},
    2000: {"feb":  1.270, "march":  1.076},
    2001: {"feb": -0.959, "march": -0.622},
    2002: {"feb":  1.381, "march":  1.304},
    2003: {"feb": -0.472, "march":  0.128},
    2004: {"feb": -1.686, "march": -1.528},
    2005: {"feb":  0.356, "march": -1.271},
    2006: {"feb": -0.170, "march": -0.156},
    2007: {"feb":  2.034, "march": -1.307},
    2008: {"feb":  0.819, "march":  0.938},
    2009: {"feb":  0.800, "march": -0.672},
    2010: {"feb": -2.587, "march": -4.266},
    2011: {"feb": -1.683, "march":  1.575},
    2012: {"feb": -0.220, "march": -0.036},
    2013: {"feb": -0.610, "march": -1.007},
    2014: {"feb": -0.969, "march":  0.044},
    2015: {"feb":  1.092, "march":  1.043},
    2016: {"feb": -1.449, "march": -0.024},
    2017: {"feb":  0.942, "march":  0.340},
    2018: {"feb": -0.281, "march":  0.113},
    2019: {"feb": -0.713, "march":  1.149},
    2020: {"feb":  2.419, "march":  3.417},
    2021: {"feb": -2.484, "march": -1.191},
    2022: {"feb":  0.848, "march":  1.544},
    2023: {"feb": -0.674, "march":  1.600},
    2024: {"feb": -0.210, "march":  0.635},
    2025: {"feb": -0.069, "march": -0.864},
    2026: {"feb": -2.048, "march": -1.256},
}

# Pacific Decadal Oscillation index
pdo_index = {
    1989: {"feb": -1.24, "march": -1.45},
    1990: {"feb": -0.42, "march": -1.28},
    1991: {"feb": -1.80, "march": -1.09},
    1992: {"feb":  0.08, "march":  0.17},
    1993: {"feb": -0.28, "march":  0.02},
    1994: {"feb":  0.85, "march":  0.33},
    1995: {"feb": -0.86, "march":  0.02},
    1996: {"feb":  1.01, "march":  1.01},
    1997: {"feb":  0.44, "march":  0.29},
    1998: {"feb":  1.05, "march":  1.52},
    1999: {"feb": -0.78, "march": -1.06},
    2000: {"feb": -2.20, "march": -1.28},
    2001: {"feb":  0.48, "march": -0.01},
    2002: {"feb": -0.42, "march": -1.51},
    2003: {"feb":  1.45, "march":  1.23},
    2004: {"feb": -0.55, "march": -0.21},
    2005: {"feb": -0.15, "march": -0.01},
    2006: {"feb":  0.54, "march":  0.38},
    2007: {"feb": -0.69, "march": -0.71},
    2008: {"feb": -1.50, "march": -1.46},
    2009: {"feb": -1.81, "march": -1.78},
    2010: {"feb":  0.05, "march":  0.25},
    2011: {"feb": -1.80, "march": -1.46},
    2012: {"feb": -1.85, "march": -1.35},
    2013: {"feb": -1.10, "march": -1.42},
    2014: {"feb": -0.57, "march": -0.42},
    2015: {"feb":  1.51, "march":  1.52},
    2016: {"feb":  0.75, "march":  1.29},
    2017: {"feb": -0.06, "march": -0.02},
    2018: {"feb":  0.38, "march": -0.09},
    2019: {"feb": -0.34, "march": -0.72},
    2020: {"feb": -1.41, "march": -1.48},
    2021: {"feb": -0.61, "march": -1.09},
    2022: {"feb": -2.40, "march": -1.91},
    2023: {"feb": -1.24, "march": -1.65},
    2024: {"feb": -1.57, "march": -1.34},
    2025: {"feb": -1.29, "march": -1.40},
    2026: {"feb": -1.24, "march": -1.00},
}


def main():
    csv_path = "NenanaIceClassic_1917-2021.csv"
    df = pd.read_csv(csv_path)

    for col in ["nino34_feb", "nino34_march", "ao_feb", "ao_march", "pdo_feb", "pdo_march"]:
        df[col] = None

    for idx, row in df.iterrows():
        year = int(row["Year"])
        if year in nino34:
            df.at[idx, "nino34_feb"]    = nino34[year]["feb"]
            df.at[idx, "nino34_march"]  = nino34[year]["march"]
            df.at[idx, "ao_feb"]        = ao_index[year]["feb"]
            df.at[idx, "ao_march"]      = ao_index[year]["march"]
            df.at[idx, "pdo_feb"]       = pdo_index[year]["feb"]
            df.at[idx, "pdo_march"]     = pdo_index[year]["march"]

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with climate indices for {len(nino34)} years.")
    print("New columns:", ["nino34_feb", "nino34_march", "ao_feb", "ao_march", "pdo_feb", "pdo_march"])


if __name__ == "__main__":
    main()

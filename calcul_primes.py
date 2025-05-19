import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Setup logging
logging.basicConfig(filename='premium_calculation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CONFIG = {
    'input_file': "Base de doonées projet vie.xlsb",
    'output_file': "résultats_primesS.xlsx",
    'sample_size': 10,
    'random_state': 42,
    'provision_date': datetime(2024, 12, 31),
    'capital_threshold': 200000,
    'h_high': 0.01,  # Acquisition fee for SA0 >= 200,000 TND
    'h_low': 0.10,   # Acquisition fee for SA0 < 200,000 TND
    'technical_rate': 0.02,  # Default annual technical rate (i)
    'f': 12,  # Monthly payments
    'capital_decrease': 'interest',  # Interest-based decrease
    'mortality_table': None,  # Path to TD 99 table CSV
    'required_columns': [
        'DATE_NAISSANCE', 'EFFET', 'FR', 'NB_MENSUALITES', 'CAPITAL',
        'TX_intérêt', 'TAUX_TECHNIQUE', 'FR_GESTION', 'TX_AGGRAV', 'POLICE'
    ]
}

def load_mortality_table(file_path):
    """Load TD 99 mortality table (lx values) from CSV."""
    if file_path and os.path.exists(file_path):
        try:
            table = pd.read_csv(file_path, index_col='age')
            return table['lx']
        except Exception as e:
            logging.error(f"Failed to load mortality table: {e}")
            return None
    return None

def get_mortality_rate(age, j, lx_table):
    """Calculate qx for age + j/12 years."""
    if lx_table is not None:
        x = int(age + (j - 1) // 12)
        if x in lx_table.index and x + 1 in lx_table.index:
            lx = lx_table[x]
            lx_plus_1 = lx_table[x + 1]
            return (lx - lx_plus_1) / lx if lx > 0 else 0.001
        logging.warning(f"Mortality data missing for age {x}")
    return 0.001 + 0.0001 * (age + (j - 1) // 12)  # Fallback

def validate_data(df, required_columns):
    """Validate required columns and data."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    null_counts = df[required_columns].isna().sum()
    if null_counts.any():
        logging.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    for col in ['TX_intérêt', 'FR_GESTION', 'TX_AGGRAV', 'TAUX_TECHNIQUE']:
        if col in df.columns:
            if (df[col] < 0).any():
                logging.warning(f"Negative values in {col}")
            if (df[col] > 100).any():
                logging.warning(f"High percentages (>100) in {col}")
    return df

def preprocess_dates(df):
    """Convert date columns to datetime."""
    try:
        df['DATE_NAISSANCE'] = pd.to_datetime(df['DATE_NAISSANCE'], errors='coerce')
        df['EFFET'] = pd.to_datetime(df['EFFET'], errors='coerce')
        null_dates = df[['DATE_NAISSANCE', 'EFFET']].isna().sum()
        if null_dates.any():
            logging.warning(f"Invalid dates found:\n{null_dates}")
        return df
    except Exception as e:
        logging.error(f"Date preprocessing failed: {e}")
        raise

def calculate_age(date_naissance, effet):
    """Calculate age as nearest whole year."""
    try:
        months_diff = (effet.year - date_naissance.year) * 12 + effet.month - date_naissance.month
        if effet.day < date_naissance.day:
            months_diff -= 1
        return round(months_diff / 12)
    except Exception as e:
        logging.error(f"Age calculation failed: {e}")
        return np.nan

def calculate_capital_remaining(j, SA0, fr, n, Te_monthly, decrease_type):
    """Calculate SAj per document formulas."""
    if j <= fr + 1:
        return SA0
    if decrease_type == 'linear':
        return SA0 * (1 - (j - fr - 1) / (n - fr))
    elif decrease_type == 'interest':
        return SA0 * (1 - (1 + Te_monthly) ** (-(n + 1 - j))) / (1 - (1 + Te_monthly) ** (-(n - fr)))
    raise ValueError(f"Invalid decrease_type: {decrease_type}")

def calculate_premiums(row, config, lx_table):
    """Calculate PU, PU', PU'', and kVx per document formulas."""
    try:
        # Extract parameters
        date_naissance = pd.to_datetime(row['DATE_NAISSANCE'])
        effet = pd.to_datetime(row['EFFET'])
        age = calculate_age(date_naissance, effet)
        fr = int(row['FR'])
        n = int(row['NB_MENSUALITES'])
        SA0 = float(row['CAPITAL'])
        Te_annual = float(row['TX_intérêt']) / 100  # Annual interest rate
        Te_monthly = (1 + Te_annual) ** (1 / 12) - 1  # Convert to monthly
        i = float(row['TAUX_TECHNIQUE']) / 100 if pd.notna(row['TAUX_TECHNIQUE']) else config['technical_rate']
        g = float(row['FR_GESTION']) / 100  # Management fee
        aggr = float(row['TX_AGGRAV']) / 100  # Aggravation rate
        f = config['f']
        decrease_type = config['capital_decrease']
        provision_date = config['provision_date']

        # Validate
        if pd.isna([age, fr, n, SA0, Te_annual, i, g, aggr]).any():
            raise ValueError("Missing or invalid data")

        # Calculate SAj
        SAj = np.array([calculate_capital_remaining(j, SA0, fr, n, Te_monthly, decrease_type)
                        for j in range(1, n + 1)])

        # Calculate qx, lx, and discount factors
        qx = np.array([get_mortality_rate(age, j, lx_table) for j in range(1, n + 1)])
        lx = np.array([lx_table[int(age + (j - 1) // 12)] if lx_table is not None else 1000
                       for j in range(1, n + 1)])
        lx_plus_1 = np.array([lx_table[int(age + (j - 1) // 12) + 1] if lx_table is not None else 1000
                              for j in range(1, n + 1)])
        discount_pu = (1 + i) ** (-((np.arange(n) + 0.5) / 12))
        discount_g = (1 + i) ** (-(np.arange(n) / 12))

        # Pure premium (PU)
        if lx_table is not None:
            terms = SAj * ( лей - lx_plus_1) / (lx * discount_pu)
            prime_pure = (1 + aggr) / f * np.sum(terms)
        else:
            terms = SAj * qx / discount_pu
            prime_pure = (1 + aggr) / f * np.sum(terms)

        # Inventory premium (PU')
        g_terms = SAj * lx / (lx * discount_g) * g
        prime_inventaire = prime_pure + (1 + aggr) / f * np.sum(g_terms)

        # Commercial premium (PU'')
        H = config['h_high'] if SA0 >= config['capital_threshold'] else config['h_low']
        prime_commerciale = prime_inventaire / (1 - H)

        # Mathematical provision (kVx)
        mois_effet = max(0, (provision_date - effet).days // 30)
        if mois_effet >= n:
            pm = 0
        else:
            k = mois_effet
            SAj_future = SAj[k:n]
            lx_future = lx[k:n]
            lx_plus_1_future = lx_plus_1[k:n]
            discount_pu_future = (1 + i) ** (-((np.arange(k, n) + 0.5 - k) / 12))
            discount_g_future = (1 + i) ** (-((np.arange(k, n) - k) / 12))
            if lx_table is not None:
                terms = SAj_future * (lx_future - lx_plus_1_future) / (lx_future[k] * discount_pu_future)
                g_terms = SAj_future * lx_future / (lx_future[k] * discount_g_future) * g
                pm = (1 + aggr) / f * np.sum(terms + g_terms)
            else:
                qx_future = qx[k:n]
                terms = SAj_future * qx_future / discount_pu_future
                g_terms = SAj_future / discount_g_future * g
                pm = (1 + aggr) / f * np.sum(terms + g_terms)

        return pd.Series([int(age), round(prime_pure, 2), round(prime_inventaire, 2),
                          round(prime_commerciale, 2), round(pm, 2)],
                         index=['AGE', 'PRIME_PURE', 'PRIME_INVENTAIRE', 'PRIME_COMMERCIALE', 'PM_31_12_2024'])
    except Exception as e:
        logging.error(f"Error in contract {row.get('POLICE', 'N/A')}: {e}")
        return pd.Series([None, None, None, None, None],
                         index=['AGE', 'PRIME_PURE', 'PRIME_INVENTAIRE', 'PRIME_COMMERCIALE', 'PM_31_12_2024'])

def main():
    print("Starting premium calculation...")
    
    # Load mortality table
    lx_table = load_mortality_table(CONFIG['mortality_table'])
    
    # Load data
    try:
        df = pd.read_excel(CONFIG['input_file'], engine='pyxlsb')
        print("Base chargée avec succès !")
    except Exception as e:
        logging.error(f"Failed to load file: {e}")
        raise

    # Sample data
    try:
        df = df.sample(n=min(CONFIG['sample_size'], len(df)), random_state=CONFIG['random_state'])
    except ValueError as e:
        logging.warning(f"Sampling error: {e}. Using full dataset.")
        df = df

    # Validate and preprocess
    df = validate_data(df, CONFIG['required_columns'])
    df = preprocess_dates(df)

    # Calculate premiums
    print("Calcul en cours...")
    resultats = df.apply(lambda row: calculate_premiums(row, CONFIG, lx_table), axis=1)

    # Combine results
    df_resultats = pd.concat([df, resultats], axis=1)

    # Save output
    output_file = CONFIG['output_file']
    if os.path.exists(output_file):
        output_file = f"backup_{output_file}"
        logging.warning(f"Output file exists, saving to {output_file}")
    
    df_resultats.to_excel(output_file, index=False, engine='openpyxl')
    print(f"✅ Calcul terminé ! Résultats enregistrés dans : {output_file}")

    # Summary
    summary = {
        'Total Contracts': len(df),
        'Successful Calculations': resultats['PRIME_PURE'].notna().sum(),
        'Errors': resultats['PRIME_PURE'].isna().sum()
    }
    print("Summary:", summary)
    logging.info(f"Calculation completed: {summary}")

if __name__ == "__main__":
    main()
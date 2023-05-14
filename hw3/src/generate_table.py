import datetime as dt

from typing import Any
from sys import argv

import pandas as pd
import numpy as np


RANDOM_SEED = 42
ROW_AMOUNT = 1000


np.random.seed(RANDOM_SEED)


def generate_hash_string(string_len: int = 30) -> str:

    symbols_choosen = np.random.choice(
        list('abcdef') + [str(i) for i in range(10)], 
        size=string_len
    )

    return ''.join(symbols_choosen.tolist())

def generate_random_date(
        limits_year: tuple[int, int] = (1950, 1990),
        limits_month: tuple[int, int] = (1, 12),
        limits_day: tuple[int, int] = (1, 28)       
    ) -> dt.date:
    
    years = list(range(limits_year[0], limits_year[-1] + 1))
    months = list(range(limits_month[0], limits_month[-1] + 1))
    days = list(range(limits_day[0], limits_day[-1] + 1))

    return dt.date(
        year=np.random.choice(years, 1)[0],
        month=np.random.choice(months, 1)[0],
        day=np.random.choice(days, 1)[0]
    )

def add_noise(
        df: pd.DataFrame, column_name: str,
        limits_nan_amount: tuple[int, int] = (5, 10),
        noise_value: Any = np.nan
    ) -> pd.DataFrame:

    nan_amount = np.random.randint(
        limits_nan_amount[0], limits_nan_amount[-1],
        size=1
    )[0]
    row_amount = df.shape[0]

    noise_indices = np.random.choice(
        np.arange(row_amount), size=nan_amount,
        replace=False
    )

    df.loc[noise_indices, column_name] = noise_value

    return df

def generate_table(row_amount: int) -> pd.DataFrame:
    
    pacient_ids = [generate_hash_string() for _ in range(row_amount)]
    birthdays = [generate_random_date() for _ in range(row_amount)]
    genders = np.random.choice(
        ['female', 'male'], size=row_amount, p=[2./3, 1./3]
    )
    smoking = np.random.choice(
        ['no', 'yes'], size=row_amount, p=[0.85, 0.15]
    )
    obesity = np.random.choice(
        ['no', 'yes'], size=row_amount, p=[0.75, 0.25]
    )
    diabetes = np.random.choice(
        ['no', 'yes'], size=row_amount, p=[0.81, 0.19]
    )
    hypertension = np.random.choice(
        ['no', 'yes'], size=row_amount, p=[0.13, 0.87]
    )
    devices = np.random.choice(
        ['CRT-D', 'CRT'], size=row_amount, p=[2./ 3, 1. / 3]
    )

    mitrals_before = np.random.choice(
        [0, 1, 2, 3, 4],
        size=row_amount, p=[0.69, 0.03, 0.18, 0.07, 0.03]
    )
    mitrals_after = mitrals_before.copy()
    indicators = np.random.randint(0, 2, size=row_amount)
    mitrals_after[(indicators == 1) & (mitrals_after > 1)] -= 1

    fc_before = np.random.choice(
        [1, 2, 3, 4], size=row_amount,
        p=[0.1, 0.57, 0.27, 0.06]
    )
    fc_after = fc_before.copy()
    mask = fc_after > 1
    deltas = np.random.randint(0, 4, size=np.sum(mask)) % fc_after[mask]
    fc_after[mask] -= deltas

    df = pd.DataFrame({
        'pacient_id': pacient_ids,
        'birthday': birthdays,
        'gender': genders,
        'smoking': smoking,
        'obesity': obesity,
        'diabete': diabetes,
        'hypertension': hypertension,
        'device': devices,
        'mitral_before': mitrals_before,
        'mitral_after': mitrals_after,
        'fc_before': fc_before,
        'fc_after': fc_after
    })

    df = add_noise(df, 'gender', (20, 40), 'helicopter')
    df = add_noise(df, 'smoking')
    df = add_noise(df, 'obesity', noise_value='+/-')
    df = add_noise(df, 'device')
    df = add_noise(df, 'device', noise_value='ps5')
    df = add_noise(df, 'mitral_before')
    df = add_noise(df, 'mitral_after')

    return df


if __name__ == '__main__':

    if len(argv) != 2:
        raise RuntimeError('not enough arguments;')
    
    if not 'csv' in argv[-1]:
        argv[-1] += '.csv'

    df = generate_table(ROW_AMOUNT)
    df.to_csv(argv[-1], index=False)

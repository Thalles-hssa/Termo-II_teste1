import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from chemicals.vapor_pressure import Lee_Kesler
from chemicals.identifiers import CAS_from_any
from chemicals.critical import Tc, Pc
from chemicals.acentric import omega
from thermo import Chemical

CSV_PATH = 'Constantes_Antoine_corrigido.csv'
COMPOSTOS = ['CO', 'CO2']


def carregar_biblioteca(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    df = df.rename(columns={
        'Formula': 'formula',
        'Composto': 'composto',
        'Ant(A)': 'A',
        'Ant(B)': 'B',
        'Ant(C)': 'C',
        'Max': 'Tmax',
        'Min': 'Tmin',
    })
    for col in ['A', 'B', 'C', 'Tmin', 'Tmax']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['formula'] = df['formula'].astype(str).str.strip()
    df['composto'] = df['composto'].astype(str).str.strip()
    return df


def linha_antoine(df, formula):
    f = formula.upper().strip()

    # Correção da inconsistência da planilha: CO aparece com fórmula "CO2"
    if f == 'CO':
        m = df[df['composto'].str.lower().eq('carbon monoxide')]
        if m.empty:
            raise ValueError('CO não encontrado na base (procure por Carbon Monoxide).')
        return m.iloc[0]

    if f == 'CO2':
        m = df[df['composto'].str.lower().eq('carbon dioxide')]
        if m.empty:
            raise ValueError('CO2 não encontrado na base (procure por Carbon dioxide).')
        return m.iloc[0]

    m = df[df['formula'].str.upper().eq(f)]
    if m.empty:
        raise ValueError(f'{formula} não encontrado na base.')
    return m.iloc[0]


def antoine_bar(T_K, A, B, C):
    # ln(P_mmHg) = A - B/(T + C)
    P_mmHg = np.exp(A - B/(T_K + C))
    return P_mmHg * 0.001333223684  # mmHg -> bar


def wagner_generalizada_bar(T_K, Tc_K, Pc_Pa):
    # Forma generalizada usada no seu notebook
    Tr = T_K / Tc_K
    tau = 1.0 - Tr
    A = -7.85951783
    B = 1.84408259
    C = -11.7866497
    D = 22.6807411
    ln_Pr = (A*tau + B*tau**1.5 + C*tau**3 + D*tau**6) / Tr
    return (Pc_Pa * np.exp(ln_Pr)) / 1e5


def referencia_real_bar(T_K, nome_thermo):
    """
    Referência real/experimental aproximada com dados do pacote `thermo`:
    - abaixo da temperatura de triplo: SublimationPressure
    - acima: VaporPressure (HEOS_FIT)
    """
    chem = Chemical(nome_thermo)
    vp = chem.VaporPressure
    sp = chem.SublimationPressure
    T_trip = chem.Tt

    if 'Fit 2023' in sp.all_methods:
        sp_method = 'Fit 2023'
    elif 'LANDOLT' in sp.all_methods:
        sp_method = 'LANDOLT'
    else:
        sp_method = 'PSUB_CLAPEYRON'

    P = np.empty_like(T_K, dtype=float)
    for i, T in enumerate(T_K):
        if T < T_trip:
            P[i] = sp.calculate(float(T), method=sp_method)
        else:
            P[i] = vp.calculate(float(T), method='HEOS_FIT')

    return P / 1e5


def plot_composto(formula, df):
    row = linha_antoine(df, formula)
    A, B, C = float(row['A']), float(row['B']), float(row['C'])
    Tmin, Tmax = float(row['Tmin']), float(row['Tmax'])
    T_K = np.linspace(Tmin, Tmax, 250)

    cas = CAS_from_any(formula)
    tc, pc, w = Tc(cas), Pc(cas), omega(cas)

    P_ant = antoine_bar(T_K, A, B, C)
    P_lk = np.array([Lee_Kesler(float(T), tc, pc, w) for T in T_K]) / 1e5
    P_wag = wagner_generalizada_bar(T_K, tc, pc)

    nome_thermo = 'carbon monoxide' if formula == 'CO' else 'carbon dioxide'
    P_ref = referencia_real_bar(T_K, nome_thermo)

    eps = 1e-12
    err_ant = 100.0*(P_ant - P_ref)/(P_ref + eps)
    err_lk = 100.0*(P_lk - P_ref)/(P_ref + eps)
    err_wag = 100.0*(P_wag - P_ref)/(P_ref + eps)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    # Grafico 1: P vs T
    ax[0].plot(T_K, P_ref, 'k', lw=2.4, label='Dados reais/ref. (thermo)')
    ax[0].plot(T_K, P_ant, label='Antoine', lw=2)
    ax[0].plot(T_K, P_lk, '--', label='Lee-Kesler', lw=2)
    ax[0].plot(T_K, P_wag, ':', label='Wagner', lw=2)

    # Pontos experimentais VDI para CO (quando houver no intervalo)
    chem = Chemical(nome_thermo)
    if 'VDI_TABULAR' in chem.VaporPressure.all_methods:
        Ts_vdi, Ps_vdi = chem.VaporPressure.tabular_data['VDI_TABULAR']
        Ts_vdi = np.asarray(Ts_vdi)
        Ps_vdi_bar = np.asarray(Ps_vdi)/1e5
        m = (Ts_vdi >= Tmin) & (Ts_vdi <= Tmax)
        if np.any(m):
            ax[0].scatter(Ts_vdi[m], Ps_vdi_bar[m], s=22, color='tab:red', alpha=0.85,
                          label='Pontos VDI (exp.)')

    ax[0].set_title(f'{formula} - P vs T')
    ax[0].set_xlabel('Temperatura (K)')
    ax[0].set_ylabel('Pressão de vapor (bar)')
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    # Grafico 2: erro percentual
    ax[1].axhline(0.0, color='k', lw=1)
    ax[1].plot(T_K, err_ant, label='Erro Antoine (%)', lw=2)
    ax[1].plot(T_K, err_lk, '--', label='Erro Lee-Kesler (%)', lw=2)
    ax[1].plot(T_K, err_wag, ':', label='Erro Wagner (%)', lw=2)
    ax[1].set_title(f'{formula} - Erro percentual vs dados reais/ref.')
    ax[1].set_xlabel('Temperatura (K)')
    ax[1].set_ylabel('Erro (%)')
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    def mape(x):
        return float(np.nanmean(np.abs(x)))

    print(f'[{formula}] Faixa T (K): {Tmin:.2f} a {Tmax:.2f}')
    print(f'[{formula}] MAPE Antoine:     {mape(err_ant):.2f}%')
    print(f'[{formula}] MAPE Lee-Kesler: {mape(err_lk):.2f}%')
    print(f'[{formula}] MAPE Wagner:     {mape(err_wag):.2f}%')


def main():
    df = carregar_biblioteca(CSV_PATH)
    for comp in COMPOSTOS:
        plot_composto(comp, df)


if __name__ == '__main__':
    main()

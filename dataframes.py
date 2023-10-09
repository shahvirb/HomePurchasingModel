import pandas as pd
import numpy as np
import formulas


def mortgage_amortization(loan_value, apr, n, duration, down_pmt, start_month_index=0):
    """
    Calculate the amortization schedule for a mortgage loan and return it as a DataFrame.

    Parameters
    ----------
    loan_value : float
        The initial loan amount, not accounting for down payment.
    apr : float
        The annual percentage rate (as a decimal) for the loan.
    n : int
        The number of payments per year.
    duration : int
        The duration of the loan, in years.
    down_pmt : float
        The initial down payment made.
    start_month_index : int, optional, default=0
        The starting month index for the amortization schedule.

    Returns
    -------
    df : DataFrame
        A pandas DataFrame containing the amortization schedule, indexed by month. The DataFrame contains
        the following columns:
        - 'mortgage balance': Remaining balance of the mortgage at each payment period.
        - 'mortgage payment': The periodic mortgage payment amount.
        - 'payments': The cumulative sum of all payments made till the current period.
        - 'equity': The owner's equity in the property, considering the down payment and the part of the 
                    loan repaid.
        - 'principal paid': The part of the mortgage payment applied to the principal till the current 
                            period.
        - 'interest paid': The part of the mortgage payment applied to the interest till the current period.
    
    """
    
    t = np.arange(duration * n + 1)
    single_payment = formulas.payment(loan_value, apr, n, duration)
    payment = np.full(t.size, single_payment)
    payment[0] = 0
    payments = np.cumsum(payment)
    amount = np.zeros(t.size)
    net_amount = np.zeros(t.size)
    amount[0] = loan_value
    net_amount[0] = loan_value
    for i in t[1:]:
        amount[i] = formulas.compound_interest_amount(net_amount[i - 1], apr / n, 1, 1)
        net_amount[i] = amount[i] - payment[i]
    equity = loan_value - net_amount
    interest = payments - equity
    principal_paid = payments - interest
    df = pd.DataFrame(
        {
            'month': t + start_month_index,
            'mortgage balance': net_amount,
            # 'amount owed': amount,
            'mortgage payment': payment,
            'payments': payments,
            'equity': equity + down_pmt,
            'principal paid': principal_paid,
            'interest paid': interest,
        }
    )
    df = df.set_index('month')
    return df


def refi_mortgage_amortization(loan_amount, n, loan_duration, down_paymt, change_period_years, periodic_interest_rates):
    ld = loan_duration
    refi_period = change_period_years * n
    first_row = True
    equity = down_paymt
    loan_value = loan_amount - down_paymt
    frames = []
    for apr in periodic_interest_rates:
        print(f'ma_df for apr = {apr}, duration = {ld}')      
        start_index = 0 if not frames else frames[-1].index[-1]
        ma_df = mortgage_amortization(loan_value, apr, n, ld, equity, start_index)
        start_slice = 1
        if first_row:
            start_slice = 0
            first_row = False
        ma_df = ma_df[start_slice:refi_period+1]
        ma_df['interest rate %'] = apr * 100
        frames.append(ma_df)
        
        ld -= change_period_years
        equity += ma_df['principal paid'].iat[-1]
        loan_value -= ma_df['principal paid'].iat[-1]
    
    return pd.concat(frames)
  

def property_tax_amortization(appraisal_val, tax_rate, duration, appraisal_growth_rate):
    """
    Calculate the property tax amortization over a specified duration considering appraisal growth.

    Parameters
    ----------
    appraisal_val : float
        The initial appraised value of the property.
    tax_rate : float
        The property tax rate (as a decimal) applicable to the appraised value.
    payments_n : int
        The number of tax payments per year. [Note: This parameter is defined but not used in the function.]
    duration : int
        The duration (in years) over which the amortization is calculated.
    appraisal_growth_rate : float
        The annual growth rate (as a decimal) of the property's appraised value.

    Returns
    -------
    df : DataFrame
        A pandas DataFrame, indexed by year, containing the following columns:
        - 'appraisal value': The appraised property value for each year.
        - 'annual tax owed': The total tax owed for each year.
        - 'monthly tax payment': The monthly portion of the tax owed for each year.
    """
    t = np.arange(duration)

    vectorized_tax_growth = np.vectorize(
        lambda x: formulas.compound_interest_amount(appraisal_val, appraisal_growth_rate, 1, x))
    # avalue = np.full(t.size, appraisal_val)
    avalue = vectorized_tax_growth(t)
    annual_tax_owed = avalue * tax_rate
    monthly_tax_owed = annual_tax_owed / 12

    df = pd.DataFrame(
        {
            'year': t + 1,
            'appraisal value': avalue,
            'annual tax owed': annual_tax_owed,
            'monthly tax payment': monthly_tax_owed,
        }
    )
    return df.set_index('year')


def amortization_summary(ma_df, tax_df, monthly_income, monthly_additional_costs):
    """
    Generate a detailed mortgage amortization summary, incorporating property tax, additional costs, and 
    various computed metrics.

    Parameters
    ----------
    ma_df : DataFrame
        A DataFrame containing the basic mortgage amortization schedule. Expected columns include 
        'mortgage payment', 'payments', 'principal paid', 'interest paid', and should be indexed by month.
    tax_df : DataFrame
        A DataFrame containing property tax information, specifically 'monthly tax payment' and 
        'appraisal value', indexed by year.
    monthly_income : float
        The borrower's monthly income.
    monthly_additional_costs : float
        Additional costs per month associated with the property, e.g., insurance, maintenance, etc.

    Returns
    -------
    df : DataFrame
        A DataFrame providing a comprehensive mortgage amortization summary, which includes the original 
        mortgage details, property tax information, additional costs, and calculated metrics like 'debt to 
        income ratio'. The DataFrame retains a monthly index.
    """
    
    df = ma_df.copy()
    years = (df.index - 1) / 12 + 1
    df['year'] = years.astype(int)
    df.at[0, 'year'] = 0
    print(df.keys())
    df = df.merge(tax_df[['monthly tax payment', 'appraisal value']], how='left', left_on='year', right_index=True,
                  copy=False)
    df['monthly payment'] = df['mortgage payment'] + df['monthly tax payment'] + monthly_additional_costs
    df['payments'] = np.cumsum(df['monthly payment'])
    df['tax paid'] = np.cumsum(df['monthly tax payment'])
    df['mortgage payment principal'] = df['principal paid'] / (df['principal paid'] + df['interest paid']) * \
                                       df['mortgage payment']
    df['mortgage payment interest'] = df['interest paid'] / (df['principal paid'] + df['interest paid']) * \
                                      df['mortgage payment']
    df['monthly income'] = monthly_income
    df['debt to income ratio'] = df['monthly payment'] / df['monthly income'] * 100
    # df = df.drop(columns=['year'])
    return df

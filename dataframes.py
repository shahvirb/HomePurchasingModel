import pandas as pd
import numpy as np
import formulas


def mortgage_amortization(principal, apr, n, duration, down_pmt):
    t = np.arange(duration * n + 1)
    single_payment = formulas.payment(principal, apr, n, duration)
    payment = np.full(t.size, single_payment)
    payment[0] = 0
    payments = np.cumsum(payment)
    amount = np.zeros(t.size)
    net_amount = np.zeros(t.size)
    amount[0] = principal
    net_amount[0] = principal
    for i in t[1:]:
        amount[i] = formulas.compound_interest_amount(net_amount[i - 1], apr / n, 1, 1)
        net_amount[i] = amount[i] - payment[i]
    equity = principal - net_amount
    interest = payments - equity
    principal_paid = payments - interest
    df = pd.DataFrame(
        {
            'month': t,
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


def property_tax_amortization(appraisal_val, tax_rate, payments_n, duration, appraisal_growth_rate):
    t = np.arange(duration)

    vectorized_tax_growth = np.vectorize(
        lambda x: formulas.compound_interest_amount(appraisal_val, appraisal_growth_rate, 1, x))
    # avalue = np.full(t.size, appraisal_val)
    avalue = vectorized_tax_growth(t)
    annual_tax_owed = avalue * tax_rate
    monthly_tax_owed = annual_tax_owed / 12

    df = pd.DataFrame(
        {
            'year': t,
            'appraisal value': avalue,
            'annual tax owed': annual_tax_owed,
            'monthly tax payment': monthly_tax_owed,
        }
    )
    return df.set_index('year')

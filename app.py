from lxml import html
import requests
import pandas as pd
import yfinance
import json
import argparse
import streamlit as st
import numpy as np
from collections import OrderedDict

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

import warnings
warnings.filterwarnings('ignore')

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made ",
        " with ❤️ by ",
        link("https://prateekmalhotra.me", "Prateek Malhotra"),
        br(),
        br(),
        "I enjoy making these tools for free but some ", 
        link("https://buymeacoffee.com/prateekm08", "pocket money"),
        " would sure be appreciated!"
    ]
    layout(*myargs)

def parse(ticker):
    url = "https://stockanalysis.com/stocks/{}/financials/cash-flow-statement".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)

    # Cash Flows

    op_fcfs = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Operating Cash Flow")]]')[0].xpath('.//td/span/text()')[1:]
    capexs = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Capital Expenditures")]]')[0].xpath('.//td/span/text()')[1:]
    dbt = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Debt Issued / Paid")]]')[0].xpath('.//td/span/text()')[1:]
    net_income = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Net Income")]]')[0].xpath('.//td/span/text()')[1:]

    op_fcfs = [float(x.replace(',', '')) for x in op_fcfs]
    capexs = [float(x.replace(',', '')) for x in capexs]
    dbt = [float(x.replace(',', '')) for x in dbt]
    net_income = [float(x.replace(',', '')) for x in net_income]

    fcfs_equity = list(np.array(op_fcfs) + np.array(capexs)) # + np.array(dbt) (difficult to predict when company will borrow money)
    
    # Revenues

    url = "https://stockanalysis.com/stocks/{}/financials".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)

    revenues = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Revenue")]]')[0].xpath('.//td/span/text()')[1:]
    revenues = [float(x.replace(',', '')) for x in revenues]

    # Debt

    url = "https://stockanalysis.com/stocks/{}/financials/balance-sheet".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)

    net_debt = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Net Cash / Debt")]]')[0].xpath('.//td/span/text()')[1:]
    net_debt = [float(x.replace(',', '')) for x in net_debt][0]

    url = "https://finance.yahoo.com/quote/{}/analysis?p={}".format(ticker, ticker)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
    parser = html.fromstring(response.content)
    tables = parser.xpath('//table')

    for t in tables:
        if t.xpath("thead//tr//th/span/text()")[0] == 'Revenue Estimate':
            vals = t.xpath("tbody//tr//td/span/text()")
            ind = vals.index("Low Estimate")
            res = [vals[ind - 2], vals[ind - 1]]
            
            for i in range(len(res)):
                if 'B' in res[i]:
                    res[i] = float(res[i].replace('B', ''))  * 1000
                elif 'M'  in res[i]:
                    res[i] = float(res[i].replace('M', ''))
        
            break

    ge = tables[-1].xpath("tbody//tr")

    for row in ge:
        label = row.xpath("td/span/text()")[0]

        if 'Next 5 Years' in label:
            try:
                ge = float(row.xpath("td/text()")[0].replace('%', ''))
            except:
                ge = []
            break

    url = "https://stockanalysis.com/stocks/{}/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    shares = parser.xpath('//div[@class="order-1 flex flex-row gap-4"]//table//tbody//tr[td/text()[contains(., "Shares Out")]]')

    shares = shares[0].xpath('td/text()')[1]
    factor = 1000 if 'B' in shares else 1 
    shares = float(shares.replace('B', '').replace('M', '')) * factor

    url = "https://stockanalysis.com/stocks/{}/financials/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    eps = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "EPS (Diluted)")]]')[0].xpath('.//td/span/text()')[1:]
    eps = float(eps[0].replace(",", ""))

    try:
        market_price = float(parser.xpath('//div[@class="price-ext"]/text()')[0].replace('$', '').replace(',', ''))
    except:
        market_price = round(yfinance.Ticker(ticker).history().tail(1).Close.iloc[0], 2)

    return {'fcf': fcfs_equity, 'op_fcfs': op_fcfs, 'capexs': capexs, 'dbt': dbt, 'ni': net_income, 'revenues': revenues, 'nd': net_debt, 'res': res, 'ge': ge, 'yr': 5, 'dr': 10, 'pr': 2.5, 'shares': shares, 'eps': eps, 'mp': market_price}

def dcf(data):
    fcf_ni_ratio = np.round(np.array(data['fcf'][:3]) / np.array(data['ni'][:3]), 2) * 100
    fcf_ni_ratio_deviation = np.std(fcf_ni_ratio)
    fcf_ni_correlation = np.corrcoef(data['fcf'][:3], data['ni'][:3]) # Direction

    if data['fcf'][0] < 0:
        st.warning("Free Cash Flow is negative")

    print(fcf_ni_ratio_deviation)
    if fcf_ni_ratio_deviation > 40:
        st.warning("Magnitude of change in FCF is not the same as that of Net Income. Fair Value might need more analysis from your side.")

    if fcf_ni_correlation[0][1] < 0.9:
        st.warning("FCF is not inline with profitability. Fair Value might not be too reliable.")

    ni_margins = np.round(np.array(data['ni']) / np.array(data['revenues']), 2) * 100

    forecast = [data['fcf'][1], data['fcf'][0]]
    rev_forecast_df = [data['revenues'][0], data['res'][0], data['res'][1]]
    net_income = [data['ni'][-1], data['ni'][0]]

    if data['ge'] == []:
    	raise ValueError("No growth rate available from Yahoo Finance")

    for i in range(1, data['yr']):
        forecast.append(round(forecast[-1] + (data['ge'] / 100) * forecast[-1], 2))
        net_income.append(round(net_income[-1] + (data['ge'] / 100) * net_income[-1], 2))

    for i in range(1, data['yr'] - 1):
        rev_forecast_df.append(round(rev_forecast_df[-1] + (data['ge'] / 100) * rev_forecast_df[-1], 2))

    rev_forecast_df = pd.DataFrame(np.array((rev_forecast_df, net_income)).T, columns=['Revenue estimate', 'Net Income'])

    forecast.append(round(forecast[-1] * (1 + (data['pr'] / 100)) / (data['dr'] / 100 - data['pr'] / 100), 2)) #terminal value
    discount_factors = [1 / (1 + (data['dr'] / 100))**(i + 1) for i in range(len(forecast) - 1)]

    pvs = [round(f * d, 2) for f, d in zip(forecast[:-1], discount_factors)]
    pvs.append(round(discount_factors[-1] * forecast[-1], 2)) # discounted terminal value
    
    cash_array = np.array((forecast[:-1], pvs[:-1])).T
    forecast_df = pd.DataFrame(cash_array, columns=['Forecasted Cash Flows', 'PV of Cash Flows'])

    st.markdown("### _Cash Flows_")
    st.line_chart(data=forecast_df)

    dcf = sum(pvs) + data['nd']
    fv = round(dcf / data['shares'], 2)

    st.markdown("### _Income_ ")
    st.line_chart(data=rev_forecast_df)

    return fv

def reverse_dcf(data):
    pass

def graham(data):
    if data['eps'] > 0:
        expected_value = round(data['eps'] * (8.5 + 2 * (data['ge'])), 2)
        
        try:
            ge_priced_in = round((data['mp'] / data['eps'] - 8.5) / 2, 2)
        except:
            ge_priced_in = "N/A"

        st.write("Expected value based on growth rate: {}".format(expected_value))
        st.write("Growth rate priced in for next 7-10 years: {}\n".format(ge_priced_in))
    else:
        st.write("Not applicable since EPS is negative.")

if __name__ == "__main__":
    st.title("Intrinsic Value Calculator")
    
    ticker_input_container = st.empty()
    ticker = ticker_input_container.text_input("Ticker", max_chars=7, help="Inset ticker symbol for the company you wish to value", placeholder="AAPL", value="AAPL")
    
    dr_input_container = st.empty()
    discount_rate = dr_input_container.number_input("Discount Rate (%)", min_value=0.0, max_value=100.0, format="%f", value=7.5, help="Insert discount rate (also called required rate of return)")
    
    ge_input_container = st.empty()
    growth_estimate = ge_input_container.number_input("Growth Estimate (%)", min_value=-100.0, max_value=100.0, format="%f", help="Estimated yoy growth rate. If left to -100, it will fetch from Yahoo Finance")
    
    tr_input_container = st.empty()
    terminal_rate = tr_input_container.number_input("Terminal Rate (%)", min_value=0.0, max_value=100.0, format="%f", help="Terminal growth rate.", value=2.5)
    
    pd_input_container = st.empty()
    period = pd_input_container.number_input("Time period (yrs)", min_value=0, max_value=25, format="%d", help="Time period for growth", value=5)

    fcf_choice_container = st.empty()
    fcf_choice = fcf_choice_container.selectbox("Initial FCF choice", ("Most Recent Free Cash Flow", "Average last 3 years", "Custom"), help="Chooses most recent FCF by default")

    yf_flag = True
    if growth_estimate != -100.0:
        yf_flag = False

    st.text("")
    compute_container = st.empty()

    if fcf_choice == "Custom":
        fcf = fcf_choice_container.number_input("Free Cash Flow (Custom) in millions", format="%f")

    if compute_container.button("Compute DCF (basic) valuation"):
        
        ticker_input_container.empty()
        dr_input_container.empty()
        ge_input_container.empty()
        tr_input_container.empty()
        pd_input_container.empty()
        compute_container.empty()
        fcf_choice_container.empty()

        data = parse(ticker)
        if fcf_choice == "Custom":
            data['fcf'][0] = fcf

        if fcf_choice == "Average last 3 years":
            data['fcf'][0] = np.mean(data['fcf'][:3])

        if period is not None:
            data['yr'] = int(period)
        if yf_flag == False:
            data['ge'] = float(growth_estimate)
        if discount_rate is not None:
            data['dr'] = float(discount_rate)
        if terminal_rate is not None:
            data['pr'] = float(terminal_rate)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("", "")
            st.metric(f"Ticker", ticker, delta=None, delta_color="normal")

        with col2:
            st.metric(f"Market Price", data['mp'], delta=None, delta_color="normal")

            if yf_flag:
                st.metric("Growth estimate (Yahoo Finance)", str(data['ge']) + " %", delta=None, delta_color="normal")
            else:
                st.metric("Growth estimate", str(data['ge']) + " %", delta=None, delta_color="normal")

            st.metric("Discount Rate", str(data['dr']) + " %", delta=None, delta_color="normal")

        
        with col3:
            st.metric("EPS", data['eps'], delta=None, delta_color="normal")
            st.metric("Term", str(data['yr']) + " years", delta=None, delta_color="normal")
            st.metric("Perpetual Rate", str(data['pr']) + " %", delta=None, delta_color="normal")
        

        fv = dcf(data)

        with col4:
            st.metric("", "")
            st.metric("Fair Value", fv)

        footer()

        # st.write("=" * 80)
        # st.write("Graham style valuation basic (Page 295, The Intelligent Investor)")
        # st.write("=" * 80 + "\n")

        # graham(data)

import pandas as pd
import numpy as np
from datetime import datetime


def get_data(file_path):
    return pd.read_csv('armut_data.csv')

def clean_data(df):
    return df.dropna()

def transform_data(df):
    return df

def save_data(df, file_path):
    df.to_csv('cleaned_armut_data.csv', index=False)

def combine_columns(df, col1, col2, col3=None):
    df[col1] = df[col1].astype(str)
    df[col2] = df[col2].astype(str)
    if col3 is None:
        col3 = col1 + '_' + col2
    df[col3] = df[col1] +'_'+ df[col2]
    return df


def extract_month_day(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])

    df['New_Date'] = df[column_name].dt.strftime('%Y-%m')

    return df

def main():
    df = get_data('armut_data.csv')
    df = clean_data(df)
    df = transform_data(df)
    df = combine_columns(df, 'ServiceId', 'CategoryId','Service')
    df = extract_month_day(df,"CreateDate")
    df = combine_columns(df, 'UserId', 'New_Date', 'SepetId')
    save_data(df, 'cleaned_armut_data.csv')

    return df


if __name__ == '__main__':
    df_to_pivot= main()

df_pivoted = df_to_pivot.pivot_table(index='SepetId', columns='Service',aggfunc='size', fill_value=0)

print(df_pivoted)

#Association Rules Analysis (Apriori Algorithm)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

frequent_itemsets = apriori(df_pivoted.astype('bool'), min_support=0.01, use_colnames=True)  # Lower min_support if you do not get any results
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)  # Adjusted min_threshold if necessary

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

print(rules.head(),"\n")  # Check if any rules are now found


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values('lift', ascending=False)

    recommendation_list = []

    for i, row in sorted_rules.iterrows():
        if product_id in row['antecedents']:
            recommendation_list.extend(list(row['consequents']))
        if len(recommendation_list) >= rec_count:
            break

    return recommendation_list[:rec_count]


print(arl_recommender(rules,"2_0",2))
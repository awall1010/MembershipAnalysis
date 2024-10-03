import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot styles
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Title for the app
st.title("Rush Soccer Club Membership Analysis")

st.markdown("""
This report provides an analysis of how membership per club has changed from Fall 2023 to Fall 2024.
Upload the membership data files to get started.
""")

# -----------------------------------------
# File Upload Section
# -----------------------------------------

st.header("Upload Membership Data Files")

fall_23_file = st.file_uploader("Upload Membership_Numbers_Fees_Fall_23.csv", type="csv", key="fall_23")
spring_24_file = st.file_uploader("Upload Membership_Numbers_Fees_Spring_24.csv", type="csv", key="spring_24")
fall_24_file = st.file_uploader("Upload Membership_Numbers_Fees_Fall_24.csv", type="csv", key="fall_24")

if fall_23_file is not None and spring_24_file is not None and fall_24_file is not None:
    # Read the CSV files
    df_fall_23 = pd.read_csv(fall_23_file)
    df_spring_24 = pd.read_csv(spring_24_file)
    df_fall_24 = pd.read_csv(fall_24_file)

    # -----------------------------------------
    # Data Preparation
    # -----------------------------------------

    # Define the columns to be renamed
    columns_to_rename = ['Dev #s', 'Comp Under HS #s', 'Comp HS #s', 'Total #']

    # Function to rename columns by adding the season prefix
    def rename_columns(df, season_prefix):
        new_columns = {}
        for col in columns_to_rename:
            new_columns[col] = f"{season_prefix} {col}"
        df.rename(columns=new_columns, inplace=True)
        return df

    # Rename columns for each DataFrame
    df_fall_23 = rename_columns(df_fall_23, 'F23')
    df_spring_24 = rename_columns(df_spring_24, 'S24')
    df_fall_24 = rename_columns(df_fall_24, 'F24')

    # Update the values for the club "Northeast Combination" including the "F23 Total #" column
    df_fall_23.loc[df_fall_23['Club'] == 'Northeast Combination',
                   ['F23 Dev #s', 'F23 Comp Under HS #s', 'F23 Comp HS #s', 'F23 Total #']] = [76, 1205, 0, 1281]

    # Merge the data on 'Club'
    df_merged = df_fall_23.merge(df_spring_24, on='Club', how='outer')
    df_merged = df_merged.merge(df_fall_24, on='Club', how='outer')

    # Fill missing values with 0 for better analysis
    df_merged.fillna(0, inplace=True)

    # Exclude 'New Mexico' from comparisons
    df_comp = df_merged[df_merged['Club'] != 'New Mexico'].copy()

    # Calculate percentage change from Fall 2023 to Fall 2024
    df_comp['F23 to F24 Change (%)'] = ((df_comp['F24 Total #'] - df_comp['F23 Total #']) / df_comp['F23 Total #']) * 100

    # Replace infinite values with NaN
    df_comp.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with 0
    df_comp.fillna(0, inplace=True)

    # Exclude clubs where 'F23 Total #' is 0 to avoid division by zero
    df_growth = df_comp[df_comp['F23 Total #'] > 0].copy()

    # -----------------------------------------
    # Additional Calculations
    # -----------------------------------------

    # Calculate total growth from Fall 2023 to Fall 2024
    df_comp['Total Growth F23 to F24'] = df_comp['F24 Total #'] - df_comp['F23 Total #']

    # Calculate growth for Dev #s from Fall 23 to Fall 24 and from Fall 23 to Spring 24
    df_comp['Dev Growth F23 to F24'] = df_comp['F24 Dev #s'] - df_comp['F23 Dev #s']
    df_comp['Dev Growth F23 to S24'] = df_comp['S24 Dev #s'] - df_comp['F23 Dev #s']

    # Calculate growth for Comp Under HS #s from Fall 23 to Fall 24 and from Fall 23 to Spring 24
    df_comp['Comp Under HS Growth F23 to F24'] = df_comp['F24 Comp Under HS #s'] - df_comp['F23 Comp Under HS #s']
    df_comp['Comp Under HS Growth F23 to S24'] = df_comp['S24 Comp Under HS #s'] - df_comp['F23 Comp Under HS #s']

    # -----------------------------------------
    # Overall Membership Summary
    # -----------------------------------------

    st.header("Overall Membership Summary")

    # Calculate total membership in Fall 2023, Spring 2024, and Fall 2024
    total_members_fall_2023 = df_merged['F23 Total #'].sum()
    total_members_spring_2024 = df_merged['S24 Total #'].sum()
    total_members_fall_2024 = df_merged['F24 Total #'].sum()

    # Overall change from Fall 2023 to Spring 2024
    overall_change_spring = ((total_members_spring_2024 - total_members_fall_2023) / total_members_fall_2023) * 100

    # Overall change from Fall 2023 to Fall 2024
    overall_change_fall = ((total_members_fall_2024 - total_members_fall_2023) / total_members_fall_2023) * 100

    st.write(f"Total Membership in Fall 2023: **{total_members_fall_2023}**")
    st.write(f"Total Membership in Spring 2024: **{total_members_spring_2024}**")
    st.write(f"Total Membership in Fall 2024: **{total_members_fall_2024}**")
    st.write(f"Overall Percentage Change from Fall 2023 to Spring 2024: **{overall_change_spring:.2f}%**")
    st.write(f"Overall Percentage Change from Fall 2023 to Fall 2024: **{overall_change_fall:.2f}%**")

    # -----------------------------------------
    # Summary Report
    # -----------------------------------------

    st.header("Summary Report")

    # Calculate median total growth
    median_total_growth = df_comp['Total Growth F23 to F24'].median()

    st.write(f"Total Clubs Analyzed: **{len(df_comp)}**")
    st.write(f"Median Total Growth F23 to F24: **{median_total_growth:.2f}**")

    # Prepare the dataframe to display
    df_growth_display = df_comp[['Club', 'F23 Total #', 'F24 Total #', 'Total Growth F23 to F24', 'F23 to F24 Change (%)']]

    # Sort by 'Total Growth F23 to F24' descending
    df_growth_display = df_growth_display.sort_values(by='Total Growth F23 to F24', ascending=False)

    st.write("Membership Growth from Fall 2023 to Fall 2024 for Each Club:")
    st.dataframe(df_growth_display)

    # Identify clubs that have not uploaded their numbers for Fall 2024
    clubs_missing_f24 = df_comp[df_comp['F24 Total #'] == 0]

    if not clubs_missing_f24.empty:
        st.write("The following clubs have not uploaded their membership numbers for Fall 2024:")
        st.write(clubs_missing_f24['Club'])
    else:
        st.write("All clubs have uploaded their membership numbers for Fall 2024.")

    # -----------------------------------------
    # Visualizations
    # -----------------------------------------

    # Let the user select the number of clubs to display
    num_clubs_to_display = st.slider('Select the number of clubs to display in the bar charts:', min_value=1, max_value=len(df_comp), value=20)

    # Bar Chart: Membership Growth for Clubs
    st.subheader("Membership Growth for Clubs (Fall 23 to Fall 24)")
    st.write(f"This bar chart shows the membership growth for the top {num_clubs_to_display} clubs from Fall 2023 to Fall 2024.")

    # Sort df_comp by 'Total Growth F23 to F24'
    df_comp_sorted = df_comp.sort_values(by='Total Growth F23 to F24', ascending=False)

    # Select the top clubs to display
    df_comp_display = df_comp_sorted.head(num_clubs_to_display)

    fig_growth, ax_growth = plt.subplots(figsize=(12, num_clubs_to_display * 0.5))
    sns.barplot(x='Total Growth F23 to F24', y='Club', data=df_comp_display, palette='viridis', ax=ax_growth)
    ax_growth.set_title(f'Membership Growth for Top {num_clubs_to_display} Clubs (Fall 23 to Fall 24)')
    ax_growth.set_xlabel('Total Growth F23 to F24')
    ax_growth.set_ylabel('Club')
    st.pyplot(fig_growth)

    # Bar Chart: Percentage Growth for Clubs
    st.subheader("Percentage Growth for Clubs (Fall 23 to Fall 24)")
    st.write(f"This bar chart shows the percentage growth in membership for the top {num_clubs_to_display} clubs from Fall 2023 to Fall 2024.")

    # Sort df_growth by 'F23 to F24 Change (%)' descending
    df_growth_sorted = df_growth.sort_values(by='F23 to F24 Change (%)', ascending=False)

    # Select the top clubs to display
    df_growth_display_top = df_growth_sorted.head(num_clubs_to_display)

    fig_percentage_growth, ax_percentage_growth = plt.subplots(figsize=(12, num_clubs_to_display * 0.5))
    sns.barplot(x='F23 to F24 Change (%)', y='Club', data=df_growth_display_top, palette='magma', ax=ax_percentage_growth)
    ax_percentage_growth.set_title(f'Percentage Growth for Top {num_clubs_to_display} Clubs (Fall 23 to Fall 24)')
    ax_percentage_growth.set_xlabel('Percentage Change in Membership')
    ax_percentage_growth.set_ylabel('Club')
    st.pyplot(fig_percentage_growth)

    # Histogram: Distribution of Total Growth from Fall 23 to Fall 24
    st.subheader("Distribution of Total Growth (Fall 23 to Fall 24)")
    st.write("This histogram displays the distribution of total membership growth from Fall 2023 to Fall 2024 across all clubs.")

    fig_hist_total_growth, ax_hist_total_growth = plt.subplots()
    sns.histplot(df_comp['Total Growth F23 to F24'], bins=30, kde=True, color='blue', ax=ax_hist_total_growth)
    ax_hist_total_growth.set_title('Distribution of Total Growth (Fall 23 to Fall 24)')
    ax_hist_total_growth.set_xlabel('Total Growth F23 to F24')
    ax_hist_total_growth.set_ylabel('Number of Clubs')
    st.pyplot(fig_hist_total_growth)

    # -----------------------------------------
    # Scatter Plot: Club Size vs. Percentage Change
    # -----------------------------------------

    # Filter out clubs with 100% decrease or no reported membership
    filtered_clubs = df_growth[
        (df_growth['F23 to F24 Change (%)'] > -100) & (df_growth['F23 to F24 Change (%)'] < 100) & (df_growth['F24 Total #'] > 0)
    ]
    filtered_clubs['Change Type'] = filtered_clubs['F23 to F24 Change (%)'].apply(lambda x: 'Increase' if x > 0 else 'Decrease')

    st.subheader("Club Size vs. Percentage Change (Fall 23 to Fall 24)")
    st.write("This scatter plot visualizes the relationship between club size in Fall 2023 and the percentage change in membership from Fall 2023 to Fall 2024.")

    # Scatter plot
    fig_scatter, ax_scatter = plt.subplots()
    sns.scatterplot(
        x='F23 Total #',
        y='F23 to F24 Change (%)',
        data=filtered_clubs,
        hue='Change Type',
        palette={'Increase': 'green', 'Decrease': 'red'},
        s=100,
        legend=True,
        ax=ax_scatter
    )
    ax_scatter.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax_scatter.set_title('Club Size in Fall 2023 vs. Percentage Change (Fall 23 to Fall 24)')
    ax_scatter.set_xlabel('Membership in Fall 2023')
    ax_scatter.set_ylabel('Percentage Change in Membership')
    st.pyplot(fig_scatter)

    # -----------------------------------------
    # Pie Chart of Clubs that Increased vs. Decreased
    # -----------------------------------------

    st.subheader("Percentage of Clubs that Increased or Decreased")
    st.write("This pie chart shows the percentage of clubs that experienced an increase or decrease in total membership from Fall 2023 to Fall 2024.")

    # Calculate the number of clubs that increased and decreased
    num_increased = len(df_growth[df_growth['F23 to F24 Change (%)'] > 0])
    num_decreased = len(df_growth[df_growth['F23 to F24 Change (%)'] < 0])
    total_clubs_analyzed = num_increased + num_decreased

    # Prepare data for the pie chart
    labels = ['Increased', 'Decreased']
    sizes = [num_increased, num_decreased]
    colors = ['green', 'red']

    # Plot the pie chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax_pie.set_title('Percentage of Clubs that Increased or Decreased (Fall 23 to Fall 24)')
    st.pyplot(fig_pie)

    # -----------------------------------------
    # Distribution of Percentage Changes
    # -----------------------------------------

    st.subheader("Distribution of Percentage Change in Membership")
    st.write("This histogram displays the distribution of percentage changes in club memberships from Fall 2023 to Fall 2024.")

    # Plot the distribution of percentage changes
    fig_hist_percentage, ax_hist_percentage = plt.subplots()
    sns.histplot(df_growth['F23 to F24 Change (%)'], bins=30, kde=True, color='blue', ax=ax_hist_percentage)
    ax_hist_percentage.set_title('Distribution of Percentage Change in Membership (Fall 23 to Fall 24)')
    ax_hist_percentage.set_xlabel('Percentage Change in Membership')
    ax_hist_percentage.set_ylabel('Number of Clubs')
    st.pyplot(fig_hist_percentage)

    # -----------------------------------------
    # Average Membership Across All Clubs
    # -----------------------------------------

    average_f23 = df_merged['F23 Total #'].mean()
    average_s24 = df_merged['S24 Total #'].mean()
    average_f24 = df_merged['F24 Total #'].mean()

    st.subheader("Average Membership Across All Clubs")
    st.write("This section shows the average membership across all clubs for each season.")

    st.write(f"Average Membership in Fall 2023: **{average_f23:.2f}**")
    st.write(f"Average Membership in Spring 2024: **{average_s24:.2f}**")
    st.write(f"Average Membership in Fall 2024: **{average_f24:.2f}**")

    # -----------------------------------------
    # Data Preparation for Spring 24 to Fall 24 Growth Analysis
    # -----------------------------------------

    # Calculate growth from Spring 24 to Fall 24
    df_comp['Dev Growth S24 to F24'] = df_comp['F24 Dev #s'] - df_comp['S24 Dev #s']
    df_comp['Comp Under HS Growth S24 to F24'] = df_comp['F24 Comp Under HS #s'] - df_comp['S24 Comp Under HS #s']

    # Filter to keep only clubs with positive growth in Dev #s from Spring 24 to Fall 24
    positive_dev_growth_spring_fall = df_comp[(df_comp['F24 Dev #s'] > df_comp['S24 Dev #s'])].copy()

    # Calculate the total number of players added in Dev #s from Spring 24 to Fall 24
    positive_dev_growth_spring_fall['Total Dev #s Added'] = positive_dev_growth_spring_fall['Dev Growth S24 to F24']

    # Check if the DataFrame is empty
    if positive_dev_growth_spring_fall.empty:
        st.warning("No clubs have positive growth in Development numbers from Spring 2024 to Fall 2024.")
    else:
        # Let the user select the number of clubs to display
        num_dev_clubs = st.slider('Select the number of clubs to display for Dev #s growth:', min_value=1, max_value=len(positive_dev_growth_spring_fall), value=10)

        # Get the top clubs with the highest growth in 'Dev #s'
        top_dev_growth_s24_f24 = positive_dev_growth_spring_fall.nlargest(num_dev_clubs, 'Total Dev #s Added')

        # Plotting code
        st.subheader(f"Top {num_dev_clubs} Clubs with Most Growth in Dev #s (Spring 24 to Fall 24)")
        st.write(f"This bar chart shows the top {num_dev_clubs} clubs with the highest growth in Development numbers from Spring 2024 to Fall 2024, including start and end values.")

        fig_dev_growth_s24_f24, ax_dev_growth_s24_f24 = plt.subplots(figsize=(12, num_dev_clubs * 0.5))
        sns.barplot(x='Total Dev #s Added', y='Club', data=top_dev_growth_s24_f24, palette='viridis', ax=ax_dev_growth_s24_f24)

        # Add annotations for start and end values
        for index, value in enumerate(top_dev_growth_s24_f24['Total Dev #s Added']):
            start_value = top_dev_growth_s24_f24.iloc[index]['S24 Dev #s']
            end_value = top_dev_growth_s24_f24.iloc[index]['F24 Dev #s']
            ax_dev_growth_s24_f24.text(value + 1, index, f'From {start_value} to {end_value}', va='center', fontsize=9, color='black')

        ax_dev_growth_s24_f24.set_title(f'Top {num_dev_clubs} Clubs with Most Growth in Dev #s (Spring 24 to Fall 24)')
        ax_dev_growth_s24_f24.set_xlabel('Total Dev #s Added')
        ax_dev_growth_s24_f24.set_ylabel('Club')
        st.pyplot(fig_dev_growth_s24_f24)

    # Similar checks and plotting for Comp Under HS #s
    positive_comp_under_hs_growth_spring_fall = df_comp[(df_comp['F24 Comp Under HS #s'] > df_comp['S24 Comp Under HS #s'])].copy()
    positive_comp_under_hs_growth_spring_fall['Total Comp Under HS #s Added'] = positive_comp_under_hs_growth_spring_fall['Comp Under HS Growth S24 to F24']

    if positive_comp_under_hs_growth_spring_fall.empty:
        st.warning("No clubs have positive growth in Competitive Under HS numbers from Spring 2024 to Fall 2024.")
    else:
        num_comp_clubs = st.slider('Select the number of clubs to display for Comp Under HS #s growth:', min_value=1, max_value=len(positive_comp_under_hs_growth_spring_fall), value=10)

        top_comp_under_hs_growth_s24_f24 = positive_comp_under_hs_growth_spring_fall.nlargest(num_comp_clubs, 'Total Comp Under HS #s Added')

        st.subheader(f"Top {num_comp_clubs} Clubs with Most Growth in Comp Under HS #s (Spring 24 to Fall 24)")
        st.write(f"This bar chart shows the top {num_comp_clubs} clubs with the highest growth in Competitive Under High School numbers from Spring 2024 to Fall 2024, including start and end values.")

        fig_comp_under_hs_growth_s24_f24, ax_comp_under_hs_growth_s24_f24 = plt.subplots(figsize=(12, num_comp_clubs * 0.5))
        sns.barplot(x='Total Comp Under HS #s Added', y='Club', data=top_comp_under_hs_growth_s24_f24, palette='magma', ax=ax_comp_under_hs_growth_s24_f24)

        # Add annotations for start and end values
        for index, value in enumerate(top_comp_under_hs_growth_s24_f24['Total Comp Under HS #s Added']):
            start_value = top_comp_under_hs_growth_s24_f24.iloc[index]['S24 Comp Under HS #s']
            end_value = top_comp_under_hs_growth_s24_f24.iloc[index]['F24 Comp Under HS #s']
            ax_comp_under_hs_growth_s24_f24.text(value + 1, index, f'From {start_value} to {end_value}', va='center', fontsize=9, color='black')

        ax_comp_under_hs_growth_s24_f24.set_title(f'Top {num_comp_clubs} Clubs with Most Growth in Comp Under HS #s (Spring 24 to Fall 24)')
        ax_comp_under_hs_growth_s24_f24.set_xlabel('Total Comp Under HS #s Added')
        ax_comp_under_hs_growth_s24_f24.set_ylabel('Club')
        st.pyplot(fig_comp_under_hs_growth_s24_f24)

    # -----------------------------------------
    # Clubs with Consistent Growth
    # -----------------------------------------

    # Calculate growth between seasons
    df_comp['Growth F23 to S24'] = df_comp['S24 Total #'] - df_comp['F23 Total #']
    df_comp['Growth S24 to F24'] = df_comp['F24 Total #'] - df_comp['S24 Total #']

    # Identify clubs with consistent growth in both periods
    consistent_growth = df_comp[(df_comp['Growth F23 to S24'] > 0) & (df_comp['Growth S24 to F24'] > 0)]

    st.subheader("Clubs with Consistent Growth")
    st.write("These clubs showed consistent growth from Fall 2023 to Spring 2024 and from Spring 2024 to Fall 2024.")

    st.write(consistent_growth[['Club', 'Growth F23 to S24', 'Growth S24 to F24']])

    # Plot consistent growth
    if not consistent_growth.empty:
        num_consistent_clubs = st.slider('Select the number of clubs to display for consistent growth:', min_value=1, max_value=len(consistent_growth), value=10)

        fig_consistent_growth, ax_consistent_growth = plt.subplots(figsize=(12, num_consistent_clubs * 0.5))
        top_consistent_growth = consistent_growth.nlargest(num_consistent_clubs, 'Growth S24 to F24')

        sns.barplot(x='Growth S24 to F24', y='Club', data=top_consistent_growth, palette='viridis', ax=ax_consistent_growth)
        ax_consistent_growth.set_title(f'Top {num_consistent_clubs} Clubs with Consistent Growth (Spring 24 to Fall 24)')
        ax_consistent_growth.set_xlabel('Growth (S24 to F24)')
        ax_consistent_growth.set_ylabel('Club')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_consistent_growth)
    else:
        st.write("No clubs have consistent growth across both periods.")

    # -----------------------------------------
    # Growth by Club Size Category
    # -----------------------------------------

    # Categorize clubs based on Fall 2023 total membership size
    bins = [0, 500, 1000, 1750, float('inf')]
    labels = ['0-500', '500-1000', '1000-1750', '1750+']
    df_comp['Size Category'] = pd.cut(df_comp['F23 Total #'], bins=bins, labels=labels, right=False)

    # Calculate average growth by size category
    avg_growth_by_category = df_comp.groupby('Size Category')['Total Growth F23 to F24'].mean().reset_index()

    # Get counts per size category
    counts_per_category = df_comp['Size Category'].value_counts().reset_index()
    counts_per_category.columns = ['Size Category', 'Club Count']

    # Merge counts with avg_growth_by_category
    avg_growth_by_category = avg_growth_by_category.merge(counts_per_category, on='Size Category')

    st.subheader("Average Growth by Club Size Category (Fall 23 to Fall 24)")
    st.write("This bar chart shows the average membership growth from Fall 2023 to Fall 2024 for clubs categorized by their size in Fall 2023.")

    st.write("Club Counts per Size Category:")
    st.write(avg_growth_by_category[['Size Category', 'Club Count']])

    fig_growth_by_size_category, ax_growth_by_size_category = plt.subplots()
    sns.barplot(x='Size Category', y='Total Growth F23 to F24', data=avg_growth_by_category, palette='magma', ax=ax_growth_by_size_category)
    ax_growth_by_size_category.set_title('Average Growth by Club Size Category (Fall 23 to Fall 24)')
    ax_growth_by_size_category.set_xlabel('Size Category')
    ax_growth_by_size_category.set_ylabel('Average Growth (F23 to F24)')
    st.pyplot(fig_growth_by_size_category)

    # -----------------------------------------
    # Contribution of Player Categories to Total Growth
    # -----------------------------------------

    # Calculate growth for each category from Fall 23 to Fall 24
    df_comp['Dev Growth F23 to F24'] = df_comp['F24 Dev #s'] - df_comp['F23 Dev #s']
    df_comp['Comp Under HS Growth F23 to F24'] = df_comp['F24 Comp Under HS #s'] - df_comp['F23 Comp Under HS #s']

    # Plot growth contribution by different player categories
    categories = ['Dev Growth F23 to F24', 'Comp Under HS Growth F23 to F24']
    growth_contributions = df_comp[categories].sum().reset_index()
    growth_contributions.columns = ['Category', 'Total Growth']

    st.subheader("Contribution of Player Categories to Total Growth (Fall 23 to Fall 24)")
    st.write("This bar chart shows how different player categories contributed to the total membership growth from Fall 2023 to Fall 2024.")

    fig_growth_contributions, ax_growth_contributions = plt.subplots()
    sns.barplot(x='Category', y='Total Growth', data=growth_contributions, palette='viridis', ax=ax_growth_contributions)
    ax_growth_contributions.set_title('Contribution of Player Categories to Total Growth (Fall 23 to Fall 24)')
    ax_growth_contributions.set_xlabel('Player Category')
    ax_growth_contributions.set_ylabel('Total Growth')
    st.pyplot(fig_growth_contributions)

    # -----------------------------------------
    # Percentage Growth by Club Size Category
    # -----------------------------------------

    # Calculate percentage growth from Fall 23 to Fall 24
    df_comp['Percentage Growth F23 to F24'] = ((df_comp['F24 Total #'] - df_comp['F23 Total #']) / df_comp['F23 Total #']) * 100

    st.subheader("Percentage Growth by Club Size Category (Fall 23 to Fall 24)")
    st.write("This box plot shows the distribution of percentage membership growth from Fall 2023 to Fall 2024 for clubs in different size categories.")

    fig_percentage_growth_by_size, ax_percentage_growth_by_size = plt.subplots()
    sns.boxplot(x='Size Category', y='Percentage Growth F23 to F24', data=df_comp, palette='magma', ax=ax_percentage_growth_by_size)
    ax_percentage_growth_by_size.set_title('Percentage Growth by Club Size Category (Fall 23 to Fall 24)')
    ax_percentage_growth_by_size.set_xlabel('Size Category')
    ax_percentage_growth_by_size.set_ylabel('Percentage Growth')
    st.pyplot(fig_percentage_growth_by_size)

    # -----------------------------------------
    # Clubs with Declining Membership
    # -----------------------------------------

    # Filter clubs with negative growth and F24 Total # > 1
    declining_clubs = df_comp[(df_comp['Total Growth F23 to F24'] < 0) & (df_comp['F24 Total #'] > 1)]

    st.subheader("Clubs with Declining Membership (Fall 23 to Fall 24)")
    st.write("This bar chart shows the clubs that experienced a decline in total membership from Fall 2023 to Fall 2024.")

    # Plot clubs with declining membership
    if not declining_clubs.empty:
        fig_declining_clubs, ax_declining_clubs = plt.subplots(figsize=(12, len(declining_clubs) * 0.5))
        sns.barplot(x='Total Growth F23 to F24', y='Club', data=declining_clubs.sort_values('Total Growth F23 to F24'), palette='rocket', ax=ax_declining_clubs)
        ax_declining_clubs.set_title('Clubs with Declining Membership (Fall 23 to Fall 24)')
        ax_declining_clubs.set_xlabel('Total Growth')
        ax_declining_clubs.set_ylabel('Club')
        st.pyplot(fig_declining_clubs)

        # Print clubs with most decline
        st.write("Clubs with Decline:")
        declined_clubs = declining_clubs[['Club', 'F23 Total #', 'F24 Total #', 'Total Growth F23 to F24']].sort_values(by='Total Growth F23 to F24')
        st.write(declined_clubs)
    else:
        st.write("No clubs have declining membership.")

    # -----------------------------------------
    # End of Analysis
    # -----------------------------------------
    st.markdown("---")
    st.write("This concludes the enhanced membership analysis report for Rush Soccer Clubs.")

else:
    st.warning("Please upload all three CSV files to proceed with the analysis.")

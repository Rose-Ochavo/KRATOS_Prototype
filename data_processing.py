import csv
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler # LabelEncoder = a class
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np # library
import streamlit as st
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import calendar
import pandas as pd
import datetime
def load_and_process_data(uploaded_file):
    data = []

    # Read lines from uploaded_file and decode each line
    # utf-8 = supports all characters from all writing systems in the world
    decoded_file = [line.decode('utf-8') for line in uploaded_file.readlines()] # reads all the lines from `uploaded_file`
    csv_reader = csv.reader(decoded_file) # read the csv format
    # You can then use this csv_reader object to iterate through the rows and columns of the CSV data.

    for row in csv_reader:
        data.append(row) # append = add an element to the end of an existing list

    # ['ID', 'TRANSACTION', 'BRAND', 'ITEM', 'QUANTITY', 'PRICE', 'DATE']
    headers = data[0] # first row of the data list
    data = data[1:] # slice the list starting from 1

    # Convert price to float
    # iterating through a dataset row by row,
    for row in data:
        # Process each row, where row is a list of column values
        try:
            row[4] = float(row[4]) # Access the fifth column of the current row
        except ValueError: # correct data type but an inappropriate value
            row[4] = 0.0 # assign it to 0.0

        try:
            row[5] = float(re.sub('[^0-9.]', '', row[5])) # remove any characters that are not digits or periods (i.e., non-numeric characters) 
        except ValueError:
            row[5] = 0.0  # Or another placeholder value

    # Using label encoders to convert string data into numeric for clustering
    # This encoder is used to convert categorical data (like brand names) into numerical labels.
    # convert strings (categorical labels) into unique numbers.
        # assigns a unique integer to each unique string label
    brand_encoder = LabelEncoder() 
    date_encoder = LabelEncoder()

    # create list of brands and dates
    # iterates through each `row` in the `data`
    # For each `row`, it extracts the value in the third column (column index 2) and adds that value to the `brands` list.
    # documentionlink(1)
    brands = [row[2] for row in data]
    dates = [row[6] for row in data]

    # fit_transform(brands) = a method call on the brand_encoder object. This method does two things:
    """ *Fitting = It learns the mapping between unique brand names and numeric labels.
          assign a unique numeric label to each unique brand in the `brands` list. """  
    # *Transforming = replacing each brand name with its corresponding numeric label
    encoded_brands = brand_encoder.fit_transform(brands)
    encoded_dates = date_encoder.fit_transform(dates)

    # idx = index number of the data from the list ex.:
    """
        data = [['John', 'Doe', '30'],
            ['Jane', 'Smith', '25'],
            ['Bob', 'Johnson', '40']]

        for idx, row in enumerate(data):
            print(f"Index: {idx}, Data: {row}")

        # Output
        Index: 0, Data: ['John', 'Doe', '30']
        Index: 1, Data: ['Jane', 'Smith', '25']
        Index: 2, Data: ['Bob', 'Johnson', '40']
    """
    # enumerate() function is used to get both the index (idx) and the content of each row (row) in the list.
    # It will iterate through each row in the `data` list and replace the values in specific columns with their corresponding encoded values
    # row[2] and row[6] in eacht row of data will be replace with encoded data `encoded_brands[idx]` 
    for idx, row in enumerate(data): # iterates through each row in the `data` list
        row[2] = encoded_brands[idx]  # Encoded brand
        row[6] = encoded_dates[idx]  # Encoded date

    return data, headers,brand_encoder, date_encoder

# involved training
def visualize_data(data, headers,brand_encoder, date_encoder):
    # In the encoder=None parameter, if no argument is provided when calling the function, it will be automatically set to `None` as its default value.
    def apply_kmeans_on_single_column(data, column_idx, encoder=None, n_clusters=4):
        # Extract the relevant column data
        # Similar to documentionlink(1)
        X = np.array([[row[column_idx]] for row in data])
        
        # If the data is continuous, it's a good idea to scale it
        # responsible for scaling the data when necessary before applying K-Means clustering
        # isinstance(X[0][0], (int, float)) = checks whether the first element (row 0, column 0) of the data array X is an instance of either an int or a float. 
        # column_idx not in [5, 6] = checks whether the `column_idx` is not in the list [5, 6]
            # it checks if the column being processed is not the 6th or 7th column
        if isinstance(X[0][0], (int, float)) and column_idx not in [5, 6]:  # Exclude special columns from scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) # fit_transform method of the StandardScaler
            # Standardize the data in X, ensuring it has a mean of 0 and a standard deviation of 1.
            # Scaling = process of transforming the numeric values in a dataset so that they have a similar scale or magnitude
                #  It involves adjusting the range of values while preserving the underlying patterns and relationships in the data.
                # Scaling method used: [https://www.simplilearn.com/normalization-vs-standardization-article]
                    # `StandardScaler` from the sklearn.preprocessing module is used for scaling the data before applying the K-Means

            # Why Scaling Matters:
                # Imagine you have a dataset with multiple columns or features, and these features have different units or measurement scales. For example, one feature could represent age in years, another could represent income in dollars, and a third could represent the number of bedrooms in a house. Each of these features can have vastly different ranges of values.

            # K-Means clustering = These algorithms often use mathematical calculations that involve the data's magnitude. If the features are not on a similar scale, it can lead to various problems:
                # Magnitude Dominance: Features with larger magnitudes may dominate the influence over those with smaller magnitudes. As a result, the algorithm may incorrectly prioritize certain features.
                # Algorithm Sensitivity: Certain algorithms, such as K-Means clustering, rely on distance calculations. Features with larger scales can disproportionately affect the distance calculations and cluster assignments.
        else:
            X_scaled = X
        
        # Apply KMeans clustering
        # `random_state` parameter is used to control the random initialization of centroids in the clustering process.
            # random initialization of centroids
            # Avoiding Initialization Bias
        # .fit(X_scaled) is the part where the model is trained. It takes the scaled data X_scaled and performs the following steps:
            # Initializes cluster centroids randomly (or according to some initialization method).
            # Repeatedly assigns each data point to the nearest cluster centroid.
            # Updates the cluster centroids based on the mean of the data points assigned to each cluster.
            # Repeats the assignment and update steps until convergence (i.e., when the cluster assignments and centroids no longer change significantly).
        # The training process involves initializing cluster centroids randomly, assigning data points to the nearest cluster centroids, updating cluster centroids based on the mean of assigned data points, and repeating these steps until convergence.
	        # After training, the model stores the cluster assignments in kmeans.labels_.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
        labels = kmeans.labels_ # is an attribute of the fitted KMeans object, which stores the cluster labels assigned to each data point.
        
        # Calculate the silhouette score
        # The silhouette score tells us how good our clusters are:
            # - It measures how close the items inside a cluster are to each other compared to items in other clusters.
            # - It gives us a score between -1 and +1:
            #   - A higher score means our clusters are good and distinct.
            #   - A lower score means our clusters might not be very clear.
        silhouette_avg = silhouette_score(X_scaled, labels) # silhouette_score = a function
        st.text(f"Silhouette Score for column {headers[column_idx]}: {silhouette_avg:.2f}")

        
        # Visualization
        # skip plt
        plt.figure(figsize=(10, 6))
        plt.scatter(X, [i for i in range(len(X))], c=kmeans.labels_, cmap='rainbow')
        
        # Get centroid colors
        centroid_colors = [plt.cm.rainbow(label/n_clusters) for label in range(n_clusters)]
        
        centroid_scatter = plt.scatter(kmeans.cluster_centers_, [i for i in range(n_clusters)], s=200, c=centroid_colors, marker='X')
        
        # If an encoder is provided and it's a label encoded column, reverse the encoded values and set legend
        if encoder:
            original_values = [encoder.inverse_transform([int(center)])[0] for center in np.round(kmeans.cluster_centers_).flatten()]
            # Create a legend for encoded columns
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=value, markersize=10, markerfacecolor=centroid_colors[i]) for i, value in enumerate(original_values)]
            plt.legend(handles=legend_handles, title=headers[column_idx])
        else:
            # Create a legend for other columns with centroids
            legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Centroid {i+1}", markersize=10, markerfacecolor=centroid_colors[i]) for i in range(n_clusters)]
            plt.legend(handles=legend_handles, title=headers[column_idx])
        
        # Special x-labels for price and date columns
        if column_idx == 5:
            plt.xlabel("Price")
            flattened_X = X.flatten()
            plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[f"{tick:.2f}" for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
        elif column_idx == 6:
            plt.xlabel("Date")
            flattened_X = X.flatten()
            plt.xticks(ticks=np.linspace(min(flattened_X), max(flattened_X), n_clusters), labels=[encoder.inverse_transform([int(tick)])[0] for tick in np.linspace(min(flattened_X), max(flattened_X), n_clusters)])
        else:
            plt.xlabel(headers[column_idx])
        
        plt.ylabel('Data Points Index')
        plt.title(f'KMeans Clustering on {headers[column_idx]}')
        plt.show()
        st.pyplot(plt.gcf())

    # Call the function with encoder parameter for brand and date columns
    apply_kmeans_on_single_column(data, 2, brand_encoder)
    apply_kmeans_on_single_column(data, 4)
    apply_kmeans_on_single_column(data, 5)
    apply_kmeans_on_single_column(data, 6, date_encoder)




    #clustering two clumn together
    # Custom function to apply KMeans and calculate silhouette score
    def apply_kmeans_and_evaluate(data, columns, encoders=None, n_clusters=3):
        # # Create a list of column indices by iterating through the specified columns and finding their indices in the 'headers' list
        col_indices = [headers.index(col) for col in columns] # col = columns iterate and find col from headers in index(col)
        X = np.array([[row[i] for i in col_indices] for row in data]) # create an array list of all ["BRAND", "DATE"] from the data
        
        # Apply KMeans clustering
        # `random_state` parameter is used to control the random initialization of centroids in the clustering process.
            # random initialization of centroids
            # Avoiding Initialization Bias
        # .fit(X_scaled) is the part where the model is trained. It takes the scaled data X_scaled and performs the following steps:
            # Initializes cluster centroids randomly (or according to some initialization method).
            # Repeatedly assigns each data point to the nearest cluster centroid.
            # Updates the cluster centroids based on the mean of the data points assigned to each cluster.
            # Repeats the assignment and update steps until convergence (i.e., when the cluster assignments and centroids no longer change significantly).
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        labels = kmeans.labels_ # is an attribute of the fitted KMeans object, which stores the cluster labels assigned to each data point.
        
        # Calculate the silhouette score
        # The silhouette score tells us how good our clusters are:
            # - It measures how close the items inside a cluster are to each other compared to items in other clusters.
            # - It gives us a score between -1 and +1:
            #   - A higher score means our clusters are good and distinct.
            #   - A lower score means our clusters might not be very clear.
        silhouette_avg = silhouette_score(X, labels)
        st.write(f"Silhouette Score for columns {columns}: {silhouette_avg:.2f}")

        
        # Plotting the clusters
        # skip plt
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X')
        
        # Adjust labels for BRAND vs DATE
        if columns == ["BRAND", "DATE"]:
            unique_brands = sorted(list(set(encoders[0].inverse_transform(X[:, 0].astype(int)))))
            dates = sorted(list(set(encoders[1].inverse_transform(X[:, 1].astype(int)))))
            min_date = dates[0]
            max_date = dates[-1]
            mid_date = dates[len(dates)//2]
            plt.xticks(ticks=range(len(unique_brands)), labels=unique_brands)
            plt.yticks(ticks=[0, len(dates)//2, len(dates)-1], labels=[min_date, mid_date, max_date])
        elif columns == ["ITEM", "PRICE"]:
            total_items = len(set([row[headers.index("ITEM")] for row in data]))
            plt.xticks(ticks=range(0, total_items, total_items//3))
        
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        
        # Create a legend for cluster centers
        legend_handles = [plt.Line2D([0], [0], marker='X', color='w', label=f"Consumers {i+1}", markersize=10, markerfacecolor='black') for i in range(n_clusters)]
        plt.legend(handles=legend_handles, title="Group of Consumers")
        
        plt.title(f'KMeans Clustering: {columns[0]} vs {columns[1]}')
        plt.show()
        st.pyplot(plt.gcf())

    # 1. Brand and Date
    apply_kmeans_and_evaluate(data, ["BRAND", "DATE"], encoders=[brand_encoder, date_encoder])

    # 2. Price and Quantity
    apply_kmeans_and_evaluate(data, ["PRICE", "QUANTITY"])

    # Using label encoders to convert string data into numeric for clustering
    # This encoder is used to convert categorical data (like brand names) into numerical labels.
    product_encoder = LabelEncoder()

    # in column 3 [ITEM] in each row of the data
    # fit_transform(ITEM) = a method call on the product_encoder object. This method does two things:
    """ *Fitting = It learns the mapping between unique ITEM names and numeric labels.
            assign a unique numeric label to each unique ITEM in the `ITEM` list. """  
    # *Transforming = replacing each ITEM name with its corresponding numeric label
    encoded_products = product_encoder.fit_transform([row[3] for row in data]) 

    # idx = index number of the data from the list ex.:
    """
        data = [['John', 'Doe', '30'],
            ['Jane', 'Smith', '25'],
            ['Bob', 'Johnson', '40']]

        for idx, row in enumerate(data):
            print(f"Index: {idx}, Data: {row}")

        # Output
        Index: 0, Data: ['John', 'Doe', '30']
        Index: 1, Data: ['Jane', 'Smith', '25']
        Index: 2, Data: ['Bob', 'Johnson', '40']
    """
    # enumerate() function is used to get both the index (idx) and the content of each row (row) in the list.
    # It will iterate through each row in the `data` list and replace the values in specific columns with their corresponding encoded values
    # row[3] in each row of data will be replace with encoded data `encoded_products[idx]` 
    for idx, row in enumerate(data): # iterates through each row in the `data` list
        row[3] = encoded_products[idx]  # Encoded product

    # Apply k-means clustering on 'Product' and 'Month'
    apply_kmeans_and_evaluate(data, ["ITEM", "PRICE"])


# Does not involved training
# provides insights into associations between items in the data.
def display_association_results(uploaded_file):
    # Read directly from uploaded_file using csv.reader
    text_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Read the CSV from the decoded content
    data = [row for row in csv.reader(text_data)]
    
    # This part of the code creates a dictionary called transactions using defaultdict(set). It's used to group products by transactions, where each transaction ID is associated with a set of items.
    # It iterates through each row in the data list (which represents the CSV rows) and extracts the transaction_id (column at index 1) and the item (column at index 3) for each row.
    # It then adds each item to the set of items associated with the corresponding transaction_id in the transactions dictionary.
    # Grouping products by transactions
    transactions = defaultdict(set)
    for row in data:
        transaction_id = row[1]
        item = row[3]
        transactions[transaction_id].add(item)

    # Count occurrences of individual items
    item_counts = Counter(item for items in transactions.values() for item in items)

    # Calculate support for each item
    total_transactions = len(transactions)
    itemset_support = {frozenset([item]): count/total_transactions for item, count in item_counts.items()}

    # Count itemsets
    itemset_counts = Counter()
    max_itemset_size = 2 # item pairs

    for items in transactions.values():
        if len(items) < 2: # checks if the transaction has fewer than 2 items, then continue to another transaction
            continue
        for L in range(2, min(len(items), max_itemset_size) + 1): # Without + 1, the loop would iterate up to, but not including, max_itemset_size.
                # if max_itemset_size is set to 2, and a transaction has 2 items, you would want the loop to generate itemsets of size 2 (i.e., pairs). Without + 1, the loop would stop at 1, and pairs would not be generated. With + 1, it ensures that pairs are generated as well.
            for subset in combinations(items, L): # generates all possible combinations of L items from the items list.
                itemset = frozenset(subset)
                itemset_counts[itemset] += 1 # This counts how many times each itemset appears in the transactions.

    # Calculate support for each itemset
    itemset_support.update({itemset: count/total_transactions for itemset, count in itemset_counts.items()})

    # Sorting the itemsets by support
    sorted_itemsets = sorted(itemset_support.items(), key=lambda x: x[1], reverse=True)

    # extracts the top 10 2-itemsets (pairs) with the highest support values from the sorted_itemsets list.
    # Extract 2-itemsets
    sorted_2_itemsets = [(itemset, support) for itemset, support in sorted_itemsets if len(itemset) == 2][:10]

    # Calculate Association Rules
    association_rules = []
    for itemset, support in itemset_support.items():
        if len(itemset) == 2:
            for item in itemset: # For each 2-itemset, create association rules.
                antecedent = frozenset([item])
                consequent = itemset - antecedent # It's what you expect to be purchased alongside the antecedent.

                antecedent_support = itemset_support.get(antecedent, 0)
                confidence = support / antecedent_support
                association_rules.append((antecedent, consequent, confidence)) # adds an association rule to a list

    # Sort rules by support and then confidence
    sorted_rules = sorted(association_rules, key=lambda x: (itemset_support[x[0] | x[1]], x[2]), reverse=True)

    # Displaying the results in Streamlit
    # Skip
    st.subheader("Top Itemsets with Support:")
    for itemset, support in sorted_itemsets[:10]:
        st.write(f"{', '.join(itemset)}: Support = {support:.2f}")

    st.subheader("Pairs with High Support and High Confidence:")
    for antecedent, consequent, confidence in sorted_rules[:10]:
        support_value = itemset_support[antecedent | consequent]
        st.write(f"{', '.join(antecedent)} -> {', '.join(consequent)}: Support = {support_value:.2f}, Confidence = {confidence:.2f}")

    # Plotting
    if sorted_2_itemsets:
        labels = [', '.join(itemset) for itemset, _ in sorted_2_itemsets]
        supports = [support for _, support in sorted_2_itemsets]

        plt.figure(figsize=(10, 6))
        plt.barh(labels, supports, color='lightblue')
        plt.xlabel('Support')
        plt.ylabel('Itemsets')
        plt.title('Top 10 2-Itemsets with Highest Support')
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
    else:
        st.write("No 2-itemsets available for plotting!")
        
    


# involved training
def run_random_forest_regression(uploaded_file):
    text_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    
    # Read the CSV from the decoded content
    data = [row for row in csv.reader(text_data)]
    header = data[0]  
    df = pd.DataFrame(data[1:], columns=header)  
    
    # Convert DATE to datetime format and extract month names
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['Month_Name'] = df['DATE'].dt.strftime('%B')
    df['BRAND'] = df['BRAND'].str.strip().str.upper()

    # Brands in the data
    brands = df['BRAND'].unique()
    month_order = sorted(df['Month_Name'].unique().tolist(), key=lambda x: datetime.datetime.strptime(x, '%B'))

    # Set up plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Evaluation scores
    evaluation_scores = {}

    for brand in brands:
        brand_data = df[df['BRAND'] == brand]
        # .size() = count the number of occurrences
        # .reindex() =  includes all months
            # If a month is missing in the original data, it will be added to the Series 
        monthly_sales = brand_data.groupby('Month_Name').size().reindex(month_order, fill_value=0).reset_index(name='Occurrences')
        months = np.array(range(1, len(monthly_sales) + 1)).reshape(-1, 1) # create an array `months` that represents the months as numerical values. 
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(months, monthly_sales['Occurrences'], test_size=0.3, random_state=42)
        
        # Using a Random Forest Regressor
        # training and prediction
        # n_estimators=100: This parameter specifies the number of decision trees 
            # 100 trees strike a balance between accuracy and computational efficiency.
        # RandomForestRegressor = a class
            # 'random_state' makes sure that random results remain the same in every execution, as long as you use the same 'random_state' number.
        model = RandomForestRegressor(n_estimators=100, random_state=42) # Model Initialization:
        model.fit(X_train, y_train) # Model Training.. fit() = method
            # fit method is to train the machine learning model on a given dataset. During training, the model learns the underlying patterns and relationships in the data that allow it to make predictions.
            # trains the Random Forest Regressor model using the provided training data.
            
            # X_train: months
            # y_train: monthly sales occurrences corresponding to each month in X_train.

        predictions = model.predict(X_test) # Making Predictions
            # predict brand popularity for a set of future time periods (X_test). 
        
        # Evaluate the model
        # smaller MAE the better the model's predictions 
        mae = mean_absolute_error(y_test, predictions) # measures the average absolute difference between the actual and predicted values

        # Lower MSE values indicate better model performance.
        mse = mean_squared_error(y_test, predictions) # calculates the average of the squared differences 
        r2 = r2_score(y_test, predictions) # calculates the R-squared (R²) score
            # R squared R2 score ranges between 0 and 1, where 1 indicates a perfect fit, and lower values indicate worse model performance.
        evaluation_scores[brand] = (mae, mse, r2) # stores evaluation scores in a dictionary
        
        # Predict the occurrences for the next 3 months
        # redundancy = len(monthly_sales) + 1
        # np.array = convert indices to numpay array
        # .reshape = reshapes the NumPy array into a two-dimensional array
            # model.predict function expects input data to be in a two-dimensional format
        future_months = np.array(range(len(monthly_sales) + 1, len(monthly_sales) + 4)).reshape(-1, 1)
        future_predictions = model.predict(future_months) # uses a trained machine learning model to make predictions for the future months
        
        # Plot the data
        all_data = list(monthly_sales['Occurrences']) + list(future_predictions) # concat the two lists
        all_months = month_order + [calendar.month_name[(month_order.index(month_order[-1]) + 1 + i) % 12 + 1] for i in range(3)] # concat the list of months and the names of the next 3 months

        ax.plot(all_months, all_data, marker='o', label=brand)
        ax.scatter(month_order, monthly_sales['Occurrences'], color='gray')  # actual data points

    # Formatting the plot
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Occurrences')
    ax.set_title('Predicted Brand Popularity in Upcoming Months')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True)

    return fig, evaluation_scores






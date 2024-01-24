#!/usr/bin/env python
# coding: utf-8

# # Case Study: Predictive Analytics for E-commerce

# ## Data Preparation

# In[1]:


import pandas as pd


# In[44]:


customer = pd.read_csv('customer_interactions.csv')


# In[45]:


customer


# In[31]:


product = pd.read_csv('product_details.csv')


# In[32]:


product


# In[33]:


product = pd.read_csv('product_details.csv', sep = ";")


# In[34]:


product


# In[35]:


product = product.loc[:, ~product.columns.str.contains('^Unnamed')]


# In[36]:


product


# In[38]:


purchase = pd.read_csv('purchase_history.csv')


# In[39]:


purchase


# In[40]:


purchase = pd.read_csv('purchase_history.csv', sep = ";")


# In[41]:


purchase


# In[42]:


purchase = purchase.loc[:, ~purchase.columns.str.contains('^Unnamed')]


# In[43]:


purchase


# In[223]:


# Merged Dataset
df = pd.merge(customer, purchase, on='customer_id')
df = pd.merge(df, product, on='product_id')
df


# ## Data Exploration and Preprocessing

# In[227]:


print("Dataset Overview:\n")
print(df.info())


# In[228]:


print("Summary Statistics:\n")
print(df.describe())


# In[229]:


print("Missing Values:\n")
print(df.isnull().sum())


# In[230]:


print("Duplicated Values:\n")
print(df.duplicated().sum())


# In[231]:


print("Correlation Between Columns:\n")
print(df.corr())


# In[233]:


df.shape


# In[234]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[307]:


# Scatterplot
plt.figure(figsize = (7,5))
sns.scatterplot(x = 'page_views', y = 'time_spent', data = df, hue ='product_id', s=150)
plt.title('Scatter Plot: Page Views vs Time Spent')
plt.xlabel('Page Views')
plt.ylabel('Time Spent')
plt.legend(title='Product ID')
plt.show()

plt.figure(figsize = (7,5))
sns.scatterplot(x = 'page_views', y = 'price', data = df, hue ='product_id', s=150)
plt.title('Scatter Plot: Page Views vs Price')
plt.xlabel('Page Views')
plt.ylabel('Price')
plt.legend(title='Product ID')
plt.show()

plt.figure(figsize = (7,5))
sns.scatterplot(x = 'page_views', y = 'ratings', data = df, hue ='product_id', s=150)
plt.title('Scatter Plot: Page Views vs Ratings')
plt.xlabel('Page Views')
plt.ylabel('Ratings')
plt.legend(title='Product ID')
plt.show()

plt.figure(figsize = (7,5))
sns.scatterplot(x = 'time_spent', y = 'price', data = df, hue ='product_id', s=150)
plt.title('Scatter Plot: Time Spent vs Price')
plt.xlabel('Time Spent')
plt.ylabel('Price')
plt.legend(title='Product ID')
plt.show()

plt.figure(figsize = (7,5))
sns.scatterplot(x = 'time_spent', y = 'ratings', data = df, hue ='product_id', s=150)
plt.title('Scatter Plot: Time Spent vs Ratings')
plt.xlabel('Time Spent')
plt.ylabel('Ratings')
plt.legend(title='Product ID')
plt.show()

plt.figure(figsize = (7,5))
sns.scatterplot(x = 'price', y = 'ratings', data = df, hue ='customer_id', s = 150)
plt.title('Scatter Plot: Price vs Ratings')
plt.xlabel('Price')
plt.ylabel('Rataings')
plt.legend(title = 'Customer ID')
plt.show


# In[261]:


# Detect outliers using boxplot
# Partial Boxplot

plt.figure(figsize = (5,5))
sns.boxplot(y = df['page_views'])
plt.title('Boxplot of Page Views')
plt.ylabel('Page Views')
plt.show()

plt.figure(figsize = (5,5))
sns.boxplot(y = df['time_spent'])
plt.title('Boxplot of Time Spent')
plt.ylabel('Time Spent')
plt.show()

plt.figure(figsize = (5,5))
sns.boxplot(y = df['price'])
plt.title('Boxplot of Price')
plt.ylabel('Price')
plt.show()

plt.figure(figsize = (5,5))
sns.boxplot(y = df['ratings'])
plt.title('Boxplot of Ratings')
plt.ylabel('Ratings')
plt.show()


# ## Model Development

# In[298]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[299]:


# Split dataframe in x = features and y = target varaibles
x = df[['page_views', 'time_spent', 'price', 'ratings']]
y = df['product_id']


# In[300]:


# Split dataframe into training set and testing set (65% tarining and 35% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[327]:


# Create random forest classifier object
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train random forest classifier
model.fit(x_train, y_train)


# In[328]:


# Predict the respons for the test dataframe
y_predict = model.predict(x_test)
y_predict


# In[329]:


# Model accuracy, how often is the classifier correct?
accuracy = accuracy_score(y_test, y_predict)
print("Akurasi Model: {:.2f}%".format(accuracy * 100))


# ## Web Application Developmen

# In[326]:


from flask import Flask, render_template, request
app = Flask(__name__)


# In[330]:


# Fungsi untuk memprediksi produk berdasarkan ID pelanggan
def predict_product(customer_id):
    # Lakukan pemrosesan data sesuai kebutuhan
    customer_data = df[df['customer_id'] == customer_id][['page_views', 'time_spent', 'price', 'ratings']]
    
    # Gunakan model yang sudah dilatih untuk memprediksi pembelian
    predicted_purchase = model.predict(customer_data)
    
    # Filter produk yang belum dibeli
    recommended_products = product_details_df[product_details_df['product_id'].isin(customer_data[customer_data.index == True].index)]['product_id']
    
    return recommended_products


# In[331]:


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        customer_id = request.form['customer_id']
        recommended_products = predict_product(customer_id)
        return render_template('result.html', customer_id=customer_id, recommended_products=recommended_products)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





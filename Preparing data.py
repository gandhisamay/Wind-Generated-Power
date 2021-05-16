train_df = pd.read_csv('/content/drive/MyDrive/HackerEarth Power Generated Challenge /train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/HackerEarth Power Generated Challenge /test.csv')

train_df.head()

test_df

train_df.describe()

#Replacing low, medium, extremely low in cloud level column with their one hot encoded values

train_df['cloud_level'] = train_df['cloud_level'].replace({'Low' : 0 , 'Medium' : 1, 'Extremely Low' : -1})

#Doing the same as above for test set

test_df['cloud_level'] = test_df['cloud_level'].replace({'Low' : 0 , 'Medium' : 1, 'Extremely Low' : -1})

#Replacing the various text described values in the column turbine_status with their one hot encode values
onehotencods = {}
for i in range(len(train_df['turbine_status'].unique())):
  onehotencods[train_df['turbine_status'].unique()[i]] = i
  train_df['turbine_status'].replace({train_df['turbine_status'].unique()[i] : i}, inplace = True)

onehotencods

#Applying these one hot encoded numbers in test dataset
test_df['turbine_status'].replace(onehotencods, inplace = True)

print(test_df['turbine_status'].value_counts())

#Replace the nan values in the train dataset by corresponding median values of the column

for i in range(len(train_df.describe().columns)):
  train_df.iloc[:,i+2].replace(np.NAN,train_df.iloc[:,i+2].median(),inplace = True)

train_df.describe()

#Doing the same for the test set
for i in range(len(test_df.describe().columns)):
  test_df.iloc[:,i+2].replace(np.NAN,test_df.iloc[:,i+2].median(),inplace = True)

test_df.describe()

#Shuffling the indices
np.random.seed(10)
n_dev = 3200 #Data for dev set
shuffled_indices = np.random.permutation(train_df.shape[0])

#Load the train data into numpy arrays
x = np.array(train_df.loc[:,'wind_speed(m/s)':'windmill_height(m)'])
y = np.array(train_df['windmill_generated_power(kW/h)'])

#Make train and dev sets
x_train = x[shuffled_indices[n_dev:]]
y_train = y[shuffled_indices[n_dev:]]

x_dev = x[shuffled_indices[:n_dev]]
y_dev = y[shuffled_indices[:n_dev]]

x_train.shape

#Plot the graph between various features and Wind power values

plt.style.use('seaborn')
plt.xlabel('Wind_speed(m/s)')
plt.ylabel('Windmill Generated Power')
plt.scatter(train_df['wind_speed(m/s)'],train_df['windmill_generated_power(kW/h)'], s = 4)

plt.style.use('seaborn')
plt.xlabel('')
plt.ylabel('Windmill Generated Power')
plt.scatter(train_df['atmospheric_temperature(°C)'],train_df['windmill_generated_power(kW/h)'], s = 4)

#Reshape y_train
y_train = y_train.reshape(-1,1)
y_dev = y_dev.reshape(-1,1)

#Load the test data into numpy arrays
x_test = np.array(test_df.loc[:,'wind_speed(m/s)':'windmill_height(m)'])

#Normalizing the data now
x_train = (x_train - x_train.mean(axis = 0))/x_train.std(axis = 0)
x_dev = (x_dev - x_dev.mean(axis = 0))/ x_dev.std(axis = 0)
x_test = (x_test - x_test.mean(axis = 0))/x_test.std(axis = 0)

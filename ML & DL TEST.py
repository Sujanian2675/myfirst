# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Data
# X = np.array([[1500, 3], [2000, 4], [2200, 5], [1800, 4], [2400, 4]])
# y = np.array([250000, 320000, 300000, 350000, 350000])

# # Initialization
# models = {'Linear Regression': LinearRegression(),
#     'Decision Tree': DecisionTreeRegressor(),
#     'KNN': KNeighborsRegressor(n_neighbors=3)}

# mse_results = {}
# for name, model in models.items():
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     mse_results[name] = mse

# # MSE
# for name, mse in mse_results.items():
#     print(f"{name} MSE: {mse}")

------------------------------------------------------------


# import pandas as pd
# import numpy as np
# data = {
#     'Student': [1, 2, 3, 4, 5],
#     'Age': [20, 22, np.nan, 25, 21],
#     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
#     'Test Score': [85, 90, 75, np.nan, 80]
# }

# df = pd.DataFrame(data)

# age_median = df['Age'].median()
# df['Age'].fillna(age_median, inplace=True)

# test_score_median = df['Test Score'].median()
# df['Test Score'].fillna(test_score_median, inplace=True)

# gender_mode = df['Gender'].mode()[0]
# df['Gender'].fillna(gender_mode, inplace=True)

--------------------------------------------------------------------

# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np

# # Loading and Preparation of Data
# data = {
#     'Car': ['Car 1', 'Car 2', 'Car 3', 'Car 4', 'Car 5'],
#     'Horsepower': [130, 150, 170, 180, 200],
#     'Weight (lbs)': [2500, 2800, 3000, 3200, 3500],
#     'Fuel Efficiency (mpg)': [30, 28, 26, 25, 23]
# }

# df = pd.DataFrame(data)

# #  Computing Correlations
# correlation_matrix = df[['Horsepower', 'Weight (lbs)', 'Fuel Efficiency (mpg)']].corr()
# print("Correlation Matrix :")
# print(correlation_matrix)

# x = df[['Horsepower', 'Weight (lbs)']]
# y = df['Fuel Efficiency (mpg)']

# model = LinearRegression()
# model.fit(x, y)

# coefficients = model.coef_
# feature_importance = pd.Series(coefficients, index=['Horsepower', 'Weight (lbs)'])

# print("\nFeature Importance from Linear Regression:")
# print(feature_importance)

# standardized_coefficients = feature_importance / np.std(x, axis=0)
# print("\nStandardized Feature Importance :")
# print(standardized_coefficients)

------------------------------------------------------------------

# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import accuracy_score

# data = {
#     'Sepal Length (cm)': [5.1, 4.9, 6.9, 6.7, 6.3],
#     'Sepal Width (cm)': [3.5, 3.0, 3.1, 3.1, 2.5],
#     'Petal Length (cm)': [1.4, 1.4, 4.9, 4.7, 5.0],
#     'Petal Width (cm)': [0.2, 0.2, 1.5, 1.5, 1.9],
#     'Species': ['Setosa', 'Setosa', 'Versicolor', 'Versicolor', 'Virginica']
# }

# df = pd.DataFrame(data)

# df['Species'] = df['Species'].astype('category').cat.codes

# x = df[['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']]
# y = df['Species']

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the Decision Tree Classifier
# dt = DecisionTreeClassifier(random_state=42)

# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# grid_search.fit(x_train, y_train)

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("Best Hyperparameters:", best_params)
# print("Best Cross-Validation Score:", best_score)


# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(x_test)
# test_accuracy = accuracy_score(y_test, y_pred)

# print("Test Set Accuracy:", test_accuracy)

------------------------------------------------------------
# DEEP LEARNING TASKS

-------------------------------------------------------------

# import numpy as np
# import pandas as pd
# import gensim.downloader as api
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, Flatten
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Loading pre-trained Word2Vec model
# word2vec_model = api.load('word2vec-google-news-300')

# # Prepare sentiment data (sample dataset)
# data = {
#     'text': [
#         'I like this movie',
#         'This movie was not good',
#         'Fantastic film with great acting',
#         'I did not enjoy the film',
#         'One of the best movies I have ever seen'
#     ],
#     'label': [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
# }

# df = pd.DataFrame(data)

# # Tokenize and preprocessing of text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df['text'])
# sequences = tokenizer.texts_to_sequences(df['text'])
# word_index = tokenizer.word_index

# # Pad sequences to ensure uniformity of  input length
# max_length = max(len(seq) for seq in sequences)
# X = pad_sequences(sequences, maxlen=max_length)
# y = np.array(df['label'])

# #  Converting tokens into word vectors
# embedding_dim = 300
# embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
# for word, i in word_index.items():
#     if word in word2vec_model:
#         embedding_matrix[i] = word2vec_model[word]

# # the neural network model
# def build_model(vocab_size, embedding_dim, input_length):
#     model = Sequential([
#         Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, weights=[embedding_matrix], trainable=False),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Spliting data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Building model with pre-trained embeddings
# model = build_model(len(word_index) + 1, embedding_dim, max_length)

# # Training the model
# history = model.fit(X_train, y_train, epochs=5, batch_size=2, validation_split=0.2)

# # Evaluating the model
# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# test_accuracy = accuracy_score(y_test, y_pred)

# print(f'Test Accuracy with Pre-trained Embeddings: {test_accuracy}')

# #Repeat with random initialization of the embedding layer
# def build_model_random(vocab_size, embedding_dim, input_length):
#     model = Sequential([
#         Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, trainable=True),
#         Flatten(),
#         Dense(64, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# # Build model with random initialization
# model_random = build_model_random(len(word_index) + 1, embedding_dim, max_length)

# # Train the random initialization model
# history_random = model_random.fit(X_train, y_train, epochs=5, batch_size=2, validation_split=0.2)

# # Evaluate the random initialization model
# y_pred_random = (model_random.predict(X_test) > 0.5).astype("int32")
# test_accuracy_random = accuracy_score(y_test, y_pred_random)

# print(f'Test Accuracy with Random Initialization: {test_accuracy_random}')

------------------------------------------------------------------------------

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.losses import binary_crossentropy
# import matplotlib.pyplot as plt

# # Loading and preprocessing MNIST data
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)

# # Defining VAE model parameters
# img_shape = (28, 28, 1)
# latent_dim = 2

# inputs = layers.Input(shape=img_shape)
# x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
# x = layers.MaxPooling2D(2, padding='same')(x)
# x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
# x = layers.MaxPooling2D(2, padding='same')(x)
# x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64, activation='relu')(x)
# z_mean = layers.Dense(latent_dim)(x)
# z_log_var = layers.Dense(latent_dim)(x)

# # Sampling layer
# def sampling(args):
#     z_mean, z_log_var = args
#     batch = tf.shape(z_mean)[0]
#     dim = tf.shape(z_mean)[1]
#     epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#     return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# z = layers.Lambda(sampling)([z_mean, z_log_var])

# # Decoder
# decoder_inputs = layers.Input(shape=(latent_dim,))
# x = layers.Dense(7 * 7 * 128, activation='relu')(decoder_inputs)
# x = layers.Reshape((7, 7, 128))(x)
# x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
# x = layers.UpSampling2D(2)(x)
# x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
# x = layers.UpSampling2D(2)(x)
# outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

# # Defining VAE models
# encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
# decoder = models.Model(decoder_inputs, outputs, name='decoder')

# vae_outputs = decoder(encoder(inputs)[2])
# vae = models.Model(inputs, vae_outputs, name='vae')

# # Defining VAE loss
# reconstruction_loss = binary_crossentropy(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(vae_outputs))
# reconstruction_loss *= np.prod(img_shape)
# kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
# vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
# vae.compile(optimizer='adam')

# # Training VAE
# vae.fit(x_train, x_train, epochs=50, batch_size=128, validation_split=0.2)

# # Generating new images
# def generate_images(decoder_model, n_images=10):
#     latent_samples = np.random.normal(size=(n_images, latent_dim))
#     generated_images = decoder_model.predict(latent_samples)
#     return generated_images

# generated_images = generate_images(decoder, n_images=10)

# # Displaying the results
# def display_images(images, title='Generated Images'):
#     plt.figure(figsize=(10, 10))
#     for i in range(len(images)):
#         plt.subplot(1, len(images), i + 1)
#         plt.imshow(images[i].squeeze(), cmap='gray')
#         plt.axis('off')
#     plt.suptitle(title)
#     plt.show()

# display_images(generated_images)
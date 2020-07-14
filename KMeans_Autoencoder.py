#--------------------------------------------------------------------------------------------------------------------
#K Means using Autoencoder

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
     
restored_testing_dataset = loaded_model.predict(X_test)

encoder = K.function([loaded_model.layers[0].input], [loaded_model.layers[4].output])

encoded_images = encoder([X_test])[0].reshape(-1,7*7*7)

kmeans = KMeans(n_clusters=10)
clustered_training_set = kmeans.fit_predict(encoded_images)

#Mapping the learnt labels to true labels
labels = np.zeros_like(clustered_training_set)
for i in range(num_of_clusters):
    mask = (clustered_training_set == i)
    labels[mask] = mode(y_test[mask])[0]
    
#Computing the accuracy for the given k value
accuracy_score = metrics.accuracy_score(y_test, labels)
print('Accuracy is', accuracy_score)
score.append(accuracy_score)

#Confusion matrix 
cm = metrics.confusion_matrix(y_test, labels)
print(cm)

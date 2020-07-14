#--------------------------------------------------------------------------------------------------------------------
#Gaussian Mixture Models using auto encoder

#Loading the weights into new model
print("Loading the model from disk")

#load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
decoded_images = loaded_model.predict(X_test)

#Encoding
encoder = backend.function([loaded_model.layers[0].input], [loaded_model.layers[4].output])
encoded_images = encoder([X_test])[0].reshape(-1,7*7*7)

#GMM clustering for encoded data
GMM = mixture.GaussianMixture(n_components=10)
gmm_clusters = GMM.fit_predict(encoded_images)

#Mapping the labels found to the ground truth
labels = np.zeros_like(gmm_clusters)
for i in range(num_of_clusters):
    mask = (gmm_clusters == i)
    labels[mask] = mode(y_test[mask])[0]
    
#Computing the accuracy for the given k value
accurracy = metrics.accuracy_score(y_test, labels)
print('Accuracy is : ', accurracy)

#Confusion matrix
cm = metrics.confusion_matrix(y_test, labels)
print(cm)

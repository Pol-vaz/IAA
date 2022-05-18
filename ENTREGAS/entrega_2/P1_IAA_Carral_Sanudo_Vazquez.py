import itertools
from pickle import TRUE
from tkinter import N
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

GENERAL_PATH = r"C:\Users\pablo\Desktop\ENTREGAS\entrega_1"
CSV_PATH = GENERAL_PATH + "\heart_failure_clinical_records_dataset.csv"
GRAPH_PATH = GENERAL_PATH + "\graficas"

class preparacion_dataset:

    #Función lectura de datos
    def lectura_datos(self):
        dataLabels = [
        'age',
        'anaemia',
        'creatine',
        'diabetes',
        'ejection_fraction',
        'high_blood_pressure',
        'platelets',
        'serum_creatinine',
        'serum_sodium',
        'sex',
        'smoke',
        'time',
        'class'
        ]

        self.df=pd.read_csv(CSV_PATH, sep=',', header=None, na_values=["?"])
        #print(self.df.head())
        #print(self.df.shape)
        
    #Función para eliminar valores nulos. En nuestro caso no es necesario utilizarla porque no hay ningún dato nulo en el conjunto
    def elimina_ceros(self):
        #print(self.df.isnull().sum())
        self.df=self.df.dropna(axis=0)
        #print(self.df.isnull().sum())
        #print(self.df.columns)
        print(self.df[len(self.df.columns) -1].value_counts()) 
        

    def pandas2array(self):
    
        self.feature_df = self.df[self.df.columns[0 : len(self.df.columns) - 1]]
        print(self.feature_df.head())
        self.X_readed = np.asarray(self.feature_df)
        self.X_readed_names = self.X_readed
        self.X_readed = self.X_readed[1:]
        self.y_readed = np.asarray(self.df[len(self.df.columns) -1])
        self.y_readed = self.y_readed[1:]
       

    def oversampling(self, balanceo=True):
        print ('Conjunto original', self.X_readed.shape,  self.y_readed.shape)
        unique, counts = np.unique(self.y_readed, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # Sin balanceo de datos
        if balanceo == False:
            self.X = self.X_readed
            self.y = self.y_readed
            
        #Balanceo de datos
        if balanceo:
            sm = SMOTE(random_state=4)
            self.X, self.y= sm.fit_resample(self.X_readed, self.y_readed)

            print('\nBalanceado:', self.X.shape,  self.y.shape)
            unique, counts = np.unique(self.y, return_counts=True)
            print(dict(zip(unique, counts)))

    def normalizado(self):
        scaler = preprocessing.StandardScaler()
        print('Normalizando')
        print(self.X[0:5])
        scaler.fit(self.X) # fit realiza los cálculos y los almacena
        self.X = scaler.transform(self.X) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos

        print('\n\nTras normalizado')
        print(self.X[0:5])


    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state = 1)
        print ('Train set:', self.X_train.shape,  self.y_train.shape)
        print ('Test set:', self.X_test.shape,  self.y_test.shape)

    def pca(self, n=2, components = False, variance_max = 0.95):

        if components:
            mypca = PCA(n_components=n)
            mypca.fit(self.X)
            values_proj = mypca.transform(self.X)
            X_projected = mypca.inverse_transform(values_proj)
            loss = ((self.X - X_projected) ** 2).mean()
            print("Projection loss (2 components): " + str(loss))
            self.X = X_projected	
        if components == False:
            mypca = PCA()
            mypca.fit(self.X)
            mypca.explained_variance_ratio_
            mypca.explained_variance_ratio_.sum() 
            print("\n Varianza que aporta cada componente:")
            variance = mypca.explained_variance_ratio_
            print(variance)

            print("\n Varianza acumulada:")
            acumvar = []
            for i in range(len(mypca.explained_variance_ratio_)):
                if i==0:
                    acumvar.append(variance[i])
                else:
                    acumvar.append(variance[i] + acumvar[i-1])
            
            for i in range(len(acumvar)):
                print(f" {(i+1):2} componentes: {acumvar[i]} ")
        
        
#KNN
import time

def knn(k_n=4, busca_k= False,k_max=1000):

    Ks = np.linspace(1,k_max,k_max-1)

    #Para una K específica
    if busca_k == False:
        exactitudes = []
        k = k_n+1

        # Creamos nuestra instancia del modelo
        neigh = KNeighborsClassifier(n_neighbors = k)

        #Entrenamiento del modelo, llamando a su método fit 
        ini = time.time()
        neigh.fit(X_train,y_train)
        print(f"Tiempo entrenamiento = {(time.time() - ini)*1000:.3f} ms") 
        ini = time.time()
        y_predict_knn = neigh.predict(X_test)
        print(f"Tiempo predicción = {(time.time() - ini)*1000:.3f} ms") 
        print("Exactitud media obtenida con k-NN: ", neigh.score(X_test, y_test))

        #Cálculo de la exactitud
        exactitudes.append(neigh.score(X_test, y_test))

        #Matriz de confusión
        cm_kNN = confusion_matrix(y_test, y_predict_knn)
        disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN, display_labels=['Sobrevive','Fallece'])
        disp_knn.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de confusión k-NN")
        plt.savefig(GRAPH_PATH + "\Knn_matriz_confusion.jpg")


    #Para un rango de valores de K
    if busca_k:
        
        exactitudes = []
        ini = time.time()
        for i in range(1,k_max):
            # Creamos nuestra instancia del modelo
            neigh = KNeighborsClassifier(n_neighbors = i)

            #Entrenamiento del modelo, llamando a su método fit 
            
            neigh.fit(X_train,y_train)
            print(f"Tiempo entrenamiento = {(time.time() - ini)*1000:.3f} ms, knn -> {i-2}") 

            ini = time.time()
            y_predict_knn = neigh.predict(X_test)
            #translate_prediction(y_predict_knn[0:5], 5)
            print(f"Tiempo predicción = {(time.time() - ini)*1000:.3f} ms") 
            print("Exactitud media obtenida con k-NN: ", neigh.score(X_test, y_test))

            exactitudes.append(neigh.score(X_test, y_test))
    
        best_k = exactitudes.index(max(exactitudes))
        print(f'La exactitud más alta se ha logrado con un K de {best_k}, alcanzando un valor de {max(exactitudes)}')

        #Matriz de confusión
        cm_kNN = confusion_matrix(y_test, y_predict_knn)
        disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_kNN, display_labels=['Sobrevive','Fallece'])
        disp_knn.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de confusión k-NN")
        #plt.show()
        plt.savefig(GRAPH_PATH + "\Knn_matriz_confusion.jpg")

        plt.figure(figsize=(20,5))
        plt.plot(Ks,exactitudes, marker='o')  # range(1,Ks) -> ejex, mean_acc -> ejey
        plt.xlabel('Número de vecinos (k)')
        plt.ylabel('Exactitud')
        plt.savefig(GRAPH_PATH + "\Knn_exactitudes.jpg")
        plt.show()

#SVM 
     
def svm():
    # Creamos nuestra instancia del modelo
    C_2d_range = [1e-1, 1, 1e1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1, 1e2]
    classifiers = []
    exactitudes = []
    gammas = []
    list_c_gamma = []

    for C in C_2d_range:
        for gamma in gamma_2d_range:
            ini = time.time()
            clf = SVC(C=C, gamma = gamma, kernel="rbf")
            clf.fit(X_train,y_train)
            classifiers.append((C, gamma, clf))
            gammas.append(gamma)
                        #Entrenamiento del modelo, llamando a su método fit 
            
            
            print(f"Tiempo entrenamiento = {(time.time() - ini)*1000:.3f} ms") 

            
  
           
            
    max_score,max_c,max_gamma = 0,0,0
    print(f"\n     C        gamma    exactitud ")
    print(f"---------------------------------")
    for C,gamma,clf in classifiers:
        ini = time.time()
        score = clf.score(X_test, y_test)
        exactitudes.append(score)
        print(f"{C:10.2f} {gamma:8.2f} {score:10.5f}")
        print(f"Tiempo predicción = {(time.time() - ini)*1000:.3f} ms") 
        if score > max_score:
            max_score = score
            max_c = C 
            max_gamma = gamma

    c_gamma = f"\n\nLa mayor exactitud {max_score:5f} se obtiene con C={max_c} y gamma = {max_gamma}"    
    print(c_gamma)
    list_c_gamma.append(c_gamma)
   
    #Guardar el modelo con los mejores valores de C y gamma y entrenarlo
    from joblib import dump, load
    clf_to_save = SVC(kernel='rbf', gamma=max_gamma, C= max_c)
    clf_to_save.fit(X_train, y_train)
    print("Exactitud del modelo: ", clf_to_save.score(X_test,y_test))

    dump(clf_to_save, GENERAL_PATH + '\svm_cell_samples.joblib')
    clf_loaded = load(GENERAL_PATH + '\svm_cell_samples.joblib')
    y_predict_svm = clf_loaded.predict(X_test) 
    translate_prediction(clf_loaded.predict(X_test[0:5]), 5) 
    
    cm_svm = confusion_matrix(y_test, y_predict_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=['Sobrevive','Fallece'])
    disp_svm.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión SVM")
    plt.savefig(GRAPH_PATH + "\svm_matriz_confusion.jpg")
    plt.show()

       
def translate_prediction(y_predict, num):
    translate = ['Sobrevive', 'Fallece'] 

    for i in range(num):
        if y_predict[i] == '0':
            print(translate[0])
        else:
            print(translate[1])

if __name__ == "__main__":

    #Instanciar objeto dataset de la clase preparacion_dataset
    dataset = preparacion_dataset() 

    #Ejecución de los distintos métodos para procesar los datos del dataset
    dataset.lectura_datos()
    dataset.elimina_ceros()
    dataset.pandas2array()
    dataset.oversampling(balanceo= True)
    dataset.normalizado()
    #dataset.pca(n=11,components=TRUE, variance_max= 0.95)
    dataset.split_train_test()
    
    #Obtener los datos finales de entrenamiento y test ya procesados 
    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test
    
    print(X_test.shape)
    #Algoritmos de entrenamiento

    knn(k_n=26,busca_k= True, k_max=100)

    #svm()
import itertools
from pickle import TRUE
from tkinter import N
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import ShuffleSplit, KFold, cross_validate 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

GENERAL_PATH = "C:/Users/pablo/Desktop/ENTREGAS/entrega_2/"
CSV_PATH = GENERAL_PATH + "hour.csv"
GRAPH_PATH = GENERAL_PATH + "/graficas"

class preparacion_dataset:

    #Función lectura de datos
    def lectura_datos(self):
        self.df=pd.read_csv(CSV_PATH, sep=',', na_values=["?"])
        #self.df['dteday'] = self.df['dteday'].astype(int)
        #self.df['dteday'] = pd.to_numeric(self.df['dteday']
        self.df["dteday"] = pd.to_datetime(self.df["dteday"],format="%Y-%m-%d", yearfirst= True)
        #self.df["dteday"] = self.df["dteday"].apply(str)
        #self.df['dteday'] = self.df['dteday'].astype(int)
        self.df['dteday'] = pd.to_numeric(self.df['dteday'])
        #self.df["dteday"] = int(self.df.strftime("%Y%m%d"))
        #self.df["dteday"] = int(self.df["dteday"])
        
        
        print(self.df.head())
        print(self.df.shape)
        #conversion datatime
        
    #Función para eliminar valores nulos. En nuestro caso no es necesario utilizarla porque no hay ningún dato nulo en el conjunto
    def elimina_ceros(self):
        print(self.df.isnull().sum())
        self.df=self.df.dropna(axis=0)
        print(self.df.isnull().sum())
        #print(self.df[len(self.df.columns) -1].value_counts()) 
        #print(self.df['class'].value_counts())

    def pandas2array(self):
        self.feature_df = self.df[self.df.columns[1:len(self.df.columns)-1]]
        self.X_readed = np.asarray(self.feature_df)
        self.y_readed = np.asarray(self.df[self.df.columns[len(self.df.columns)-1]])
       
    
    def oversampling(self, balanceo=True):
        print ('Conjunto original', self.X_readed.shape,  self.y_readed.shape)
        unique, counts = np.unique(self.y_readed, return_counts=True)
        #print(dict(zip(unique, counts)))
        
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
        #print(self.X[0:5])
        scaler.fit(self.X) # fit realiza los cálculos y los almacena
        self.X = scaler.transform(self.X) # aplica los cálculos sobre el conjunto de datos de entrada para escalarlos

        print('\n\nNormalizado Realizado\n')


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
        
        
def plot_boxplot(model_list, figure_size = (10,5)):
    score_list=[]
    name_list = []
    for name,results in model_list:
        score_list.append(results['test_mse'])        
        name_list.append(name.replace(' ','\n'))
    plt.boxplot(score_list, labels=name_list)
    plt.ylabel('MSE')   
    plt.xlabel('Configuraciones de red')       
    plt.show()

def show_cv_results(results):
    print()
    print("                                 Modelos                           Test r2 mean   Test r2 desv   Test MSE mean   Test MSE desv")
    print("------------------------------------------------------------------------------------")
    for title, res in results:
        fit_time = res['fit_time']
        test_r2 = res['test_r2']
        test_mse = res['test_mse']
        print(f"{title:23}         {test_r2.mean():.3f}         (+/- {(test_r2.std()):.2f})         {test_mse.mean():.3f}        (+/- {(test_mse.std()):.2f})")
        print()

def copy_and_sort(element_list):
    sorted_list = element_list[:]
    sorted_list.sort()
    return sorted_list

def plot_expected_vs_predicted(expected_list, predicted_list, num_figures, cols=3, sp_right=1.9, sp_top=1.2, fig_size=(12,10)):
    num_rows = num_figures // cols + 1
    fignumber=1
    fig = plt.figure(figsize=fig_size)
    for i in range(len(predicted_list)):
        title = predicted_list[i][0]        
        predicted = predicted_list[i][1]
        expected = expected_list[i]
        plt.subplot(num_rows,cols,fignumber) # 1 - numrows, 2 - numcols, 1 - index
        plt.title("Valores esperados vs obtenidos\n"+title)
        plt.xlabel('Elementos')
        plt.ylabel('Valor')        
        plt.scatter(range(0,len(expected)),expected, marker='+',label='esperados') 
        plt.scatter(range(0,len(predicted)),predicted, marker='+',label='obtenidos') 
        plt.legend(loc='upper left')
        fignumber = fignumber + 1
    plt.subplots_adjust(right=sp_right, top=sp_top)
    plt.show()

def modelo_red_neuronal(activation_function = 'relu', tolerance = 1e-4, iterations = 4000, batch_size_1 = 32, batch_size_2 = 128, train = True ):
    
    print('\nEntrenando todas las arquitectura de red neuronal...\n')
    random_seed = 0
    layers = ()

    models = (('MLP(4,4) batch:'+str(batch_size_1) + ' tolerance:'+str(tolerance) + ' function:'+str(activation_function)+ ' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(4,4) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(8,8), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(8,8), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(16,16) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(16,16) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(4,4,4) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4,4), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(4,4,4) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4,4,), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8,8) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(8,8,8), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8,8) batch:' + str(batch_size_1) +' tolerance:'+ str(tolerance) + ' function:' + str(activation_function)+' iterations:'+ str(iterations),MLPRegressor(hidden_layer_sizes=(8, 8, 8), batch_size=batch_size_1, activation=activation_function,tol=tolerance, max_iter=iterations, random_state=random_seed, verbose=True)),
            ('MLP(16,16,16) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16,16), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(16,16,16) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16,16), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(4,4,4,4) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4,4,4), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(4,4,4,4) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(4,4,4,4), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8,8,8) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(8,8,8,8), batch_size=batch_size_1, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(8,8,8,8) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(8,8,8,8), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(16,16,16,16) batch:'+str(batch_size_1) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16,16,16), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)),
            ('MLP(16,16,16,16) batch:'+str(batch_size_2) +' tolerance:'+str(tolerance) +' function:' +str(activation_function)+' iterations:'+str(iterations),MLPRegressor(hidden_layer_sizes=(16,16,16,16), batch_size=batch_size_2, activation = activation_function, tol=tolerance, max_iter=iterations, random_state=random_seed, verbose= True)))
            
    score_dict = {'r2': 'r2',
                'mse': make_scorer(mean_squared_error)}

    results = []
    kf10 = KFold(n_splits=2, shuffle=True, random_state=0)

    if train:
        for (title,model) in models:
            print(title)
            res_kf10= cross_validate(model, X, y, cv=kf10, return_estimator=True, return_train_score=True, scoring=score_dict)
            results.append((title,res_kf10))

        return results, kf10
    

def mejor_modelo(X,y, results, kf5):
    max_score_list = []
    for title,res in results:
        test_score = res['test_r2']
        max_score_list.append((test_score.max(),test_score.argmax()))
    max_score_list

    test_sets = []
    for train_index, test_index in kf5.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        test_sets.append((X_test,y_test))
    # Gráfica que representa los valores obtenidos con la predicción frente a los esperados con el mejor resultado
    predicted = []
    expected = []
    model= 0
    for title,result in results:
        estimators = result['estimator']
        i = max_score_list[model][1]
        estimator = estimators[i]
        y_predict = estimator.predict(test_sets[i][0])
        predicted.append((title,copy_and_sort(y_predict)))
        expected.append(copy_and_sort(test_sets[i][1]))
        model=model+1
    num_graficas = len(predicted)
    plot_expected_vs_predicted(expected, predicted, num_graficas, cols=2, sp_right=1, sp_top=2, fig_size=(20,10))

def probador_hiperparametros():
    batch_size = [16,64]
    iterations = [1000,4000]
    activation_function = ['relu']
    tol = [1e-3, 1e-4]
    resultados = []
    kf10s = []
    for bs in batch_size:
        for it in iterations:
            for af in activation_function:
                for tole in tol:
                    results, kf10 = modelo_red_neuronal(batch_size_2 =bs, iterations=it, activation_function=af, tolerance=tole)
                    resultados.append(results)
                    kf10s.append(kf10)
    # Evaluación de resultados
    for results, kf10 in zip(resultados, kf10s):
        #plot_boxplot(results)
        show_cv_results(results)
        #mejor_modelo(X,y, results, kf10)


if __name__ == "__main__":

    #Instanciar objeto dataset de la clase preparacion_dataset
    dataset = preparacion_dataset() 

    #Ejecución de los distintos métodos para procesar los datos del dataset
    dataset.lectura_datos()
    dataset.elimina_ceros()
    dataset.pandas2array()
    dataset.oversampling(balanceo= False)
    dataset.normalizado()
    #dataset.pca(n=11,components=TRUE, variance_max= 0.95)
    dataset.split_train_test()
    
    #Obtener los datos finales de entrenamiento y test ya procesados 
    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test

    #Obtener X e Y 
    X = dataset.X
    y = dataset.y
    
    #Algoritmos de entrenamiento

    resultados, kf10s = probador_hiperparametros()

    '''
    #valores definitivos hiperparametro
    bs = 16
    it = 1000
    af = 'relu'
    tole = 1e-1
    results, kf10 = modelo_red_neuronal(batch_size_2=bs, iterations=it, activation_function=af, tolerance=tole)
    
    #Evaluación de resultados
    plot_boxplot(results)
    show_cv_results(results)
    mejor_modelo(X,y, results, kf10)
    '''

    

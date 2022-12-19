#Model Evaluation Tools
lvar = ['o3', 'no', 'no2', 'nox','rh','tmp','wdr','wsp']

def cargar_resultados(estacion):
    import pandas as pd
    import numpy as np
    evaluacion = {}
    with open(f'evaluacion_{estacion}.txt') as file:
        for line in file:
            if '>' in line:
                corrida = line.replace('>','')[:-1]
                evaluacion[corrida] = {}
                print(line,end='')
                hp,R2,MSE,RMSE,MAE,MAPE = [],[],[],[],[],[]
            else:
                #print(line)
                aux = line.split()[0:4]
                L2 = line.split()[4:]
                l1 = f'{aux[0]}{aux[1]} {aux[2]}{aux[3]}'.split()
                new = l1+L2
                for elemento in new:
                    k,v = elemento.split(':')
                    #print(f'{k}-{v}',end=' ')    

                aux = line.split()[0:4]
                L2 = line.split()[4:]
                l1 = f'{aux[0]}{aux[1]} {aux[2]}{aux[3]}'.split()
                new = l1+L2
                for elemento in new:
                    k,v = elemento.split(':')
                    if   'R2'   == k: R2.append(float(v))
                    elif 'MSE'  == k: MSE.append(float(v))
                    elif 'RMSE' == k: RMSE.append(float(v))
                    elif 'MAE'  == k: MAE.append(float(v))
                    elif 'MAPE' == k: MAPE.append(float(v))
                    else: hp.append(v)
                datos = {'HP':hp,'R2':R2,'MSE':MSE,'RMSE':RMSE,'MAE':MAE,'MAPE':MAPE}
                evaluacion[corrida]= pd.DataFrame(data=datos) 
    return evaluacion

def indexByDate(df):
    '''indexa el dataframe en base la fecha y hora'''
    from datetime import datetime
    from datetime import timedelta
    import pandas as pd
    fechas = []
    for index, row in df[['fecha','hora']].iterrows():
        if row["hora"] !=24:
            fechas.append(datetime.strptime( f'{row["fecha"]} {row["hora"]}',"%Y-%m-%d %H" ))
        else:
            fecha,hora =row["fecha"]+timedelta(days=1),0
            fechas.append(datetime.strptime( f'{fecha} {hora}',"%Y-%m-%d %H" ))
    df_aux = df.set_index(pd.Series(fechas))
    year = df['fecha'][0].year
    f0 = datetime.strptime(f'{year}-01-01 0','%Y-%m-%d %H')
    f1 = f'{year}-01-01 01:00:00'
    df_aux.loc[f0] = df_aux.loc[f1]
    df_aux = df_aux.sort_index()
    df_aux = df_aux.loc[f'{year}0101':f'{year}1231']
    return df_aux.iloc[:,2:]

def byDay(df,v):
    import pandas as pd
    indice = df.index
    registros = []
    fechas = []
    for d in range(int(len(indice)/24)):
        datos = []
        for i in indice[24*d:24*(d+1)]:
            datos.append(df.loc[i][v])
        registros.append(datos)
        fechas.append(i.date())
        dfa = pd.DataFrame(registros,index=pd.DatetimeIndex(fechas))
        dfa.index.freq='D'
    return dfa

def describe(df):
    import pandas as pd
    mean = []
    median = []
    mode = []
    std = []
    cv = []
    vmax = []
    vmin = []
    for c in df.columns:
        mean.append( df.iloc[:,c].mean() )
        median.append( df.iloc[:,c].median() )
        mode.append( df.iloc[:,c].mode().values )
        std.append( df.iloc[:,c].std() )
        vmax.append(df.iloc[:,c].max())
        vmin.append(df.iloc[:,c].min())
        cv.append(df.iloc[:,c].std()/df.iloc[:,c].mean())
    
    data = {'mean':mean,'median':median,'mode':mode,'std':std,'max':vmax,'min':vmin,'cv':cv}
    return pd.DataFrame(data=data,index=list(df.columns))


def split_data(df,train_size=.75,seed=None,outlier_remove=False):
    import pandas as pd
    df = df[generar_lista(iterator=False)].dropna()
    train,test   = train_test_split( df,train_size=train_size, random_state=seed ) 
    X_train,y_train = train.iloc[:,:-1],train.iloc[:,-1]
    X_test,y_test = test.iloc[:,:-1],test.iloc[:,-1]   
    
    if outlier_remove:
            dfs_train = []
            dfs_test  = []
            for hora in range(24):
                rango_hora = df[df.hour == hora]
                Q1 = rango_hora.o3.quantile(.25)
                Q3 = rango_hora.o3.quantile(.75)
                IQR = Q3-Q1
                MaxV = Q3 + 1.5 * IQR
                MinV = Q1 - 1.5 * IQR
                train_hora = train[train.hour == hora]
                test_hora  = test[test.hour == hora]
                dfs_train.append(train_hora[ (train_hora.o3 >= MinV) & (train_hora.o3 <= MaxV)  ])
                dfs_test.append(test_hora[ (test_hora.o3 >= MinV) & (test_hora.o3 <= MaxV)  ])

            traino = pd.concat(dfs_train).sample(frac=1)
            testo  = pd.concat(dfs_test).sample(frac=1)                      
            Xo_train,yo_train = traino.iloc[:,:-1],traino.iloc[:,-1]
            Xo_test,yo_test = testo.iloc[:,:-1],testo.iloc[:,-1]   

            return X_train,y_train,X_test,y_test,Xo_train,yo_train,Xo_test,yo_test

    return X_train,y_train,X_test,y_test

    
def remove_outlier(df):
    import pandas as pd
    dfs = []
    for hora in range(24):
        rango_hora = df[df.hour == hora]
        Q1 = rango_hora.o3.quantile(.25)
        Q3 = rango_hora.o3.quantile(.75)
        IQR = Q3-Q1
        MaxV = Q3 + 1.5 * IQR
        MinV = Q1 - 1.5 * IQR
        dfs.append(rango_hora[ (rango_hora.o3 >= MinV) & (rango_hora.o3 <= MaxV)  ])
    return pd.concat(dfs).sort_index()


def generar_df(df,variables = lvar,tiempos = [1,4,8,12,24]):
    import pandas as pd
    aux= df[variables].copy()
    tiempos = [1,4,8,12,24]
    for variable in variables:
        for t in tiempos:
            aux[f'{variable}-{t}'] = aux[variable].shift(periods=t) 
    aux['date']    = aux.index
    aux['month'] = aux['date'].dt.month
    aux['weekday'] = aux['date'].dt.weekday
    aux['hour']    = aux['date'].dt.hour
    del(aux['date'])
    return aux

def generar_lista(variables = lvar,tiempos = [1,4,8,12,24],iterator=True):
    if iterator:
        aux = [[f'{var}-{t}' for var in variables ] for t in  tiempos ] 
        lista2 = ['month','weekday','hour']
        for lista in aux:
            lista+=lista2
        return  aux
    else:
        aux = [f'{var}-{t}' for t in  tiempos for var in variables] 
        lista2 = ['month','weekday','hour','o3']
        aux+=lista2
        return  aux

def probar(df):
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm  import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    from ngboost import NGBRegressor
    from sklearn.model_selection import train_test_split

    df_aux = generar_df(df)
    df_aux = df_aux[generar_lista(iterator=False)].dropna()
    lista_atributos = generar_lista()
    train,test = train_test_split( df_aux ) 
    
    for atributos in lista_atributos:
        train_ = train[atributos]
        test_  = test[atributos]
        X_train,y_train = train_.iloc[:,:-1],train_.iloc[:,-1]
        X_test,y_test = test_.iloc[:,:-1],test_.iloc[:,-1]  
        depth= range(1,21)
        print(f'{atributos}')
        for d in depth:
            regressor = modelos['RF'](max_depth=d,random_state=0) 
            R2,MSE,RMSE,MAE,MAPE = evaluar_modelo(regressor,X_train,y_train,X_test,y_test)
            print(f'depth: {d} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}')
    
    for modelo in modelos:
        modelo()
        break
        
def evaluar_modelo(model,X_train,y_train,X_test,y_test,scaling=None):
    from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    import pandas as pd
    import numpy as np
    
    if scaling== None:
        model.fit(X_train,y_train)
        y_predicted = model.predict(X_test)
    elif scaling=='MinMax':
        scaler_X_train = MinMaxScaler()
        scaler_y_train = MinMaxScaler()
        X_train = scaler_X_train.fit_transform(X_train)
        y_train = scaler_y_train.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        X_test  = scaler_X_train.transform(X_test)
        model.fit(X_train,y_train)
        y_predicted = scaler_y_train.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    elif scaling=='Zscore':
        scaler_X_train = StandardScaler()
        scaler_y_train = StandardScaler()      
        X_train = scaler_X_train.fit_transform(X_train)
        y_train = scaler_y_train.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        X_test  = scaler_X_train.transform(X_test)
        model.fit(X_train,y_train)
        y_predicted = scaler_y_train.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    
    MSE = mean_squared_error(y_test,y_predicted)       
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(y_test,y_predicted) 
    SSres = np.sum((y_test-y_predicted)**2)
    SStot = np.sum((y_test-y_test.mean())**2)
    R2 = 1 - SSres/SStot
    MAPE = mean_absolute_percentage_error(y_test,y_predicted)
    return R2,MSE,RMSE,MAE,MAPE

def random_forest(X_train,y_train,X_test,y_test,scaling=None,save=False,mode='a',filename='OUT.txt'):
    import pandas as pd
    import numpy as np
    lista_atributos = generar_lista(tiempos = [24])
    label = 'Originales'
    if save: file = open(filename,mode)
    for atributos in lista_atributos:
        depth= range(1,21)
        if scaling: label = scaling 
        if save: file.write(f'>random_forest_{label}\n')
        print(f'>random_forest_{label}')
        for d in depth:
            regressor = modelos['RF'](max_depth=d,random_state=0) 
            R2,MSE,RMSE,MAE,MAPE = evaluar_modelo(regressor,X_train[atributos],y_train,X_test[atributos],y_test,scaling=scaling)
            print(f'depth: {d} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}')
            if save: file.write(f'depth: {d} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}\n')
    if save: file.close()
            
def k_nearest_neighbor(X_train,y_train,X_test,y_test,scaling=None,save=False,mode='a',filename='OUT.txt'):
    import pandas as pd
    import numpy as np
    lista_atributos = generar_lista(tiempos = [24])
    label = 'Originales'
    if save: file = open(filename,mode)
    for atributos in lista_atributos:
        neighbors= range(1,21)
        if scaling: label = scaling 
        if save: file.write(f'>k_nearest_neighbor_{label}\n')
        print(f'>k_nearest_neighbor_{label}')
        for n in neighbors:
            regressor = modelos['KNN'](n_neighbors=n) 
            R2,MSE,RMSE,MAE,MAPE = evaluar_modelo(regressor,X_train[atributos],y_train,X_test[atributos],y_test,scaling=scaling)
            print(f'n_neighbors: {n} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}')
            if save: file.write(f'n_neighbors: {n} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}\n')
    if save: file.close()            
        
def support_vector_machine(X_train,y_train,X_test,y_test,scaling=None,save=False,mode='a',filename='OUT.txt'):
    import pandas as pd
    import numpy as np
    lista_atributos = generar_lista(tiempos = [24])
    label = 'Originales'
    if save: file = open(filename,mode)
    for atributos in lista_atributos:
        kernels= 'linear', 'poly', 'rbf'
        if scaling: label = scaling 
        if save: file.write(f'>support_vector_machine_{label}\n')
        print(f'>support_vector_machine_{label}')
        for k in kernels:
            regressor = modelos['SVR'](kernel=k) 
            R2,MSE,RMSE,MAE,MAPE = evaluar_modelo(regressor,X_train[atributos],y_train,X_test[atributos],y_test,scaling=scaling)
            print(f'kernel: {k.ljust(6," ")} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}')
            if save: file.write(f'kernel: {k.ljust(6," ")} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}\n')
    if save: file.close()

def gradient_boosting(X_train,y_train,X_test,y_test,scaling=None,save=False,mode='a',filename='OUT.txt'):
    import pandas as pd
    import numpy as np
    lista_atributos = generar_lista(tiempos = [24])
    label = 'Originales'
    if save: file = open(filename,mode)
    for atributos in lista_atributos:
        depths= range(1,21)
        if scaling: label = scaling 
        if save: file.write(f'>gradient_boosting_{label}\n')
        print(f'>gradient_boosting_{label}')
        for d in depths:
            regressor = modelos['GB'](n_estimators = 200, max_depth = d, random_state = 0)
            R2,MSE,RMSE,MAE,MAPE = evaluar_modelo(regressor,X_train[atributos],y_train,X_test[atributos],y_test,scaling=scaling)
            print(f'depths: {d} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}')
            if save: file.write(f'depths: {d} R2: {R2:.3f}\t MSE:{MSE:.3f}\tRMSE:{RMSE:.3f}\tMAE:{MAE:.3f}\tMAPE:{MAPE:.3f}\n')
    if save: file.close()            
    
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm  import SVR
from sklearn.ensemble import GradientBoostingRegressor
modelos = {'RF':RandomForestRegressor, 
           'KNN':KNeighborsRegressor, 
           'SVR':SVR, 
           'GB':GradientBoostingRegressor}
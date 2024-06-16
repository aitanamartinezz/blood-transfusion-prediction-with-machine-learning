# ENTREGA 1 AA1: PROBLEMA DE CLASIFICACIÓN
# Aitana Martínez Rey y Patricia da Cocepción Sarrate
# INSTALACIÓN DE PAQUETES
######################################################
# Package Install
#begin
#    import Pkg;
#    Pkg.add("XLSX");
#    Pkg.add("FileIO");
#    Pkg.add("JLD2");
#    Pkg.add("Flux");
#    Pkg.add("ScikitLearn");
#    Pkg.add("Plots");
#    Pkg.add("MAT");
#    Pkg.add("Images");
#    Pkg.add("DelimitedFiles");
#    Pkg.add("CSV");
#end
#######################################################

# ¿CON QUE BASE TRABAJAREMOS?
# Trabajamos con la base de datos "transfusion.data"
# Una base que contiene datos sobre distintas
# transfusiones sanguineas.
# ¿CUÁNTOS PATRONES SE TIENE?
#print(size(dataset1))
#print(size(dataset1, 1))
# Como vemos la base cuenta con 748 patrones.
# ¿CUÁNTOS ATRIBUTOS TIENE CADA PATRÓN? ¿SON RELEVANTES?
#print(size(dataset1, 2))
# Cada patrón consta de 5 atributos. A nuestro parecer
# todos ellos tiene relevancia para la resolución del
# problema.
# QUE DESCRIBE CADA ATRIBUTO?
# 1. Recency (months): Meses desde la última donación (NUMÉRICA)
# 2. Frequency (times): Número total de donaciones de una persona (NUMÉRICA)
# 3. Monetary (c.c. blood): Total de sangre donada en c.c. (NUMÉRICA)
# 4. Time (months): Meses desde la primera donación (NUMÉRICA)
# 5. "whether he/she donated blood in March 2007": SI la persona
# donó sangre en marzo del 2007 o no (VARIABLE CATEGÓRICA)
# ¿HAY ENTRADAS O SALIDAS CATEGÓRICAS?
# Sí, el atributo whether he/she donated blood in March 2007
# es una variable categórica unidimensional binaria, cuyo
# resultado dependerá de los valores numéricos de entrada.
# A la hora de procesarla se deberían convertir sus valores
# (al ser binaria) a 0 y 1, uno por cada posibilidad del atributo.
# En este caso, los valores ya vienen implementados en la BD
# como 0 y 1 .

#############################################################
# CARGAMOS LA BASE DE DATOS
using DelimitedFiles
dataset1 = readdlm("transfusion.data",',');
# SEPARAMOS LAS ENTRADAS DE LAS SALIDAS
inputs1 = dataset1[:,1:4];
targets1 = dataset1[:,5];

# TIPOS DE DATOS INICIALES
print(typeof(dataset1))
print(typeof(inputs1))
print(typeof(targets1))

# CONVERTIMOS LAS MATRICES DE ENTRADA Y DE SALIDA AL TIPO FLOAT32
inputs1 = convert(Array{Float32,2},inputs1);
targets1 = convert(Array{Float32,1},targets1);

#Comprobamos el tipo de elementos que hemos tranformado
print(typeof(inputs1))
print(typeof(targets1))


# Realizamos una función para  la codificación de la variable categórica

function codificacion(targets)
    if length(unique(targets))==2
        a=replace(targets,unique(targets)[1]=>0, unique(targets)[2]=>1)
    else
        #Comprobamos si el codigo es una matriz o no
        f=isa(targets,Matrix)
        #Creamos una matriz booleana con los diferentes tipos de datos
        a= zeros(Bool, f ? length(unique(targets)) : length(targets), f ? length(targets) : length(unique(targets)))
        for(i,j) in enumerate(unique(targets))
            if f
                a[index, getindex.(findall(isequal(j), targets), 2)] .= 1
            else
                a[findall(isequal(j), targets), i] .= 1
            end
        end
        return(a)
    end
end




####################################################################3
# Como nuestra BD solo consta de dos resultados diferentes en el atributo categórico
# probaremos la función con Iris.data que consta de 3 valores distintos para su
# variable categórica.
using DelimitedFiles
dataset2 = readdlm("Iris.data",',');
inputs2 = dataset2[:,1:4];
targets2 = dataset2[:,5];

inputs2 = convert(Array{Float64,2},inputs2);
targets2 = convert(Array{String,1},targets2);

print(codificacion(targets2)) # Vemos que funciona correctamente

###################################################################################
#Observamos que en nuestro caso al ser una base de datos binaria nos devuelve una matriz de tipo float32
#Realizamos la comprobacion con los datos de setosa para mas de dos variables categóricas
#En este caso nos devuelve una matriz booleana

#Realizamos el análisis de las matrices de entrada y salida de nuestra base de datos

println("Matriz de entradas antes de codificar: ", size(inputs1,1), "x", size(inputs1,2));
println("Longitud deseada del vector de salidas antes de codificar: ", length(targets1));
targets1 = codificacion(targets1);
println("Matriz de salidas despues de codificar: ", size(targets1,1), "x", size(targets1,2));

# Mismo número de filas entre las dos matrices
@assert (size(inputs1,1)==size(targets1,1)) "Las matrices de entradas y salidas concuerdan en el número de filas"


#######################################
# NORMALIZACIÓN
using Statistics
maxi= maximum(inputs1, dims=1)
mini = minimum(inputs1, dims=1)
dt= std(inputs1, dims=1)
m= mean(inputs1, dims=1)

# DATOS NORMALIZADOS MEDIANTE 2 MÉTODOS:

# NORMALIZACIÓN ENTRE MÁXIMO Y MÍNIMO
# MENOS ROBUSTA A VALORES ATÍPICOS, POR LO CUAL NO LA USAREMOS. SOLO LA INDICAMOS.
# norm1 = ((inputs1.-mini)./(maxi-mini))

# Para la normalización de MEDIA 0 hay que contemplar el siguiente caso:
# ¿Y SI ALGÚN ATRIBUTO TIENE DESVIACIÓN TÍPICA IGUAL A 0?
# Todos los patrones toman el mismo valor para un atributo

dt_0=findall(x->x==0, dt)
# En este caso NO EXISTEN POSICIONES CONFLICTIVAS.
# ¿PERO Y SI LAS HUBIESE?
#SOLUCIÓN: BORRAR LOS ATRIBUTOS QUE NO cAPORTEN INFORMACIÓN
function desvtipic_0(inputs1)
    if length(dt_0)!=0
        inputs1 = replace(inputs1, dt_0=>0)
        m = replace(m, dt_0=>0)
        dt = replace(dt, dt_0=>0)
    end
end
# NORMALIZACIÓN DE MEDIA 0 (robusta contra valores atípicos)
norm2 = (inputs1.-m)./dt
println(norm2)



####################################################################################
####################################################################################
# ENTREGA 2 AA1: PROBLEMA DE CLASIFICACIÓN SUPERVISADA
using Statistics
using Flux
using Flux.Losses

# Función oneHotEncoding para la codificación
# Esta recibe 2 PARÁMETROS:
# CLASSES: VALORES DE LAS CATEGORÍAS
# FEATURE: VALORES DE SALIDA DESEADOS PARA CADA PATRÓN
# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::AbstractArray{<:Any}, classes::AbstractArray{<:Any,1})
    #@assert(all([in(value, classes) for value in feature]));
    b=length(classes) # Longitud del vector classes
    # Si es igual a 2
    if b==2
        # Se convierte en un vector de valores booleanos
        OHE=replace(feature,classes[1]=>0, classes[2]=>1)

        # Se convierte en una matriz bidimensional de una columna
        OHE=reshape(OHE,length(feature),1)

    # Si es mayor que 2
    else

        f=isa(feature,Matrix)
        #Creamos una matriz booleana con los diferentes tipos de datos
        OHE= zeros(Bool, f ? length(classes) : length(feature), f ? length(feature) : length(classes))
        OHE=convert(Array{Bool,2},OHE)
        # Se itera sobre cada coluna y se asignan los valores de esa columna como resultado
        # de comparar feature con su correspondiente categorñia.
        for(i,j) in enumerate(classes)
            if f
                OHE[index, getindex.(findall(isequal(j), feature), 2)] .= 1
            else
                OHE[findall(isequal(j), feature), i] .= 1
            end
        end
        return OHE;
    end
end


print(oneHotEncoding(targets1,unique(targets1)))
print(oneHotEncoding(targets2,unique(targets2)))


# SOBRECARGA DE OHE
# Se realiza llamando a la funcion anterior sin especificar las clases sustituyendola por (unique(feature))
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature::AbstractArray{<:Any,1}, unique(feature));
#Por si le pasamos un parametro booleano
#  En este caso, el propio vector ya está codificado
oneHotEncoding(feature::Array{Bool,1})=feature;
# Esta funcion tiene un patron de solo dos posibilidades. Al ser asi devuelve una
# matriz de solo una columna


#____________________________SERIES DE FUNCIONES PARA NORMALIZACIÓN DE DATOS__________________________________-#
# FUNCIÓN 1: Función calculateMinMaxNormalizationParameters que recibe un parámetro AbstractArray{<:Real,2}
# y devuelva una tupla con 2 valores (cada uno será una matriz con una fila) con los min y max de cada columna.
calculateMinMaxNormalizationParameters(dataset1::AbstractArray{<:Real,2}; dataInRows=true)=
(minimum(dataset1, dims=(dataInRows ? 1 : 2)), maximum(dataset1, dims=(dataInRows ? 1 : 2)));

# FUNCIÓN 2: Función calculateZeroMeanNormalizationParameters que recibe un parámetro AbstractArray{<:Real,2}
# y devuelvan una tupla con dos valores (cada uno de ellos será una matriz con una fila) con las medias y desv.
# típicas para cada columna.
calculateZeroMeanNormalizationParameters(dataset1::AbstractArray{<:Real,2}; dataInRows=true)=
(mean(dataset1, dims=(dataInRows ? 1 : 2)), std(dataset1, dims=(dataInRows ? 1 : 2)) );


function normalizeMinMax!(dataset1::Matrix{Float32}, NormParameters::Tuple{Matrix{Float32}, Matrix{Float32}}; dataInRows=true)
    minimo = NormParameters[1];
    maximo = NormParameters[2];
    dataset1 .-= minimo;
    dataset1 ./= (maximo .- minimo);
end

# FUNCIÓN 3: Función de mismo nombre que la anterior pero con solo el parámetro  y calcule
# los parámetros de normalización llamando a la función anterior
normalizeMinMax!(dataset1::Array{Float32,2}; dataInRows=true) = normalizeMinMax!(dataset1, calculateMinMaxNormalizationParameters(dataset1; dataInRows=dataInRows); dataInRows=dataInRows);
# En estas funciones se modifica la matriz de valores como parámetro.

# FUNCIÓN 4: Igual que la primera pero sin modificar la matriz de entrada, creando una nueva.
# función copy

function normalizeMinMax(dataset1::Matrix{Float32}, normParameters::Tuple{Matrix{Float32}, Matrix{Float32}}; dataInRows=true)
    DATASET = copy(dataset1);
    normalizeMinMax!(DATASET, normParameters; dataInRows=dataInRows);
    return DATASET;
end;

# FUNCIÓN 5: Igual que la segunda pero sin modificar la matriz de entrada, creando una nueva.

normalizeMinMax(dataset1::Array{Float32,2}; dataInRows=true)= normalizeMinMax(dataset1, calculateMinMaxNormalizationParameters(dataset1; dataInRows=dataInRows);dataInRows=dataInRows);


###############################


# 4 FUNCIONES ANÁLOGAS CON  NORMALIZACIÓN DE MEDIA 0
# 1
function normalizeZeroMean!(dataset1::Matrix{Float32}, normParameters::Tuple{Matrix{Float32}, Matrix{Float32}}; dataInRows=true)
    avg  = normParameters[1];
    dt = normParameters[2];
    dataset1 .-= avg;
    dataset1 ./= dt;

end;
# 2
normalizeZeroMean!(dataset1::AbstractArray{<:Real,2}; dataInRows=true) = normalizeZeroMean!(dataset1, calculateZeroMeanNormalizationParameters(dataset1; dataInRows=dataInRows); dataInRows=dataInRows);
# 3
function normalizeZeroMean(dataset1::AbstractArray{<:Real,2}, normParameters::Tuple{Matrix{Float32}, Matrix{Float32}}; dataInRows=true)
    DATASET = copy(dataset1);
    normalizeZeroMean!(DATASET, normParameters; dataInRows=dataInRows);
    return DATASET;
end;
# 4
normalizeZeroMean(dataset1::AbstractArray{<:Real,2}; dataInRows=true) = normalizeZeroMean(dataset1, calculateZeroMeanNormalizationParameters(dataset1; dataInRows=dataInRows); dataInRows=dataInRows);

# FUNCION CLASSIFYOUTPUTS, QUE reciba un parámetro outputs con las salidas de un modelo
# con un patrón por fila y lo convierta en una matriz de valores booleans  que cada
# fila solo tenga un valor a true.

function classifyOutputs(outputs::AbstractArray{<:Real,2}; dataInRows::Bool=true, threshold::Float64=0.5)
    colOutputs = size(outputs, dataInRows ? 2 : 1); # Número de columnas de outputs
    # El num de columnas debe ser distinto de 2
    @assert(colOutputs!=2)
        # Si tiene una columna
    if colOutputs==1
        # Comparamos la matriz con un parámetro "threshold" para generar la matriz de valores booleanos
        return convert(Array{Bool,2}, outputs.>=threshold);
    # Más de 1
    else
        # Buscamos el valor mayor de cada instancia
        #  FUNCIÓN FINDMAX
        (_,indicesMaxEachInstance) = findmax(outputs, dims= 2);
        #  Matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs1 = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        return outputs1;
    end;
end;

#______________________________FUNCIÓN ACCURACY___________________________________#

# 1: Precisión es el valor promedio de la comparación de targets y outputs
accuracy(outputs::AbstractArray{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);
# 2: Esta vez con matrices bidimensionales
function accuracy(outputs::Array{Bool,2}, targets::AbstractArray{Bool,2}; dataInRows::Bool=true)
   if (dataInRows)
       # Solo 1 columna
       if (size(targets,2)==1)
           # LLamada a la función anterior con la primera columna de cada vector
           return accuracy(outputs[:,1], targets[:,1]);
           # Más de 2
       else
           # Se comparan ambas matrices
           classComparison = targets .== outputs
           # Miramos si todos los valores son igual a true
           correctClassifications = all(classComparison, dims=2)
           # Devolvemos la media
           return mean(correctClassifications)
       end;
   else
       # Solo 1 fila  (matrices traspuestas)
       if (size(targets,1)==1)
           return accuracy(outputs[1,:], targets[1,:]);
       else
           classComparison = targets .== outputs
           correctClassifications = all(classComparison, dims=1)
           return mean(correctClassifications)
       end;
   end;
end;


# 3. Las salidas son reales  que no se han interpretado todavía como valores de pertenencia
#AbstractArray{<:Real,1}
accuracy(outputs::Array{Float32,1}, targets::AbstractArray{Bool,1}; threshold::Float64=0.5) = accuracy(AbstractArray{Bool,1}(outputs.>=threshold), targets);
# 4. Igual que la anterior pero con matrices bidimensionales.
#AbstractArray{<:Real,2}
function accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool=true)

    if (dataInRows)
        #Una columna
        if (size(targets,2)==1)
            # LLamada a accuracy
            return accuracy(outputs[:,1], targets[:,1]);
        # Más de 2
        else
            # LLamada a classify outputs para convertirla en valores booleanos
            return accuracy(classifyOutputs(outputs; dataInRows=true), targets);
        end;
    else
        # Una fila (matrices traspuestas)
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=false), targets);
        end;
    end;
end;


#________________________FUNCIONES RELATIVAS A LAS RNA_______________________________________#

# FUNCIÓN RNAChain: Crea un RRNNAA para resolver los problemas de clasificación.
function RNAChain(numInputs::Int64, topology::AbstractArray{<:Int,1}, numOutputs::Int64)
    ann = Chain();
    numInputsLayer = numInputs
    for numOutputLayers = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
        numInputsLayer = numOutputLayers;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;


# FUNCIÓN TrainChain: Entrena la RNA.
function trainChain(topology::AbstractArray{<:Int,1}, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1)
    # MIsmo número de columnas en vectores de entrada y salida
    # Cada patrón en una fila
    @assert(size(inputs,1)==size(targets,1));
    # Creamos la RNA
    ann = RNAChain(size(inputs,2), topology, size(targets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Almacenamos los valores totales de Loss y Accuracy(precision)
    LossTotal = Float64[];
    AccuracyTotal = Float64[];
    # Empezamos en el ciclo número 01
    Epoch = 0;
    # CÁLCULO DEL VALOR DE LOSS.
    # Las matrices son traspuestas
    Loss = loss(inputs', targets');
    # SALIDA DEL RNA.
    # MATRIZ DE ENTRADA TRASPUESTA
    outputs = ann(inputs');
    # Habrá distintas formas de pasar los datos:
    # FILAS
    accu = accuracy(Array{Float32,2}(outputs'), targets; dataInRows=true);
    # COLUMNAS
    accu = accuracy(Array{Float32}(outputs'), Array{Bool,2}(targets'); dataInRows=false);

#________________________________RESULTADOS_____________________________________________
    # Imprimimos por pantalla los valores de loss y precisión en porcentaje
    println("Epoch/Ciclo ", Epoch, ": Valor de loss: ", Loss, ", Precision: ", accu, ": Porcentaje de loss: ",Loss*100," %", ": Porcentaje de accuracy: ", accu*100, " %");
    #  ALMACENAMOS LOS RESULTADOS
    push!(AccuracyTotal, accu);

    # Entrenamos hasta que se cumpla ALGUNA condicion de parada
    # o BIEN SE LLEGUE AL MÁXIMO DE CICLOS O BIEN LOSS SEA SEA MENOS O IGUAL QUE MIN LOSS
    while (Epoch<maxEpochs) && (Loss>minLoss)
        # ENTRENAMIENTO DE UN CICLO CON MATRICES TRASPUESTAS.
        # Le sumamos  1 al contador de ciclos
        Epoch += 1;

        #  ALMACENAMOS LOS RESULTADOS
        push!(LossTotal, Loss);
        push!(AccuracyTotal, accu);
    end;
    return (ann, LossTotal, AccuracyTotal);
end;



# Sobrecarga de la función, que transforma targets en un vector columna.

function trainChain(topology::AbstractArray{<:Int,1}, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,1}; maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1)
    targets = reshape(targets,1)
    return trainChain(topology,inputs, targets, maxEpochs, minLoss, learningRate)
end;

################PRUEBA###########################################
#Comprobamos que el código funciona correctamente.

topology = [3,2]; # Dos capas ocultas con 3 neuronas la primera y 2 la segunda
learningRate = 0.05; # Tasa de aprendizaje
MaxEpochspossible = 1000; # Numero maximo de ciclos de entrenamiento
# CARGAMOS LA BASE DE DATOS

using DelimitedFiles
dataset11 = readdlm("transfusion.data",',');
# SEPARAMOS LAS ENTRADAS DE LAS SALIDAS
inputs1 = dataset11[:,1:4];
inputs1 = convert(Array{Float32,2},inputs1);
typeof(inputs1)
print(inputs1)
# NORMALIZACIÓN MIN MAX
newInputs1 = normalizeMinMax(inputs1)
print(newInputs1)

# NORMALIZACIÓN DE MEDIA 0
newInputs2 = normalizeZeroMean(inputs1)
print(newInputs2)

#Finalmente, normalizamos las entradas entre maximo y minimo:
newInputs = normalizeMinMax!(inputs1);

# Y creamos y entrenamos la RNA con los parametros dados
newInputs=convert(Matrix{Float32},newInputs)
targets11 = oneHotEncoding(targets1,unique(targets1))
targets11=convert(Matrix{Bool},targets11)




# DATOS NORMALIZADOS
(ann, LossTotal, AccuracyTotal) = trainChain(topology, newInputs, targets11; maxEpochs=MaxEpochspossible, learningRate=learningRate)

# DATOS SIN NORMALIZAR
(ann, LossTotal, AccuracyTotal) = trainChain(topology, inputs1, targets11; maxEpochs=MaxEpochspossible, learningRate=learningRate)

println(ann)
println(LossTotal[1])
println(AccuracyTotal[1])

#############################################################################################################
# CONCLUSIONES:
# Como vemos no existe gran diferencia al introducir a la matriz de entrenamiento entradas sin normalizar.
# Cuando los datos están sin normalizar, tienden a tener una precisión más baja, pero estos valores fluctuan bastante por lo que no podemos obtener una conclusión clara.
# EL MEJOR VALOR DE TOPOLOGY PARA NUESTRO ENTRENAMIENTO HA SIDO:
# 3,2
# Nos aporta el menor porcentaje de loss
# Haciendo pruebas con distintas tasas de aprendizaje, hemos decidido poner como valor de learningRate 0.05



##############################################################################################################
##############################################################################################################
#ENTREGA 3 AA1: PROBLEMA DE CLASIFICACIÓN SUPERVISDADA

#Importamos librería

using Random

# Esta función es para separar los indices del número de patrones y el porcetaje
# de patrones se separan para realizar el test
function holdOut(N::Int, P::Float64)
    ind = randperm(N); # Esta función elige de manera ramdom los indices seleccionados
    numTrain = Int(round(N*(1-P)));
    return (ind[1:numTrain], ind[numTrain+1:end]); # tupla de dos vectores de los indices
    # de los patrones que serán utilizados para entrenamiento y test
    # los vectores tienen que ser disjuntos y la suma de la longitud de los vectores = N
end;


# FUNCIÓN IGUAL QUE LA ANTERIOR QUE TOMA 3 PARÁMETROS
function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    # Separamos indices de validacion y entrenamiento de los indices de test
    (trainingvalidationInd, testInd) = holdOut(N, Ptest);
    # Separamos los indices de validación de los de entrenamiento
    (trainingInd, validationInd) = holdOut(length(trainingvalidationInd), Pval*N/length(trainingvalidationInd))
    # Para ello hacemos uso de la función de holdout anterior
    return (trainingvalidationInd[trainingInd],trainingvalidationInd[validationInd], testInd);
    # Devolvemos una tupla de 3 vectores con los indices de entrenamiento, validación
    # y test
end;



#________________________FUNCIONES RELATIVAS A LAS RNA_______________________________________#

# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de test
function trainChain(topology::AbstractArray{<:Int64},
    trainingInputs::Array{Float32,2}, trainingTargets::Array{Bool,2},
    testInputs::Array{Float32,2},     testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1, showText::Bool=false)

    ####COPROBACIONES PREVIAS####
    # Coinciden número de filas en entradas y salidas de entrenamiento y test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Coinciden número de filas en entradas y salidas de entrenamiento y test
    @assert(size(trainingInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2));

    # Creamos la RNA
    ann = RNAChain(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # ALMACENAMOS los valores
    trainingLossTotal = Float64[];
    trainingAccuracyTotal = Float64[];
    testLossTotal = Float64[];
    testAccuracyTotal = Float64[];
    # ciclo 0
    Epoch = 0;

    trainingLoss = loss(trainingInputs', trainingTargets');
    testLoss     = loss(testInputs',     testTargets');
    #  salida de la RNA en entrenamiento y test. traspuestas
    training_outputs = ann(trainingInputs');
    test_outputs     = ann(testInputs');
    # Creamos código para dos opciones:
    # MANERAS DE PASAR LOS DATOS:
    # FILAS:
    training_accu = accuracy(Array{Float32,2}(training_outputs'), trainingTargets; dataInRows=true);
    test_accu     = accuracy(Array{Float32,2}(test_outputs'),     testTargets;     dataInRows=true);
    # COLUMNAS
    training_accu = accuracy((training_outputs), Array{Bool,2}(trainingTargets'); dataInRows=false);
    test_accu     = accuracy(test_outputs,     Array{Bool,2}(testTargets');     dataInRows=false);

    #____________________________RESULTADOS________________________________________#
    # Imprimimos por pantalla resultados
    #println("Epoch/ Ciclo ", Epoch, ": Loss de entrenamiento: ", trainingLoss, ", Precisión de entrenamiento: ", training_accu, "Loss de test: ", testLoss, ", Precisión de test: ", test_accu);
    #println("PORCENTAJES:")
    #println("Loss de entrenamiento  ", trainingLoss*100, " %")
    #println("Loss de test  ", testLoss*100, " %")
    #println("Precisión de entrenamiento  ", training_accu*100, " %")
    #println("Precisión de test  ", test_accu*100, " %")
    #Debido a que un RR.NN.AA es una funicón no determinista se tendra que realizar cada entrenamiento un gran número de veces lo que supone que estos resultado
    #se mostrarán muchas veces al implementar el código. Para que esto no ocurra lo dejamos comentado

    # Almacenamos
    push!(trainingLossTotal,      trainingLoss);
    push!(testLossTotal,          testLoss);
    push!(trainingAccuracyTotal,  training_accu);
    push!(testAccuracyTotal,      test_accu);


    # Entrenamos hasta que se cumpla ALGUNA condicion de parada
    # o BIEN SE LLEGUE AL MÁXIMO DE CICLOS O BIEN LOSS SEA SEA MENOS O IGUAL QUE MIN LOSS

    while (Epoch<maxEpochs) && (trainingLoss>minLoss)
        # ENTRENAMIENTO DE UN CICLO CON MATRICES TRASPUESTAS.
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        # Le sumamos  1 al contador de ciclos
        Epoch += 1;
        # ALMACENAMOS
        push!(testLossTotal,testLoss);
        push!(trainingLossTotal, trainingLoss);
        push!(testAccuracyTotal, test_accu);
        push!(trainingAccuracyTotal, training_accu);

    end;
    return (ann, trainingLossTotal, testLossTotal, trainingAccuracyTotal, testAccuracyTotal);
end;



##################################PRUEBA DE ENTRENAMIENTO SON ÚNICAMENTE 2 CONJUNTOS: TRAINING Y TEST #################################

topology = [3, 2];
learningRate = 0.02; # Tasa de aprendizaje
MaxEpochspossible = 1000; # Numero maximo de ciclos de entrenamiento
testRatio = 0.2; # Porcentaje de patrones que se usaran para test

using DelimitedFiles
dataset11 = readdlm("transfusion.data",',');
# SEPARAMOS LAS ENTRADAS DE LAS SALIDAS
inputs11 = dataset11[:,1:4];
inputs11 = convert(Array{Float32,2},inputs11);

targets11 = oneHotEncoding(targets1,unique(targets1))

targets11=convert(AbstractMatrix{Bool},targets11)

# INDICES
(trainingInd, testInd) = holdOut(size(inputs1,1), testRatio);

# Dividimos los datos
trainingInputs  = inputs11[trainingInd,:];
testInputs      = inputs11[testInd,:];
trainingTargets = targets11[trainingInd,:];
testTargets     = targets11[testInd,:];

# Calculamos los valores de normalizacion para el conjunto de entrenamiento
normParams = calculateMinMaxNormalizationParameters(trainingInputs);

# Normalizamos
normalizeMinMax!(trainingInputs, normParams);
normalizeMinMax!(testInputs,     normParams);

# Entrenaos nuevamente
(ann, trainingLossTotal, trainingAccuracyTotal) = trainChain(topology,
    trainingInputs, trainingTargets,
    testInputs,     testTargets;
    maxEpochs=MaxEpochspossible, learningRate=learningRate, showText=true);

########################################################################################################################################

# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de validacion, y parar el entrenamiento si es necesario

function trainChain(topology::AbstractArray{<:Int64},
    validationInputs::Array{Float32,2}, validationTargets::Array{Bool,2},
    trainingInputs::Array{Float32,2}, trainingTargets::Array{Bool,2},
    testInputs::Array{Float32,2},     testTargets::Array{Bool,2};
    maxEpochs::Int64=1000, maxEpochsVal::Int64=20,minLoss::Float64=0.0, learningRate::Float64=0.1, showText::Bool=false)


    ####COPROBACIONES PREVIAS####
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    @assert(size(trainingInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2));
    @assert(size(validationInputs,2)==size(validationInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2))

    # Creamos la RNA
    ann = RNAChain(size(trainingInputs,2), topology, size(trainingTargets,2));

    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # ALMACENAMOS los valores
    trainingLossTotal = Float64[];
    trainingAccuracyTotal = Float64[];
    testLossTotal = Float64[];
    testAccuracyTotal = Float64[];
    validationLossTotal = Float64[];
    validationAccuracyTotal = Float64[];
    # ciclo 0
    Epoch = 0;


    trainingLoss = loss(trainingInputs', trainingTargets');
    testLoss     = loss(testInputs',     testTargets');
    validationLoss = loss(validationInputs', validationTargets');
    #  salida de la RNA en entrenamiento y test y validación. traspuestas.
    training_outputs = ann(trainingInputs');
    test_outputs     = ann(testInputs');
    validation_outputs = ann(validationInputs');

    # Creamos código para dos opciones:
    # MANERAS DE PASAR LOS DATOS:
    # FILAS:
    training_accu = accuracy(Array{Float32,2}(training_outputs'), trainingTargets; dataInRows=true);
    test_accu     = accuracy(Array{Float32,2}(test_outputs'),     testTargets;     dataInRows=true);
    validation_accu = accuracy(Array{Float32,2}(validation_outputs'), validationTargets; dataInRows=true);
    # COLUMNAS
    training_accu = accuracy((training_outputs), Array{Bool,2}(trainingTargets'); dataInRows=false);
    test_accu    = accuracy(test_outputs,     Array{Bool,2}(testTargets');     dataInRows=false);
    validation_accu = accuracy(validation_outputs, Array{Bool,2}(validationTargets'); dataInRows=false);

    #____________________________RESULTADOS________________________________________#

    # Imprimimos por pantalla resultados
    #println("Epoch/ Ciclo ", Epoch, ": Loss de entrenamiento: ", trainingLoss, ", Precisión de entrenamiento: ", training_accu, "Loss de test: ", testLoss, ", Precisión de test: ", test_accu, "Loss de validación:", validationLoss, "Precisión de validación:", validation_accu);
    #println("PORCENTAJES:")
    #println("Loss de entrenamiento  ", trainingLoss*100, " %")
    #println("Loss de test  ", testLoss*100, " %")
    #println("Loss de validación  ", validationLoss*100, " %")
    #println("Precisión de entrenamiento  ", training_accu*100, " %")
    #println("Precisión de test  ", test_accu*100, " %")
    #println("Precisión de validación  ", validation_accu*100, " %")
    #debido a que un RR.NN.AA es una funicón no determinista se tendra que realizar cada entrenamiento un gran número de veces lo que supone que estos resultado
    #se mostrarán muchas veces al implementar el código. Para que esto no ocurra lo dejamos comentado

    # Almacenamos
    push!(trainingLossTotal,      trainingLoss);
    push!(testLossTotal,          testLoss);
    push!(validationLossTotal,      validationLoss);
    push!(trainingAccuracyTotal,  training_accu);
    push!(testAccuracyTotal,      test_accu);
    push!(validationAccuracyTotal,      validation_accu);

# SOLO SE VA A DEVOLVER LA MEJOR VALIDACIÓN

    EpochsValidation = 0; bestValidationLoss = validationLoss;
    # mejor ann conseguida durante el entrenamiento
    goodAnn = deepcopy(ann);
    # Entrenamos hasta que se cumpla ALGUNA condicion de parada
    # o BIEN SE LLEGUE AL MÁXIMO DE CICLOS O BIEN LOSS SEA SEA MENOS O IGUAL QUE MIN LOSS

    while (Epoch<maxEpochs) && (trainingLoss>minLoss)
        # ENTRENAMIENTO DE UN CICLO CON MATRICES TRASPUESTAS.
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));

        # Le sumamos  1 al contador de ciclos
        Epoch += 1;

        #  ALMACENAMOS VALORES
        push!(testLossTotal,testLoss);
        push!(trainingLossTotal, trainingLoss);
        push!(validationLossTotal, validationLoss)
        push!(testAccuracyTotal, test_accu);
        push!(trainingAccuracyTotal, training_accu);
        push!(validationAccuracyTotal, validation_accu)
        # Aplicamos la parada temprana
        if (validationLoss<bestValidationLoss)
            bestValidationLoss = validationLoss;
            EpochsValidation = 0;
            goodAnn = deepcopy(ann);
        else
            EpochsValidation += 1;
        end;
    end;
    return (goodAnn, trainingLossTotal, testLossTotal, validationLossTotal, trainingAccuracyTotal, testAccuracyTotal, validationAccuracyTotal);
end;


################# PRUEBA CON 3 CONJUNTOS: TRAINING, TEST Y VALIDATION #####################################
# PARÁMETROS
topology = [5, 4];
learningRate = 0.02; # Tasa de aprendizaje
MaxEpochsPossible = 1000;
validationRatio = 0.2;
testRatio = 0.2;
maxEpochsVal = 90;

# Cargamos el dataset
dataset11 = readdlm("transfusion.data",',');
inputs11 = dataset11[:,1:4];
inputs11 = convert(Array{Float32,2},inputs11);

targets11 = oneHotEncoding(targets1,unique(targets1))

targets11=convert(AbstractMatrix{Bool},targets11)
size(targets11)

# Normalizamos
newInputs3 = normalizeMinMax!(inputs11);
println(inputs11)
newInputs4=convert(Array{Float32,2},newInputs3)
# Creamos los indices de entrenamiento, validacion y test
(trainingInd, validationInd, testInd) = holdOut(size(newInputs4,1), validationRatio, testRatio);

# Dividimos los datos
trainingInputs    = inputs11[trainingInd,:];
validationInputs  = inputs11[validationInd,:];
testInputs        = inputs11[testInd,:];
trainingTargets   = targets11[trainingInd,:];
validationTargets = targets11[validationInd,:];
testTargets       = targets11[testInd,:];

trainingInputs=convert(Array{Float32,2},trainingInputs);
testInputs=convert(Array{Float32,2},testInputs)
validationInputs=convert(Array{Float32,2},validationInputs)

(ann, trainingLossTotal, trainingAccuracyTotal) = trainChain(topology,
    trainingInputs,   trainingTargets,
    validationInputs, validationTargets,
    testInputs,       testTargets;
    maxEpochs=MaxEpochsPossible, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);


##############################################################
###############################################################
# ENTREGA 4.1 AA1: PROBLEMA DE CLASIFICACIÓN



#________________________________FUNCIONES CONFUSIONMATRIX________________________________#
# Esta acepta vectores de igual longitud: outputs y targets.
# DEVUELVE MÉTRICAS
# Valor de precisión.
# Tasa de fallo.
# Sensibilidad.
# Especificidad.
# Valor predictivo positivo.
# F1-score.
# Matriz de confusión, como un objeto de tipo Array{Int64,2} con dos filas y dos columnas.
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Esta acepta vectores de igual longitud: outputs y targets.
    @assert(length(outputs)==length(targets));

    #______________________DEFINICIÓN DE LAS MÉTRICAS________________________#
    accuracy1    = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate   = 1. - accuracy1; # Porcentaje de error
    sensitivity = mean(outputs[targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    PPV         = mean(targets[outputs]); # Valor predictivo positivo
    NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo

    # LOS VERDADEROS NEGATIVOS CONSTITUYEN LA TOTALIDAD DE LOS PATRONES
    if isnan(PPV) && isnan(sensitivity)
        PPV = 1.;
        sensitivity = 1.;
    # LOS VERDADEROS POSIYIVOS CONSTITUYEN LA TOTALIDAD DE LOS PATRONES
    elseif isnan(specificity) && isnan(NPV)
        specificity = 1.;
        NPV = 1.;
    # SI NO HAN OCURRIDO LOS CASOS ANTERIORES, SU VALOR SERÁ 0.
    elseif isnan(sensitivity)
        sensitivity = 0
    elseif isnan(specificity)
        specificity = 0
    elseif isnan(PPV)
        PPV = 0
    elseif isnan(NPV)
        NPV = 0
    end;
    # F1
    # Si el valor de PPV y sensitivity son 0, F1 no se puede calcular, por lo tanto le asignamos valor 0.
    if sensitivity==0 && PPV==0
        F1 = 0
    # si esto no ocurre, calculamos su valor
    else
        F1 = 2*(sensitivity*PPV)/(sensitivity+PPV);
    end;
    # Ahora creamos la matriz de confusion
    # Reservamos memoria para esta
    cMatrix = Array{Int64,2}(undef, 2, 2);
    # PRIMERA FILA/ COLUMNA: VALORES DE CLASE NEGATIVOS
    cMatrix[1,1] = sum(.!targets .& .!outputs); # Verdaderos negativos
    cMatrix[1,2] = sum(.!targets .&   outputs); # Falsos posiyivos
    # SEGUNDA FILA/ COLUMNA: VALORES DE CLASE POSITIVOS
    cMatrix[2,1] = sum(  targets .& .!outputs); # Falsos negativos
    cMatrix[2,2] = sum(  targets .&   outputs); # Verdaderos positivos
    return (accuracy1, errorRate, sensitivity, specificity, PPV, NPV, F1, cMatrix)
end;

# FUNCIÓN DE IGUAL NOMBRE QUE LA ANTERIOR
# primer vector: valores reales + tercer parámetro opcional
confusionMatrix(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);


################################################################################################
#################################################################################################
#Entrega 4.2 AA1: PROBLEMA DE CLASIFICACIÓN SUPERVISADA

# ESTRATEGIA UNO CONTRA TODOS

function oneVSall(inputs::Array{Float32,2}, targets::Array{Bool,2})
    # Calculamos número de classes
    numClasses = size(targets,2);
    # creamos matriz bidimensional
    outputs = Array{Float32,2}(undef, numInstances, numClasses);
    # BUCLE que itere sobre cada clase
    for numClass in 1:numClasses
        # Modelo que indica la pertenencia o no a cada clase
        model = fit(inputs, targets[:,[numClass]]);
        outputs[:,numClass] .= model(inputs);
    end;
    # Aplicamos la funcion softmax
    outputs = softmax(outputs')';
    # salida mayor de cada clase
    vmax = maximum(outputs, dims=2);
    outputs = (outputs .== vmax);
    # Convertimos a matriz de valores booleanos
    return mean(vmax);
end;


# FUNCIÓN CONFUSIONMATRIX QUE PERMITA DEVOLVER LOS VALORES DE LAS MÉTRICAS ADAPTADAS A  LA CONDICIÓN DE TENER MÁS DE DOS CLASES.
# AÑADIENDO UN NUEVO PARÁMETRO PARA CALCULARLAS DE LAS FORMAS MACRO Y WEIGHTED.
function confusionMatrix(outputs::Array{Bool,2},targets::Array{Bool,2}; weighted::Bool=true)
    # Número de columnas en ambas matrices es igual
    @assert(size(outputs)==size(targets));
    # Es distinto de 2 el numero de columnas
    numClasses = size(targets,2);
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    # Más de dos columnas
    else
        # Reservamos memoria para los vectores de sensibilidad, especificidad, VPP, VPN y F1 con
        # un valor por clase, inicialmente iguales a 0.
        # FUNCIÓN ZEROS
        #@assert(all(sum(outputs, dims=2).==1));
        sensitivity = zeros(numClasses);
        specificity = zeros(numClasses);
        PPV         = zeros(numClasses);
        NPV         = zeros(numClasses);
        F1          = zeros(numClasses);

        # Debemos iterar para cada clase, y si hay patrones en esa clase, realizar una llamada a la función ConfusionMatrix
        # y pasarle como vectores las columnas correspondientes a la clase de esa iteración de las matrices outputs y targets.
        # PRIMERO, calculamos el número de patrones por clase.
        instancesPerClass =vec(sum(targets, dims=1));
        # Iteramos para cada clase, las que tengan como número de patrones 0 no serán contadas.
        for numClass in findall(instancesPerClass.>0)
            # Calculamos las métricas llamando a la función anterior y asignando el resultado a los distintos vectores de sensibilidad,...
            (_, _, accuracy1[numClass], specificity[numClass], PPV[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end;
        # Reservamos memoria para la matriz de confusion
        cMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        #Bucle doble para que los bucles iteren sobre las clases y que rellenen las celdas de la matriz de confusión.
        for numClassTargets in 1:numClasses, numClassOutputs in 1:numClasses
            cMatrix[numClassTargets, numClassOutputs] = sum(targets[:,numClassTargets] .& outputs[:,numClassOutputs]);
        end;
        # Unimos los valores que se tienen de sensibilidad, especificidad, etc... para cada clase en un único valor usando
        # la estrategia weighted que especificamos en el argumento de entrada.
        if weighted
            # Calculamos las ponderaciones para el promedio
            weights     = instancesPerClass./sum(instancesPerClass);
            sensitivity = sum(weights.*sensitivity);
            specificity = sum(weights.*specificity);
            PPV         = sum(weights.*PPV);
            NPV         = sum(weights.*NPV);
            F1          = sum(weights.*F1);
        else
            # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
            #  REALIZAMOS LA MEDIA SOLO CON LAS QUE TENGAN INSTANCIAS
            classesWithInstances = sum(instancesPerClass.>0);
            sensitivity          = sum(sensitivity)/classesWithInstances;
            specificity          = sum(specificity)/classesWithInstances;
            PPV                  = sum(PPV)/classesWithInstances;
            NPV                  = sum(NPV)/classesWithInstances;
            F1                   = sum(F1)/classesWithInstances;
        end;
        # Accuracy y errorRate las calculaemos con funciones ya definidas.
        accuracy1 = accuracy(outputs, targets; dataInRows=true);
        errorRate = 1 - accuracy1;
        # LA FUNCIÓN DEVUELVE:
        return (accuracy1, errorRate, sensitivity, specificity, PPV, NPV, F1, cMatrix);
    end;
end;

# FUNCIÓN confusionMatrix  QUE CONVIERTE EL PRIMER PARÁMETRO A UNA MATRIZ DE VALORES BOOLEANOS MEDIANTE classifyOutputs Y
# llama a la función anterior.
confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);

# FUNCION del mismo nombre que realiza la misma tarea pero que los vectores de entrada sean de cualquier tipo, ya que representan
# las clases de ccualquier forma.

function confusionMatrix(outputs::Array{Any,1}, targets::Array{Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    #@assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    a=oneHotEncoding(outputs, classes)
    b=oneHotEncoding(targets, classes)
    a=convert(Array{Float64,2},a)
    b=convert(Array{Bool,2},b)
    return confusionMatrix(a, b; weighted=weighted);
end;

confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);


#___________________________________FUNCIONES TOMADAS DEL CÓDIGO GUIA______________________________________#

# De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (accuracy1, errorRate, sensitivity, specificity, PPV, NPV, F1, cMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(cMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTargets in 1:numClasses
         print.(cMatrix[numClassTargets,:], "\t");
        if (numClasses==2)
            print(numClassTargets == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTargets, "\t| ");
        end;
        print.(cMatrix[numClassTargets,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", accuracy1);
    println("Error rate: ", errorRate);
    println("Recall: ", sensitivity);
    println("Specificity: ", specificity);
    println("Precision: ", PPV);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (accuracy1, errorRate, sensitivity, specificity, PPV, NPV, F1, cMatrix);
end;
printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)


##############################PRUEBAS#####################################

# PPARÁMETRPOS
topology = [4, 3];  # Cambiamos la topología a 4,3 para obtener mejores resultados en la precision
learningRate = 0.01;
MaxEpochsPossible = 1000;
validationRatio = 0.2;
testRatio = 0.2;
maxEpochsVal = 8;

# Cargamos el dataset
dataset1 = readdlm("transfusion.data",',');
# Preparamos las entradas y las salidas deseadas
inputs11 = convert(Array{Float32,2}, dataset1[:,1:4]);
targets11 = oneHotEncoding(dataset1[:,5], unique(targets11));
targets11 = convert(Array{Float32,2}, targets11);

numClasses = size(targets11,2);

# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
#### AQUÍ NO SE Q PASA @assert(numClasses>2);
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs11);
println(inputs11)
println(targets11)
typeof(inputs11)
typeof(targets11)
# Creamos los indices de entrenamiento, validacion y test
(trainingInd, validationInd, testInd) = holdOut(size(inputs11,1), validationRatio, testRatio);

# Dividimos los datos
# Dividimos los datos
trainingInputs    = inputs11[trainingInd,:];
validationInputs  = inputs11[validationInd,:];
testInputs        = inputs11[testInd,:];
trainingTargets   = targets11[trainingInd,:];
validationTargets = targets11[validationInd,:];
testTargets       = targets11[testInd,:];

trainingInputs=convert(Array{Float32,2},trainingInputs);
testInputs=convert(Array{Float32,2},testInputs)
validationInputs=convert(Array{Float32,2},validationInputs)


#Reservamos memoria para las matrices de salidas de entrenamiento, validacion y test
# En lugar de hacer 3 matrices, voy a hacerlo en una sola con todos los datos
outputs = Array{Float32,2}(undef, size(inputs11,1), numClasses);


trainingTargets   = convert(AbstractMatrix{Bool},trainingTargets)
validationTargets = convert(AbstractMatrix{Bool},validationTargets)
testTargets       = convert(AbstractMatrix{Bool},testTargets)



# Y creamos y entrenamos la RNA con los parametros dados para cada una de las clases
for numClass = 1:numClasses

    # A partir de ahora, no vamos a mostrar por pantalla el resultado de cada ciclo del entrenamiento de la RNA (no vamos a poner el showText=true)
    local ann;
    ann, = trainChain(topology,
        trainingInputs,   trainingTargets[:,[numClass]],
        validationInputs, validationTargets[:,[numClass]],
        testInputs,       testTargets[:,[numClass]];
        maxEpochs=MaxEpochsPossible, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

    # Aplicamos la RNA para calcular las salidas para esta clase concreta y las guardamos en la columna correspondiente de la matriz
    outputs[:,numClass] = ann(inputs11')';

end;
# A estas 3 matrices de resultados le pasamos la funcion softmax
# Esto es opcional, y nos vale para poder interpretar la salida de cada modelo como la probabilidad de pertenencia de un patron a una clase concreta
outputs = collect(softmax(outputs')');
targets11 = convert(AbstractMatrix{Bool}, targets11);

# Mostramos las matrices de confusion y las metricas
println("Results in the training set:")
printConfusionMatrix(outputs[trainingInd,:], trainingTargets; weighted=true);
println("Results in the validation set:")
printConfusionMatrix(outputs[validationInd,:], validationTargets; weighted=true);
println("Results in the test set:")
printConfusionMatrix(outputs[testInd,:], testTargets; weighted=true);
println("Results in the whole dataset:")
printConfusionMatrix(outputs, targets11; weighted=true);





##################################################################################################################################
########################################################################################################################################
# ENTREGA 5 AA1: PROBLEMA CLASIFICACIÓN SUPERVISADA

using Random
using Random:seed!



#_____________________________________FUNCIONES CROSSVALIDATION_________________________________________#

# FUNCION CROSSVALLIDATION: reciba número de patrones y número de subconjuntos en los que se va a partir el conjunto de datos
# y devuelva un vector de N longitud, donde cada elemento indica en que subconjunto debe ser incluido ese patrón.
function crossvalidation(N::Int64, k::Int64)
    # Vector con k elementos ordenados
    ind = repeat(1:k, Int64(ceil(N/k)));
    # Tomamos N primeros valores
    ind = ind[1:N];
    # Desordenar el vector
    shuffle!(ind);
    return ind;
end;

# FUNCIÓN CROSSVALIDATION: recibe targets y un valor k de subconjuntos en los que se dividirá el conjunto de datos
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    @assert(size(targets,1)>=10)
    N=size(targets,1)#filas de targets
    M=size(targets,2)#columnas de targets
    ind=[1:N]#vector de indices de longitud igual al número de filas
    for i in range(M) #bucle que itere sobre las columnas de la matriz targets
        a=sum(targets[:,i])#sumar los elementos de dicha columna
        b=crossvalidation(a,k)#aplicar crossvalidation sobre estos elementos sumados
        cnt=i #inicializar contador en la posicion de i
        while cnt <= N#bucle de sustitución indices
            ind[cnt]=b#sustitución de los bucles
        end
    end
    return ind #devolver los indices
end

# FUNCIÓN CROSSVALIDATION: targets de nuevo tipo y realiza validación cruzada estratificada
function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets=oneHotEncoding(targets)
    ind=crossvalidation(targets,k)
    return ind
end

######################################################################3
# TOMADO DEL CÓDIGO GUÍA
# Código de prueba:
# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(123);

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
MaxEpochsPossible = 1000; # Numero maximo de ciclos de entrenamiento
numFolds = 10;
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Cargamos el dataset
dataset5 = readdlm("transfusion.data",',');
# Preparamos las entradas y las salidas deseadas
inputs5 = convert(Array{Float32,2}, dataset5[:,1:4]);
targets5=dataset5[:,5]
targets5  = oneHotEncoding(targets5,unique(targets5))
targets5=convert(Matrix{Bool},targets5)
print(targets5)

numClasses = size(targets5,2);
print(numClasses)
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"

# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs5);

# Creamos los indices de crossvalidation
crossValidationInd = crossvalidation(size(inputs5,1), numFolds);

# Creamos los vectores para las metricas que se vayan a usar
# En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
testAccuracies = Array{Float64,1}(undef, numFolds);
testF1         = Array{Float64,1}(undef, numFolds);

# Para cada fold, entrenamos
for numFold in 1:numFolds

    # Dividimos los datos en entrenamiento y test
    local trainingInputs, testInputs, trainingTargets, testTargets;
    trainingInputs    = inputs5[crossValidationInd.!=numFold,:];
    testInputs        = inputs5[crossValidationInd.==numFold,:];
    trainingTargets   = targets5[crossValidationInd.!=numFold,:];
    testTargets       = targets5[crossValidationInd.==numFold,:];

    # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
    # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
    testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsAANTraining);
    testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsAANTraining);

    for numTraining in 1:numRepetitionsAANTraining

        if validationRatio>0

            # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
            #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
            #  Para ello, hacemos un hold out
            local trainingInd, validationInd;
            (trainingInd, validationInd) = holdOut(size(trainingInputs,1), validationRatio*size(trainingInputs,1)/size(inputs,1));
            # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

            # Entrenamos la RNA
            local ann;
            ann, = trainChain(topology,
                trainingInputs[trainingInd,:],   trainingTargets[trainingInd,:],
                trainingInputs[validationInd,:], trainingTargets[validationInd,:],
                testInputs,                          testTargets;
                maxEpochs=MaxEpochsPossible, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

        else

            # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
            local ann;
            ann, = trainChain(topology,
                trainingInputs, trainingTargets,
                testInputs,     testTargets;
                maxEpochs=MaxEpochsPossible, learningRate=learningRate);

        end;

        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

        # Almacenamos las metricas de este entrenamiento
        testAccuraciesEachRepetition[numTraining] = acc;
        testF1EachRepetition[numTraining]         = F1;

    end;

    # Almacenamos las 2 metricas que usamos en este problema
    testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
    testF1[numFold]         = mean(testF1EachRepetition);

    println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

end;

println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));


#########################################################################################################################################################
##########################################################################################################################################################
# ENTREGA 6: PROBLEMA DE CLASIFICACIÓN AA1

using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

#________________________________FUNCIÓN ADAPTADA DEL CÓDIGO GUÍA________________________________________#
#Creamos la función donde se incluyen todos los modelos que queremos entrenar
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    # Indicamos las clases de salida que tenemos
    classes = unique(targets);
    # PARA RNA
    # Codificamos mediante onehotencoding las clases
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    # Creamos los indices de crossvalidation
    crossValidationInd = crossvalidation(size(inputs,1), numFolds);

    # Creamos los vectores para las metricas
    testAccuracyTotal = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos (no rna)
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test mediante crossvalidation
            trainingInputs    = inputs[crossValidationInd.!=numFold,:];
            testInputs        = inputs[crossValidationInd.==numFold,:];
            trainingTargets   = targets[crossValidationInd.!=numFold];
            testTargets       = targets[crossValidationInd.==numFold];

            #Pasamos los parámetros correspondientes segun cada modelo
            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento mediante la función fit
            # es importante destacar que las salidas deseadas estan en forma de vector no de matriz
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de tes tmediante la función predict  que predice dichos resultados

            testOutputs = predict(model, testInputs);
            testOutputs = convert(Array{Any,1},testOutputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            #Son relevantes las métricas de la precisión y del f1-score, por lo tanto son los que mostraremos por pantalla

            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);


        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test mediante cross-validation
            trainingInputs    = inputs[crossValidationInd.!=numFold,:];
            testInputs        = inputs[crossValidationInd.==numFold,:];
            trainingTargets   = targets[crossValidationInd.!=numFold,:];
            testTargets       = targets[crossValidationInd.==numFold,:];

            #convertimos los datos a tipo vector float y matrix bool
            trainingInputs=convert(Array{Float32,2},trainingInputs);
            testInputs=convert(Array{Float32,2},testInputs)
            trainingTargets = convert(AbstractMatrix{Bool},trainingTargets)
            testTargets = convert(AbstractMatrix{Bool},testTargets)

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento

            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que idicamos mediante el parámetro nunExecutions
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Utilizamos la funcion holdout para realizar esta división
                    # Con estos índices, crearemos los vectores que mostraran el rna entrenado
                    (trainingInd, validationInd) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));

                    # Entrenamos la RNA

                    ann, = trainChain(modelHyperparameters["topology"],
                        trainingInputs[trainingInd,:],   trainingTargets[trainingInd,:],
                        trainingInputs[validationInd,:], trainingTargets[validationInd,:],
                        testInputs,                          testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    #Si no deseamos hacer esto con un conjunto de validación solo utilizamos el vector entrenamiento y test
                    #Entrenamos el rna
                    ann, = trainChain(modelHyperparameters["topology"],
                        trainingInputs, trainingTargets,
                        testInputs,     testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas más significativas para este modelo
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio los entrenamientos en el fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;



##############################PRUEBA#######################################

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);

numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 4;

# Parapetros de kNN
numNeighbors = 3;

# Cargamos el dataset
dataset1 = readdlm("transfusion.data",',');
# Preparamos las entradas y las salidas deseadas
inputs1 = convert(Array{Float32,2}, dataset1[:,1:4]);
targets1 = dataset1[:,5];


# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs1);
inputs = convert(Array{Float64,2}, inputs1);
targets = convert(Array{Any,1},targets1)
c=unique(targets)
print(c)


# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);


# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);

# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, numFolds);

# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);

#################################################################################

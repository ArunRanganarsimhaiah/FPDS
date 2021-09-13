#!/usr/bin/env python
# coding: utf-8

# # Projekt 1: Mehrfachklassifikation auf MNIST
# 
# Herzlich willkommen zu Ihrem ersten Projekt *Mehrfachklassifikation auf MNIST* im Fortgeschrittenenpraktikum Data Science! Dieses Notebook führt Sie durch einige grundlegende Schritte, die im Rahmen eines Machine-Learning-Projektes zu bearbeiten sind.
# 
# In Abschnitt *1.1 Vor dem Training* werden Sie zunächst den Datensatz MNIST laden, sich einen Überblick über die Daten verschaffen und die Daten in ein Format bringen, welches zur weiteren Verarbeitung mittels der Deep-Learning-Bibliothek Keras geeignet ist. Anschließend werden Sie mit Keras eine Modellarchitektur definieren und in Abschnitt *1.2 Trainingsphase* die Parameter des Modells trainieren. Im abschließenden Abschnitt *1.3 Nach dem Training* werden Sie schließlich Techniken zur Modellinterpretation kennenlernen und mittels Hyperparameter-Tuning die Modellarchitektur verfeinern.
# 
# Der Zeitraum für die Bearbeitung des Projektes dauert bis zum **13. November 2020** um **9:45 Uhr**. Senden Sie alle abzugebenden Dateien bis zu diesem Zeitpunkt an **m.neumann-brosig@tu-bs.de** und **m.lahmann@tu-bs.de**. Um das Projekt erfolgreich abzuschließen, ist weiterhin eine kurze Vorstellung Ihrer Lösung im Rahmen einer Videokonferenz notwendig. Hierfür erhalten Sie von uns einen individuellen Termin.

# 1. Führen Sie den folgenden Code aus, um die für dieses Projekt benötigten Python-Pakete zu laden.

# In[42]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import kerastuner as kt


# ## 1.1 Vor dem Training
# 
# ### Laden der Daten
# 2. Laden Sie den Datensatz ```mnist``` in der Version ```3.0.1```. Rufen Sie die Funktion ```tfds.load``` nur einmal auf und gestalten Sie den Aufruf so, dass Sie als Rückgabewert eine Liste ```[train_ds, val_ds, test_ds]```sowie ein Objekt ```info``` vom Typ ```tfds.core.DatasetInfo``` erhalten. Dabei soll ```train_ds``` die vorderen ```90%``` des Trainingssplits, ```val_ds``` die hinteren ```10%``` des Trainingssplits sowie ```test_ds``` den vollständigen Testsplit von ```mnist``` enthalten.
# 
# Nützliche Informationen finden Sie unter https://www.tensorflow.org/datasets/catalog/mnist und https://www.tensorflow.org/datasets/overview.

# In[43]:


# IHR CODE
(train_ds, val_ds, test_ds), info = tfds.load('mnist', split= ['train[:90%]','train[-10%:]', 'test'], with_info = True)


# Die geladenen Datensätze ```train_ds```, ```val_ds``` und ```test_ds``` sind Objekte vom Typ ```tensorflow.data.Dataset``` (oder einer davon abgeleiteten Klasse). Darüber hinaus haben wir mit ```info``` ein Objekt vom Typ ```tfds.core.DatasetInfo``` geladen, welches nützliche Informationen über die geladenen Daten enthält. Wir beschäftigen uns im Folgenden zunächst damit, welche Informationen in ```info``` enthalten sind und wie sich darauf zugreifen lässt. Anschließend lernen Sie, wie Sie auf die Elemente der Datensätze zugreifen können.

# ### Sichten der Daten
# 
# Nutzen Sie nun ```info``` dazu, sich einen groben Überblick über ```mnist``` zu verschaffen. Jedes in ```train_ds```, ```val_ds``` bzw. ```test_ds``` enthaltene Element ```item``` ist ein ```dictionary``` mit Schlüsseln ```"image"``` und ```"label"```.
# 
# 3. Lassen Sie, in der angegebenen Reihenfolge, folgene Informationen ausgeben:
#    - Das Format der Bilder ```item["image"]``` (Bildformat).
#    - Das Format der Labels ```item["label"]``` (Labelformat).
#    - Die Anzahl verschiedener Klassen (Klassenanzahl).
#    - Die Namen der Klassen (Klassennamen).
#    - Die Anzahl der Trainingsbeispiele im Trainingssplit (Anzahl Trainingsbeispiele).
#    - Die Anzahl der Testbeispiele im Testsplit (Anzahl Testbeispiele).
# 
# All diese Information können Sie aus ```info``` extrahieren. Verwenden Sie an dieser Stelle **nicht** ```train_ds```, ```val_ds``` oder ```test_ds```. Lassen Sie die erste Information mit dem Befehl ```print("Bildformat: {}".format(IHR_CODE))``` ausgeben und gehen Sie in allen anderen Fällen analog vor.

# In[44]:


# IHR CODE
print("Bildformat: {}".format(info.features['image'].shape))
print("Labelformat: {}".format(info.features['label'].shape))
print("Klassenanzahl: {}".format(info.features["label"].num_classes))
print("Klassennamen: {}".format(info.features["label"].names))
print("Anzahl Trainingsbeispiele: {}".format(info.splits['train'].num_examples))
print("Anzahl Testbeispiele: {}".format(info.splits['test'].num_examples))


# **Checkpoint**: Sie sollten folgenden Output erhalten:
# 
# ```
# Bildformat: (28, 28, 1)
# Labelformat: ()
# Klassenanzahl: 10
# Klassennamen: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Anzahl Trainingsbeispiele: 60000
# Anzahl Testbeispiele: 10000
# ```

# Als Nächstes wollen wir uns eine kleine Teilmenge der geladenen Bilder ansehen.
# 
# 4. Verwenden Sie die Funktion ```tfds.show_examples``` mit optionalen Argumenten ```rows=2``` und ```cols=5```, um eine zehnelementige Stichprobe aus ```train_ds``` anzeigen zu lassen.

# In[45]:


# IHR CODE
fig = tfds.show_examples(train_ds, info, rows = 2, cols = 5)


# Wir wissen bereits, wie viele Elemente insgesamt im Trainings- bzw. Testsplit von ```mnist``` enthalten sind. Beim Laden der Daten haben wir allerdings den Trainingssplit weiter in Trainings- und Validierungsdaten unterteilt. Zudem wissen wir noch nicht, wie viele Elemente aus den jeweiligen Klassen in den einzelnen Splits enthalten sind.
# 
# 5. Schreiben Sie eine Funktion ```count_occurences```, welche als Argument einen Datensatz ```ds``` und ein Objekt ```info``` erhält und als Rückgabewert ein Array ```num_occurences``` liefert. Dabei steht ```num_occurences[i]``` für die Anzahl der Elemente mit ```item["label"] = i``` in ```ds```.
# 
# Sie können über alle Objekte in ```ds``` iterieren, indem Sie eine Schleife der Form ```for item in ds: ...``` verwenden.

# In[46]:


# IHR CODE
def count_occurences(ds, info):
        num_occurences = np.zeros(10)
        for item in ds:
            i = item['label']
            num_occurences[i]+= 1
        return num_occurences


# 6. Wenden Sie Ihre Funktion dann auf ```train_ds```, ```val_ds``` und ```test_ds``` an und lassen Sie die Auftrittshäufigkeiten aller Klassen in den jeweiligen Datensätzen ausgeben. Gestalten Sie die erste Ausgabe nach dem Muster
# ```python
# train_occurences = IHR_CODE
# print("train_ds:\t {}".format(train_occurences))
# ```
# und die beiden anderen entsprechend.

# In[47]:


# IHR CODE
train_occurences = count_occurences(train_ds, info)
print("train_ds:\t {}".format(train_occurences))
val_occurences = count_occurences(val_ds, info)
print("val_ds:\t {}".format(val_occurences))
test_occurences = count_occurences(test_ds, info)
print("test_ds:\t {}".format(test_occurences))


# **Checkpoint:** Sie sollten folgenden Output erhalten:
# 
# ```
# train_ds:  [5359 6070 5359 5503 5235 4856 5341 5641 5267 5369]
# val_ds:    [564 672 599 628 607 565 577 624 584 580]
# test_ds:   [ 980 1135 1032 1010  982  892  958 1028  974 1009]
# ```
# 
# Die Anzahl der Elemente in einem Datensatz können Sie alternativ mit Hilfe der Methode ```tensorflow.data.Dataset.cardinality``` ausgeben lassen.

# Zuletzt wollen wir noch das Datenformat und den Wertebereich der geladenen Bilder in Erfahrung bringen.
# 
# 7. Lassen Sie für ein beliebiges Element ```item``` aus ```train_ds``` das Datenformat ```item["image"].dtype``` sowie den kleinsten und den größten Eintrag ausgeben. Gestalten Sie die Ausgabe analog zu oben.
# 
# Mit dem Aufruf ```train_ds.take(1)``` erhalten Sie einen neuen Datensatz vom Typ ```tensorflow.data.Dataset```, welcher nur das erste Element aus ```train_ds``` enthält und über dessen Element(e) Sie wie oben beschrieben iterieren können. Für die Berechnung des kleinsten und des größten Eintrags können die Funktionen ```np.min``` bzw. ```np.max``` hilfreich sein. Obwohl ```item["image"]``` ein Tensor ist, können Sie die genannten ```numpy``` Funktionen direkt darauf anwenden. Falls Sie ```item["image"]``` (oder einen anderen Tensor) einmal explizit als ```numpy``` Array benötigen (was hier nicht der Fall ist), dann liefert die Methode ```item["image"].numpy()``` das gewünschte Resultat.

# In[48]:


# IHR CODE

for item in train_ds:
    dtype = item["image"].dtype
    min   = item["image"].numpy().min()
    max   = item["image"].numpy().max()
    break
print(dtype)
print(min)
print(max)
    


# ### Vorverarbeiten der Daten
# Im vorherigen Schritt haben wir festgestellt, dass die geladenen Bilder ganzzahlig vorliegen und dass jeder Eintrag aus dem Wertebereich $\{0,\dots,255\}$ stammt (ggf. mit Abweichungen, da wir nur ein einzelnes Bild betrachtet haben). Für einen später zu verwendenden Trainingsalgorithmus kann es günstiger sein, den Wertebereich der Einträge vorab auf ein kleineres Intervall zu verdichten. Im nächsten Schritt wollen wir sämtliche Bilder aus ```train_ds```, ```val_ds``` und ```test_ds``` auf den Wertebereich $[0, 1]$ normalisieren, indem wir jeweils ```item["image"]``` durch $255$ teilen. Im selben Zuge wollen wir alle Elemente ```item``` von einem ```dictionary``` zu einem Tupel ```(item["image"] / 255, item["label"])``` transformieren.
# 
# 8. Schreiben Sie eine Funktion ```preprocess```, die als Argument ein Element ```item``` erhält und ein Tupel der beschriebenen Form zurückgibt.

# In[49]:


# IHR CODE
def preprocess(item):
    return(item["image"] / 255, item["label"])
    


# 9. Transformieren Sie die Datensätze ```train_ds```, ```val_ds``` und ```test_ds```, indem Sie jeweils Ihre Funktion ```preprocess``` darauf anwenden. Um den vollständigen Datensatz zu transformieren, genügt jeweils ein Aufruf der Form ```ds = ds.map(preprocess)```.

# In[50]:


# IHR CODE
train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)


# Die Klasse ```tf.data.Dataset``` hält verschiedenste Methoden bereit, welche das Transformieren von Elementen oder eines Datensatzes an sich sehr erleichtern können. Zwei dieser Methoden sind die bereits verwendeten ```tf.data.Dataset.take```, welche ein Argument ```count``` benötigt und einen neuen Datensatz mit den ersten ```count``` Elementen des aufrufenden Datensatzes zurückgibt, sowie ```tf.data.Dataset.map```, welche als Argument eine Abbildungsvorschrift für ein einzelnes Element in Form einer Funktion (bei uns ```preprocess```) benötigt, und diese Funktion automatisch auf alle Elemente eines Datensatzes anwendet.
# 
# Weitere Methoden von ```tf.data.Dataset```, die wir nun zur Konstruktion einer **Inputpipeline** verwenden wollen, sind
# - ```tf.data.Dataset.repeat``` um einen Datensatz zu vervielfältigen,
# - ```tf.data.Dataset.shuffle``` um einen Datensatz zufällig zu durchmischen,
# - ```tf.data.Dataset.batch``` um einen Datensatz in Stapel einer zu spezifizierenden Größe aufzuteilen und
# - ```tf.data.Dataset.prefetch```.
# 
# Die Nutzung von ```tf.data.Dataset``` ist insbesondere aus folgendem Grund vergleichsweise effizient: Während des Trainings wird stets nur ein Bruchteil eines Datensatzes im Arbeitsspeicher gehalten. Benötigte Daten werden sequentiell aus dem Festplattenspeicher in den Arbeitsspeicher geladen und nach ihrer Verarbeitung wieder aus letzterem entfernt. Die Methode ```prefetch``` sorgt dafür, dass eine zu spezifizierende Anzahl in kommenden Schritten zu verarbeitender Datenstapel stets in einem Puffer im Arbeitsspeicher vorgehalten wird. Auf diese Weise werden Verzögerungen zwischen der Verarbeitung von Datenstapeln vermieden. Weitere Informationen finden Sie unter https://www.tensorflow.org/api_docs/python/tf/data/Dataset.
# 
# 10. Definieren Sie zunächst ```batch_size = 32``` (wird später noch benötigt). Wenden Sie dann, in der angegebenen Reihenfolge, die Methoden ```repeat``` (ohne Argument), ```shuffle``` (mit den Argumenten ```buffer_size=1024``` und ```seed=0```), ```batch``` (mit dem Argument ```batch_size=batch_size```) und ```prefetch``` (mit dem Argument ```buffer_size=1```) auf ```train_ds``` an. Beachten Sie, dass jede der genannten Methoden einen neuen Datensatz zurückgibt, anstatt einen bestehenden Datensatz zu verändern.

# In[51]:


# IHR CODE
batch_size = 32
train_ds = train_ds.repeat().shuffle(1024, seed = 0).batch(batch_size).prefetch(1)


# 11. Führen Sie jetzt folgenden Code aus:

# In[52]:


for item in train_ds.take(1):
    print(item[0].shape)
    plt.imshow(item[0][18, ...], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f'Label: {item[1][18]}')
    plt.show()


# **Checkpoint:** Sie sollten als Output ```(32, 28, 28, 1)``` sowie ein Bild mit einer handgeschriebenen Sieben erhalten.

# 12. Transformieren Sie ```val_ds``` und ```test_ds``` auf die gleiche Art und Weise, lassen Sie lediglich den Aufruf von ```shuffle``` weg (das Durchmischen von Validierungs- und Testdaten hätte keinen Effekt auf das Training) und verwenden Sie für ```test_ds``` das Argument ```batch_size=1```.
# 
# Sie können all diese Aufrufe in einer einzigen Codezeile pro Datensatz unterbringen oder für jede Transformation eine neue Zeile verwenden.

# In[53]:


# IHR CODE
val_ds = val_ds.repeat().batch(batch_size).prefetch(1)
test_ds = test_ds.repeat().batch(1).prefetch(1)


# ## 1.2 Trainingsphase
# 
# Nachdem Sichtung und Vorverarbeitung der Daten abgeschlossen sind, wollen wir nun mit ```tensorflow.keras``` ein neuronales Netz definieren und dieses später auf den ```mnist``` Daten trainieren.

# ### Modelldefinition
# 
# 13. Nutzen Sie die Keras Sequential API, um ein *vollständig verbundenes* Modell ```model``` mit folgender Architektur zu definieren:
# 
#  - Eine (parameterfreie) Eingabeschicht zur Vektorisierung eingegebener Bilddaten,
#  - eine verdeckte Schicht mit 10 Neuronen und ReLU-Aktivierungsfunktion,
#  - eine Ausgabeschicht mit passender Neuronenanzahl und Softmax-Aktivierungsfunktion.
# 
#     
# Hilfreiche Informationen finden Sie unter https://keras.io/models/sequential/. Verwenden Sie für die Eingabeschicht ```keras.layers.Flatten()``` und setzen Sie das Argument ```input_shape```, um dem Modell das Datenformat mitzuteilen. Nutzen Sie in diesem Zusammenhang erneut ```info.features["image"].shape```.

# In[54]:


# IHR CODE
model = tf.keras.Sequential()
input_layer = keras.layers.Flatten(input_shape = info.features['image'].shape)
hidden_layer = keras.layers.Dense(10, activation='relu')
output_layer = keras.layers.Dense(10, activation='softmax')
model.add(input_layer)
model.add(hidden_layer)
model.add(output_layer)


# ### Modellvisualisierung
# 
# 14. Verwenden Sie nun die Methode ```model.summary```, um eine Zusammenfassung Ihres Modells ausgeben zu lassen.

# In[55]:


# IHR CODE
model.summary()


# **Checkpoint:** Das Ende des Outputs von ```model.summary()``` sollte folgendermaßen aussehen:
# 
# ```
# Total params: 7,960
# Trainable params: 7,960
# Non-trainable params: 0
# ```

# ### Training
# 
# 15. Nutzen Sie nun die Methode ```model.compile```, um Ihr Modell zu kompilieren. Verwenden die Verlustfunktion ```keras.losses.sparse_categorical_crossentropy```, den Optimierungsalgorithmus ```keras.optimizers.SGD``` mit Lernate 0.01 sowie die Metrik ```keras.metrics.sparse_categorical_accuracy```.

# In[56]:


# IHR CODE
model.compile(keras.optimizers.SGD(0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


# 16. Rufen Sie nun die Methode ```model.fit``` mit Rückgabewert ```history``` auf, um Ihr Modell für 100 Epochen mit Trainingsdaten ```train_ds``` zu trainieren. Teilen Sie der Methode über das Argument ```steps_per_epoch``` mit, wie viele Stapel pro Epoche zu verarbeiten sind. Diese Anzahl entspricht der Summe aller Einträge von ```train_occurences``` geteilt durch ```batch_size``` (aufgerundet). Übergeben Sie außerdem Validierungsdaten ```val_ds``` und, analog zu ```steps_per_epoch``` im Falle der Trainingsdaten, das Argument ```validation_steps```.

# In[57]:


# IHR CODE
val_steps = round(np.sum(val_occurences)/batch_size)
train_steps = round(np.sum(train_occurences)/batch_size)
test_steps = round(np.sum(test_occurences)/1)

history = model.fit(train_ds, epochs = 100, batch_size = batch_size, 
                steps_per_epoch = train_steps,
                validation_data = val_ds, validation_steps = val_steps)


# 17. Speichern Sie Ihr Modell nach Ende des Trainings unter dem Dateinamen ```my_first_model.h5```.

# In[58]:


# IHR CODE
model.save('my_first_model.h5')


# 18. Laden Sie das gespeicherte Modell, indem Sie folgenden Code ausführen:

# In[59]:


# IHR CODE
model = keras.models.load_model('my_first_model.h5')


# **Checkpoint:** Das Laden Ihres Modells sollte fehlerfrei funktioniert haben.

# ### Lernkurven
# 
# Als Rückgabewert erhalten Sie von ```model.fit``` ein Objekt ```history``` vom Typ ```keras.callbacks.History```. Dieses verfügt insbesondere über die Attribute ```history.history``` und ```history.params```. Bei ersterem handelt es sich um ein ```dictionary``` mit Schlüsseln ```"loss"```, ```"accuracy"```, ```"val_loss"``` sowie ```"val_accuracy"```. Die zugehörigen Werte sind jeweils Listen, welche die den Schlüsseln entsprechenden Kennzahlen zum Ende aller Epochen enthalten.
# 
# 19. Konstruieren Sie aus ```history.history``` ein Objekt ```history_frame``` der Klasse ```pd.DataFrame```. Rufen Sie dann nacheinander ```history_frame.plot()``` und ```plt.show()``` auf.
# 
# Belassen Sie den bereits eingefügten Code am Anfang der Zelle. Die Verwendung des Paketes ```seaborn``` sorgt an dieser Stelle lediglich für eine standardisierte und vergleichsweise ansprechende Darstellung der zu generierenden Plots.

# In[60]:


sns.set()
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5})

# IHR CODE
history_frame = pd.DataFrame(history.history)
history_frame.plot()
plt.show()


# ### Modellevaluation
# 
# Während des Trainings konnten Sie bereits die Entwicklung der Werte ```loss```, ```accuracy```, ```val_loss``` und ```val_accuracy``` zum Ende jeder Epoche verfolgen. Anschließend haben Sie mittels ```history``` die zugehörigen Lernkurven visualisiert. Um Ihr Modell manuell auf einem Datensatz auszuwerten, können Sie außerdem die Methode ```model.evaluate``` aufrufen.
# 
# 20. Wenden Sie ```model.evaluate``` jeweils einmal an, um die Performance Ihres Modells auf den Trainings- sowie den Validierungsdaten zu evaluieren.

# In[61]:


# IHR CODE
train_loss, train_accuracy = model.evaluate(train_ds, batch_size = batch_size, steps = train_steps)
val_loss, val_accuracy = model.evaluate(val_ds, batch_size = batch_size, steps = val_steps)
print(train_accuracy)
print(val_accuracy)


# ***Herzlichen Glückwunsch!***
# 
# Sie haben ein erstes Modell zum Zweck der Mehrfachklassifikation auf ```mnist``` trainiert und das ist ein wichtiger Meilenstein!

# ## 1.3 Nach dem Training
# 
# Sie haben im obigen Projektabschnitt ein erstes Modell trainiert. Abgesehen von der gewählten Modellarchitektur und den betrachteten Kennzahlen ```loss```, ```accuracy```, ```val_loss``` und ```val_accuracy```, wissen wir an diesem Punkt jedoch wenig über die Funktionsweise unseres Modells. In diesem Abschnitt werden Sie zwei Techniken zur Visualisierung und Interpretation von Deep-Learning-Modellen kennenlernen, und anschließend die Modellarchitektur mittels Hyperparameter-Tuning verfeinern.

# ### Vorwärtsausbreitung und Modellvorhersage
# 
# Bevor wir zu einer ersten Visualisierungstechnik übergehen, betrachten wir die Vorwärtsausbreitung eines eingegebenen Bildes durch ein vollständig verbundenes neuronales Netz. Für ein Modell mit einer verdeckten Schicht und $\mathrm{ReLU}$- bzw. $\mathrm{softmax}$-Aktivierungsfunktion sieht diese Abbildung folgendermaßen aus:
# 
# $f(\mathbf x) = \underbrace{\mathrm{softmax}(\mathbf W^{[2]} \underbrace{\mathrm{ReLU}(\mathbf W^{[1]}\mathbf x + \mathbf b^{[1]})}_{\mathbf a^{[1]}} + \mathbf b^{[2]})}_{\mathbf a^{[2]}}$
# 
# In einem ersten Schritt wollen wir nun $f(\mathbf x)$ mithilfe von ```model``` berechnen. Der direkte und einfachste Weg hierfür ist grundsätzlich die Nutzung der Methode ```model.predict```. An dieser Stelle wollen wir dennoch einmal jede Schicht von ```model``` einzeln auswerten, um den korrekten Funktionswert zu berechnen. Auf die einzelnen Schichten können Sie über ```model.layers[l]``` zugreifen und dieses Objekt als Funktion verwenden. 
# 
# 21. Vervollständigen Sie den Code in der folgenden Zelle, indem Sie an den mit ```None``` markierten Stellen Ihre Lösung einfügen. Die Variable ```output``` soll letztlich $f(\mathbf x)$ enthalten und ```prediction``` die dementsprechend vorherzusagende Klasse.

# In[62]:


model.layers


# In[63]:


# get one batch from train_ds and select first item
for item in train_ds.take(1):
    demo_image = item[0][:1, ...]
    demo_label = item[1][0]
    
# compute hidden activations, model output and predicted_label
flattened_input = model.layers[0](demo_image) # IHR CODE
hidden_activations = model.layers[1](flattened_input) # IHR CODE
output = model.layers[2](hidden_activations) # IHR CODE
predicted_label = model.predict(demo_image) # IHR CODE
    
# plot demo_image, print demo_label
plt.imshow(demo_image[0, ...], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.xlabel('Input demo_image')
plt.show()
print(f'True demo_label: {demo_label}\n')

# print my predicted class probabilities and demo_label
print('My Predicted Class Probabilities:\n')
for i in range(output.shape[1]):
    print(f'{i}: {output[0][i]:.5f}')
print(f'\nMy Predicted demo_label: {predicted_label}')

# print model predicted class probabilities
model_output = model.predict(demo_image)
print('\nModel Predicted Class Probabilities:\n')
for i in range(model_output.shape[1]):
    print(f'{i}: {model_output[0][i]:.5f}')


# **Checkpoint:** Die Werte ```My Predicted Class Probabilities``` sollten genau mit ```Model Predicted Class Probabilities``` übereinstimmen. Außerdem muss der Wert ```My Predicted Label``` mit dem Index des maximalen Eintrags von ```My Predicted Class Probabilities``` übereinstimmen.

# ### Visualiserung von Neuronen
# 
# Oben haben wir uns die Modellausgabe $f(\mathbf x) = \mathbf a^{[2]}$ angesehen. Hier wollen wir nun die Gewichte $\mathbf W^{[1]}$ und den Aktivierungsvektor $\mathbf a^{[1]}$ in der verdeckten Schicht betrachten. Die transponierte Gewichtungsmatrix $(\mathbf W^{[\ell]})^T$ erhalten Sie über ```model.layers[l].weights[0]``` (und den zugehörigen transponierten Biasvektor bei Bedarf über ```model.layers.weights[l]```).
# 
# 22. Ergänzen Sie den Code in der folgenden Zelle zunächst derart, dass nach dem ersten Schritt in ```hidden_weights``` die Matrix $(\mathbf W^{[1]})^T$ gespeichert ist. Dann ist ```hidden_weights``` ein Objekt vom Typ ```tf.Tensor``` (oder einer davon abgeleiteten Klasse). Nutzen Sie anschließend die Methode ```tf.Tensor.numpy```, um ```hidden_weights``` in ein Objekt vom Typ ```np.ndarray``` zu konvertieren. Füllen Sie nun noch die zwei durch ```None``` gekennzeichneten Lücken in der ```for```-Schleife darunter. Dort soll folgendes passieren: Mit dem ersten Aufruf von ```plt.imshow``` soll immer das Bild ```demo_image``` von oben angezeigt werden. Mit dem zweiten Aufruf von ```plt.imhsow``` soll die ```i```-te Zeile von $\mathbf W^{[1]}$ als Bild angezeigt werden. Bringen Sie die genannte Zeile jeweils in das Format ```(28, 28)```, was genau dem Format von ```demo_image``` entspricht.

# In[64]:


# get hidden weights
hidden_weights = model.layers[1].weights[0] # IHR CODE
hidden_weights = hidden_weights.numpy() # IHR CODE

# plot hidden weights
print('\nHidden Weights:\n')
plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(demo_image[0,...], cmap=plt.cm.gray, norm=mcolors.TwoSlopeNorm(vcenter=0.), alpha=.5) # IHR CODE
    plt.imshow(hidden_weights[:,i].reshape(28,28), cmap=plt.cm.coolwarm, norm=mcolors.TwoSlopeNorm(vcenter=0.), alpha=.5) # IHR CODE
    plt.xlabel(f'{hidden_activations[0][i]:.2f}')
    plt.xticks([])
    plt.yticks([])
plt.show()


# **Checkpoint**: Wenn Sie die letzte Zelle korrekt ergänzt haben, sollten Sie nun ein Bild pro Neuron in der verdeckten Schicht sehen. In jedem Bild ist eine Überlagerung der eingehenden Gewichte des Neurons (Grau entspricht Null, Rot einem positiven und Blau einem negativen Wert) und des Bildes ```demo_image``` (helle Grautöne entsprechen positiven Werten) zu sehen. Unter jedem Bild sehen Sie außerdem die Aktivierung $a^{[1]}_i$ des jeweiligen Neurons für die Eingabe $\mathbf x=$```demo_image```. Stellen Sie sicher, dass Sie den Zusammenhang zwischen dem Inhalt der Bilder und den zugehörigen Aktivierungswerten verstanden haben und diskutieren Sie darüber innerhalb Ihres Teams.

# ### Interpretation von Modellvorhersagen
# 
# Nachdem wir oben den Zusammenhang zwischen Modellparametern und Aktivierungen in der verdeckten Schicht unseres Modells visualisiert haben, wollen wir nun eine Methode implementieren, welche einen Erklärungsansatz für den Zusammenhang zwischen dem durch das Modell vorhergesagten Label ```predicted_label``` und dem eingegebenen Bild ```demo_image``` liefert. Diese Methode names *Layer-Wise Relevance Propagation* (LRP) soll letztlich einen sogenannten Relevanzwert für jeden Pixel des eingegebenen Bildes bzgl. des vorhergesagten Labels liefern. Auf dem Weg dahin werden Relevanzwerte für jedes Neuron in jeder Schicht generiert und sukzessive bis in die Eingabeschicht durch das neuronale Netz zurückpropagiert.
# 
# Initial werden die Relevanzwerte für die Ausgabeschicht mit der vorhergesagten Wahrscheinlichkeit für ```predicted_label``` gleichgesetzt ($k$ symbolisiert hier den entsprechenden Index):
# 
# $\mathbf r^{[L]} := \begin{pmatrix}0\\\vdots\\\mathbf a^{[L]}_k\\\vdots\\0\end{pmatrix}\in \mathbb R^{n_L}$ 
# 
# Anschließend werden die Relevanzwerte $\mathbf r^{[L-1]}, \mathbf r^{[L-2]}, \dots, \mathbf r^{[1]}, \mathbf r^{[0]}$ (in dieser Reihenfolge) für alle weiteren Schichten folgendermaßen berechnet:
# 
# $\displaystyle \mathbf r^{[\ell-1]} := \alpha \ \frac{\mathbf a^{[\ell -1]}\odot \mathbf W^{[\ell]\top+}}{\mathbf a^{[\ell -1]\top} \mathbf W^{[\ell]\top+}} \ \mathbf r^{[\ell]} \ - \ \beta \ \frac{\mathbf a^{[\ell -1]}\odot \mathbf W^{[\ell]\top-}}{\mathbf a^{[\ell -1]\top} \mathbf W^{[\ell]\top-}} \ \mathbf r^{[\ell]}$
# 
# Dabei bezeichnet
# 
# $\mathbf W^{[\ell]\top+} := \max(\mathbf W^{[\ell]\top}, \mathbf 0)$
# 
# den Positivteil und
# 
# $\mathbf W^{[\ell]\top-} := -\min(\mathbf W^{[\ell]\top}, \mathbf 0)$
# 
# den Negativteil von $\mathbf W^{[\ell]\top} \in \mathbb R^{n_{\ell-1}\times n_{\ell}}$, wobei Maximum und Minimum jeweils komponentenweise berechnet werden. Weiterhin ist 
# 
# $\mathbf a^{[\ell -1]}\odot \mathbf W^{[\ell]\top+}\in \mathbb R^{n_{\ell-1}\times n_{\ell}}$
# 
# das Ergebnis der komponentenweisen Multiplikation von $\mathbf a^{[\ell -1]}\in \mathbb R^{n_{\ell-1}}$ mit den Spalten von $\mathbf W^{[\ell]\top+}$ (und entsprechend für den Negativteil). Analog symbolisiert der Bruchstrich die komponentenweise Division der Zeilen von von $\mathbf a^{[\ell -1]}\odot \mathbf W^{[\ell]\top+}$ durch $\mathbf a^{[\ell -1]\top} \mathbf W^{[\ell]\top+}\in \mathbb R^{1\times n_{\ell}}$ (analog für den anderen Bruch). Im Ergebnis repräsentiert jeder Bruch eine $n_{\ell-1}\times n_{\ell}$ Matrix, die dann mit $\mathbf r^{[\ell]}\in \mathbb R^{n_{\ell}\times 1}$ multipliziert wird.
# 
# Die Skalare $\alpha, \beta \geq 0$ werden mit $\alpha - \beta = 1$ gewählt. Damit lässt sich zeigen, dass
# 
# $\sum_j \mathbf r^{[L]}_j = \sum_j \mathbf r^{[L-1]}_j = \sum_j \mathbf r^{[L-2]}_j = \dots = \sum_j \mathbf r^{[1]}_j = \sum_j \mathbf r^{[0]}_j$
# 
# gilt. LRP ist also *relevanzerhaltend*.
# 
# Die obige Darstellung ist so kompakt wie möglich gehalten, besitzt aber eine plausible Interpretation: Jedes Neuron in einer Schicht leistet anteilig einen positiven oder einen negativen Beitrag zu jeder Neuronenaktivierung in der nachfolgenden Schicht. Der Relevanzwert eines Neurons ergibt sich als Summe all dieser Beiträge, und genau das spiegelt die Abbildungsvorschrift im Rahmen von LRP wieder.
# 
# Wenn Sie Fragen zu LRP haben oder mehr erfahren möchten, dann kommen Sie idealerweise in die offene Sprechstunde am 29.10.2020 oder werfen Sie einen Blick in den frei zugänglichen Artikel https://www.sciencedirect.com/science/article/pii/S1051200417302385?via%3Dihub.
# 
# Nun zur Aufgabe:
# 
# 23. Vervollständigen Sie die folgende Funktion an den mit ```None``` markierten Stellen, sodass diese genau der oben definierten Abbildungsvorschrift entspricht. Berücksichtigen Sie dabei erneut, dass ```model.layers[l].weights[0]``` der transponierten Gewichtungsmatrix $\mathbf W^{[\ell]\top}$ entspricht. Wenn Sie sich bei der Dimension von Tensoren unsicher sind, können Sie ```tf.Tensor.shape``` für Zwischenausgaben nutzen. Beachten Sie außerdem, dass die Funktion LRP für vollständig verbundene $\mathrm{ReLU}$-Netzwerke beliebiger Tiefe $L$ mit $\mathrm{softmax}$-Aktivierungsfunktion in der Ausgabeschicht funktionieren soll.

# In[ ]:


activations = [model.layers[0](demo_image)]
for l in range(1, 3):
        activations.append(model.layers[l](activations[-1]))


# In[ ]:


a = activations[0]


# In[ ]:


w = model.layers[1].weights[0]


# In[ ]:


scores = [tf.transpose(activations[-1])]


# In[ ]:


scores[0].shape


# In[ ]:


def lrp(alpha, beta, model, x):
    
    # get number of layers (including input layer)
    num_layers = len(model.layers)
    
    # get activations
    activations = [model.layers[0](x)]
    for l in range(1, num_layers):
        activations.append(model.layers[l](activations[-1]))
    
    # compute scores
    scores = [tf.transpose(activations[-1])]
#     print(tf.transpose(activations[-1]).shape)
    c = tf.squeeze(tf.math.argmax(scores[0]))
    for l in range(num_layers-2, -1, -1):
#         print(l)
        w = model.layers[l+1].weights[0]
        if l == num_layers-2:
            w_pos = tf.clip_by_value(w[:, c:c+1], 0, np.inf)
            w_neg = -tf.clip_by_value(w[:, c:c+1], -np.inf, 0)
            s = scores[0][c:c+1]
            scores[0] = s
        else:
            w_pos = tf.clip_by_value(w, 0, np.inf)
            w_neg = -tf.clip_by_value(w, -np.inf, 0)
            s = scores[0]
        a = activations[l]
        
        score_pos = alpha * tf.matmul((tf.transpose(a)* w_pos)/(tf.matmul(a, w_pos)) , s) # IHR CODE
        score_neg = beta * tf.matmul((tf.transpose(a)*w_neg)/(tf.matmul(a, w_neg)),  s) # IHR CODE
        score_total = score_pos - score_neg  # IHR CODE
        scores.insert(0, score_total)
        
    return scores, activations


# 24. Wenden Sie nun LRP mit $\alpha=2$ und $\beta=1$ auf ```model``` und ```demo_image``` an. Speichern Sie in ```scores_input``` die Relevanzwerte für die Eingabeschicht und bringen Sie diese in das Format $28\times 28$.

# In[ ]:


scores, _ = lrp(2, 1, model, demo_image) # IHR CODE
scores_input = tf.reshape(scores[0],[28,28]) # IHR CODE

print('Predicted Probability for Predicted Class (Relevance in Output Layer):\n')
print(f'{tf.squeeze(scores[-1]).numpy():.5f}\n')

print('\nSum of all Relevances in Input Layer:\n')
print(f'{tf.squeeze(tf.reduce_sum(scores[0])).numpy():.5f}\n')

plt.figure()
plt.imshow(demo_image[0, ...], cmap=plt.cm.gray, norm=mcolors.TwoSlopeNorm(vcenter=0.), alpha=.5)
plt.imshow(scores_input, cmap=plt.cm.coolwarm, norm=mcolors.TwoSlopeNorm(vcenter=0.), alpha=.5)
plt.xticks([])
plt.yticks([])
plt.show()


# **Checkpoint**: Die beiden oben ausgegebenen Zahlenwerte stimmen überein, wenn Sie LRP korrekt implementiert haben. Können Sie einen Zusammenhang mit den Bildern aus dem Abschnitt *Visualisierung von Neuronen* feststellen? Diskutieren Sie diese Frage innerhalb Ihres Teams.

# ### Hyperparameter Tuning
# 
# Das oben trainierte Modell verfügt über lediglich eine verdeckte Schicht mit einer vergleichsweise kleinen Anzahl von zehn Neuronen. In diesem Abschnitt wollen wir weitere Modelle trainieren, um eine möglichst hohe Genauigkeit ```val_accuracy``` auf den Validierungsdaten zu erzielen.
# 
# 25. Trainieren Sie mindestens zehn weitere Modelle. Sie können dabei die Anzahl der verdeckten Schichten, die Anzahl ```units``` der jeweils enthaltenen Neuronen, den optionalen Parameter ```dropout``` in ```keras.layers.Dense```, den zu verwendenden Optimierungsalgorithmus ```optimizer``` samt Lernrate ```learning_rate``` und die Epochenanzahl ```epochs``` variieren. Sie können die genannten Hyperparameter von Hand oder auf eine Art Ihrer Wahl systematisch wählen. Ändern Sie nichts an der Eingabe- sowie der Ausgabeschicht und verwenden Sie weiterhin $\mathrm{ReLU}$ als Aktivierungsfunktion in allen verdeckten Schichten. Speichern Sie Ihr hinsichtlich ```val_accuracy``` bestes Modell als ```my_best_model.h5```.
# 
# 
# 26. Machen Sie sich unter https://www.tensorflow.org/tutorials/keras/keras_tuner mit *Keras Tuner* vertraut und versuchen Sie, ```val_accuracy``` mit dessen Hilfe weiter zu verbessern. Speichern Sie Ihr bestes Modell als ```my_best_model_kt.h5```.
# 
# 
# 27. Evaluieren Sie Ihr insgesamt bestes Modell auf den Testdaten ```test_ds``` und wenden Sie anschließend LRP mit diesem Modell auf ```demo_image``` und mindestens zehn weitere Bilder aus ```test_ds``` an. Visualisieren Sie außerdem mindestens zehn Neuronen aus der ersten verdeckten Schicht Ihres besten Modells auf die oben beschriebene Weise.

# ### Herzlichen Glückwunsch zur Fertigstellung von Projekt 1!
# 
# Speichern Sie dieses Notebook inklusive der generierten Ausgaben ab und senden Sie die Datei mitsamt der gespeicherten Modelle ```my_first_model.h5```, ```my_best_model.h5``` und ```my_best_model_kt.h5``` in einem komprimierten Ordner ```[nachname_teammitglied1]_[nachname_teammitglie2].zip``` (statt ```.zip``` können Sie auch ein anderes geeignetes Format verwenden) an die oben genannten E-Mail-Adressen. Finden Sie sich außerdem am 13.11.2020 pünktlich zum vereinbarten Termin zwecks Vorstellung Ihrer Ergebnisse in dem mitgeteilten Web-Konferenzraum ein (alle Teammitglieder müssen anwesend sein).

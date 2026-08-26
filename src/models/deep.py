import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from scikeras.wrappers import KerasClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from tensorflow.keras.callbacks import EarlyStopping
from MLPipeline import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import gc
from sklearn.utils.class_weight import compute_class_weight
from utils import ASSETS_DIR

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Applica il memory growth a tutte le GPU rilevate
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Allocazione dinamica della memoria GPU attivata.")
    except RuntimeError as e:
        # Questo errore capita se TensorFlow ha già inizializzato la GPU
        print(e)




def build_sprase_autoencoder(input_dim=130):
    inputs = layers.Input(shape=(input_dim,))

    encoder = layers.Dense(64, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01))(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dropout(0.3)(encoder)

    encoder = layers.Dense(64, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01))(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dropout(0.3)(encoder)


    encoder = layers.Dense(32, activation="gelu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.1))(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Dropout(0.2)(encoder)

    sparse_bottleneck = layers.Dense(16, activation="relu", kernel_initializer="he_normal", name='sparse_bottleneck_layer', activity_regularizer=regularizers.l1(1e-5))(encoder)


    #Decoder

    decoder = layers.Dense(16, activation="gelu", kernel_initializer="he_normal", activity_regularizer=regularizers.l2(0.01))(sparse_bottleneck)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dropout(0.2)(decoder)

    decoder = layers.Dense(32, activation="gelu", kernel_initializer="he_normal", activity_regularizer=regularizers.l2(0.01))(sparse_bottleneck)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dropout(0.2)(decoder)

    decoder = layers.Dense(64, activation="gelu", kernel_initializer="he_normal", activity_regularizer=regularizers.l2(0.01))(sparse_bottleneck)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Dropout(0.3)(decoder)



    output = layers.Dense(3, activation='softmax')(decoder)

    model = models.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )

    return model


class ClearMemoryCallback(Callback):
    def on_train_end(self, logs=None):
        """
        Questo metodo scatta in automatico nell'istante in cui 
        l'EarlyStopping ferma l'addestramento del fold.
        """
        gc.collect()          # Forza Python a svuotare la RAM di sistema
        K.clear_session()     # Distrugge il grafo di TensorFlow, liberando la GPU

# 2. Inizializziamo il nostro spazzino
memory_cleaner = ClearMemoryCallback()

df = pd.read_parquet(ASSETS_DIR / 'final_df.parquet')


# 1. Separiamo la matrice delle Feature (X) dal Target (y)
X = df.drop(columns=['TARGET'])
X.drop(columns=['LBDEVAL'], inplace=True, errors='ignore')
y = df['TARGET']

# 2. TRAIN-TEST SPLIT (80% Train, 20% Test)
# stratify=y è fondamentale per mantenere le stesse percentuali di malati nei due set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Buchi (NaN) iniziali in X_train: {X_train.isna().sum().sum()}")

# 3. CONFIGURAZIONE DEL KNN IMPUTER

imputer = KNNImputer(n_neighbors=7, weights='distance')

# 4. ADDESTRAMENTO E TRASFORMAZIONE
# Il modello "impara" le distribuzioni SOLO da X_train
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Il modello applica quanto imparato su X_test (senza barare)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 5. ARROTONDAMENTO PER DATI CLINICI DISCRETI
# Riportiamo le medie del KNN a numeri interi (es. 0.66 diventa 1.0)
X_train_imp = X_train_imp.round()
X_test_imp = X_test_imp.round()

print(f"Buchi (NaN) finali in X_train: {X_train_imp.isna().sum().sum()}")
print(f"Buchi (NaN) finali in X_test: {X_test_imp.isna().sum().sum()}")



early_stopping = EarlyStopping(
    monitor='val_loss',         # Guarda l'errore sul set di validazione interno
    patience=10,                # Aspetta 10 epoche senza miglioramenti prima di arrendersi
    restore_best_weights=True   # IL SEGRETO: Riporta indietro la rete alla sua forma perfetta
)


modello_visivo = build_sprase_autoencoder(input_dim=128)
# Genera un'immagine PNG del tuo modello
tf.keras.utils.plot_model(
    modello_visivo,
    to_file='architettura_autoencoder_selu.png',
    show_shapes=True,           # FONDAMENTALE: Mostra come i dati si restringono (es. 131 -> 64 -> 32 -> 16)
    show_layer_names=True,      # Mostra i nomi dei layer (es. 'sparse_bottleneck_layer')
    show_layer_activations=True,# Mostra le attivazioni (vedrai GELU sui bordi e ReLU al centro)
    dpi=300                     # Altissima risoluzione (perfetta per la stampa)
)

print("Immagine 'architettura_autoencoder_selu.png' salvata con successo!")

with tf.device('/GPU:0'):
    
    keras_mlp = KerasClassifier(
        model=build_sprase_autoencoder,
        epochs=100, # Iniziamo con 50 per un test rapido
        batch_size=128,
        verbose=1, # 1 ti mostrerà la barra di caricamento Keras per ogni epoca
        random_state=42,
        validation_split=0.15,
        callbacks=[early_stopping, memory_cleaner],

    )

    pipeline_tf = ImbPipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', keras_mlp)
    ])

    print("Avvio K-Fold Cross Validation con Rete Neurale (Auto-tuning delle epoche attivato)...")

    # 4. Esecuzione tramite la TUA funzione
    # Usa i dati X_iter e y_iter che abbiamo pulito prima col taglio basato sulla gravità
    # La funzione calcolerà e stamperà tutte le tue metriche personalizzate
    y_true_bin, y_proba_tf = train_model_evaluate(
        X=X_train_imp, 
        y=y_train, 
        model=pipeline_tf, 
        use_cv=True,
        cv_strategy=cv_strategy,
    )
    print("Test completato!")


    print("Avvio  Rete Neurale (Auto-tuning delle epoche attivato)...")
    
    # 4. Esecuzione tramite la TUA funzione
    # Usa i dati X_iter e y_iter che abbiamo pulito prima col taglio basato sulla gravità
    # La funzione calcolerà e stamperà tutte le tue metriche personalizzate
    y_true_bin, y_proba_tf, y_pred = train_model_evaluate(
        X=X_train_imp, 
        y=y_train, 
        model=pipeline_tf, 

        X_test=X_test_imp,
        y_test=y_test
    )
    print("Test completato!")


    cm = confusion_matrix(y_test, y_pred)

    etichette = ['Sani (0)', 'Alzheimer (1)', 'Lewy Body (2)']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=etichette, yticklabels=etichette,
                linewidths=1, linecolor='black')

    plt.title("confusion amtrix", fontsize=14, pad=15)
    plt.ylabel('Diagnosi Reale (Medico)', fontsize=12, fontweight='bold')
    plt.xlabel(f'Previsione ({type(pipeline_tf).__name__})', fontsize=12, fontweight='bold')
    plt.show()

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=etichette))

    plot_reliability_diagram(y_true_bin, y_proba_tf[:, 2])
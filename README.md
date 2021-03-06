* # Distilled-Single-Shot-Detector (DSSD): un nuovo modello di guida autonoma ad alta inferenza
La seguente repository contiene il progetto della mia tesi magistrale. Il tema principale è rivolto verso lo sviluppo di un nuovo modello di rete neurale utile nel contesto automotive.
Lo studio vede una sua evoluzione grazie ad una nota problematica che sta influenzando, in maniera negativa, l'intero settore industriale. La crisi, soprannominata `"Crisi dei Semiconduttori"`, ha colpito maggiormente tutte le aziende produttrici di veicoli. Quest'ultime si sono ritrovate con una perdita complessiva di circa 210 Miliardi di dollari e, purtroppo, tale numerò tenderà ad aumentare anche nel prossimo anno. 
L'assenza, o la carenza, di semiconduttori, da poter installare all'interno del comparto `ECM` (Engine Control Module), costringe alcune filiere ad interrompere la loro produzione in attesa di un nuovo lotto. 
Ad essere le più colpite, sono le nuove auto elettriche che a bordo possiedono i Sistemi Avanzati di Assistenza alla guida (aka `ADAS`).
La soluzione proposta da tale elaborato, mira a favorire il riuso di sitemi con risorse computazionali inferiori rispetto a quelli comunemente richiesti. 

### Tabella dei contenuti
* [Lavoro di tesi](#lavoro-di-tesi)
* [Tecniche di Compressione e Ottimizzazione](#tecniche-di-compressione-e-ottimizzazione)
* [Pruning](#pruning)
* [Knowledge Distillation](#knowledge-distillation)
* [Metodologia proposta](#metodologia-proposta)
* [Modelli Insegnante e Studenti](#modelli-insegnante-e-studenti)
* [Distilled Single Shot Detector DSSD](#distilled-single-shot-detector-dssd)
* [Architetture di riferimento](#architetture-di-riferimento)
* [Dataset](#dataset)
* [Risultati sperimentali Pruning](#risultati-sperimentali-pruning)
* [Risultati sperimentali DSSD Dimensioni](#risultati-sperimentali-dssd-dimensioni)
* [Risultati sperimentali DSSD Inferenza](#risultati-sperimentali-dssd-inferenza)
* [Esempi visivi SSD Vs DSSD](#esempi-visivi-ssd-vs-dssd)
* [Conclusioni e Sviluppi futuri](#conclusioni-e-sviluppi-futuri)

## Lavoro di tesi

Lo studio della tesi si è concentrato sulla ricerca e sull’implementazione di varie tecniche di compressione e, allo stesso tempo, di ottimizzazione, in grado di offrire supporto allo sviluppo di un nuovo modello di guida autonoma efficiente e ad alta velocità di inferenza.
Quest’ultimo, deriva dallo studio di due modelli già noti allo stato dell’arte:
1. **MobileNet-V1**: specializzato nel task di Image classification;
2. **Single-Shot-Detector (SSD)**: specializzato nel task di Object Detection.

## Tecniche di Compressione e Ottimizzazione

In letteratura esistono varie tecniche di compressione/ottimizzazione da poter applicare sulle reti neurali profonde. In questo elaborato però, si è preferito optare verso l'utilizzo di tre tecniche, due delle quali ben conosciute allo stato dell'arte. Queste sono:
1. **Pruning** *(Potatura)*: Azzeramento di determinati parametri nella rete;
2. **Knowledge Distillation** *(Conoscenza Distillata)*: Trasferimento della "Conoscenza" da un modello di grandi dimensioni, verso un modello più piccolo;
3. **Metodologia Proposta**: combinazione della tecnica di Knowledge Distillation con l’iper-parametro `width-multiplier α` per la derivazione del modello proposto.

## Pruning

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/pruning%20no%20name.png">
</p>

Partendo dalla prima tecnica di compressione, all'interno della *Pruning* viene definito un **indice di sparsità** utile a poter definire la quantità di paramentri da poter azzerare all'interno di un modello. In questo specifico caso, il modello sottoposto alla tecnica è il già citato modello *Single-Shot-Detector (SSD)*. Su quest'ultimo, vengono applicate tre tipologie di pruning, ognua agente su una specifica parte del modello:
1. **Structured**: rimuove interi filtri (canali);
2. **Unstructured**: rimuove i parametri (es: pesi e bias) in un layer;
3. **Global-Unstructured**: rimuove i parametri su più layer.

Teoricamente, dopo aver azzerato una o più tipologie di parametri, si procede alla loro rimozione per poter ridurre le dimensioni complessive del modello. Purtroppo, ad oggi, non esiste un framework in grado di eseguire questo step, e per framework si intendono *PyTorch* e *TensorFlow*. 

## Knowledge Distillation

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/KD_losses.png">
</p>

La seconda tecnica di compressione, chiamata *Knowledge Distillation*, ha l'obiettivo di trasferire la *conoscenza distillata* da un modello di grandi dimensioni, chiamato *Insegnante*, verso un modello di piccole dimensioni chiamato *Studente*. 
Per far ciò, la Knowledge Distillation si basa su tre elementi chiave:
1. **Temperatura (T)**: iper-parametro legato al livello di generalità  presente all'interno del modello Studente;
2. **Soft-Targets** probabilità derivanti dall'applicazione della *Temperatura* sui *logits* delle *Softmax* presenti in ognuno dei due modelli: <br />
 ![](https://latex.codecogs.com/svg.image?q_j&space;=&space;\frac{e^{z_j/T}}{\sum_{k=1}^K&space;e^{z_k/T}})
3. **Perdita complessiva**: formata dalla somma della perdita dell'Insegnante e dello Studente: <br />
![](https://latex.codecogs.com/svg.image?L=&space;L_{hard}&plus;T^2L_{soft})

## Metodologia proposta

La terza e ultima tecnica di compressione riguarda proprio quella proposta. Questa si basa nell'esecuzione di sei differenti step al termine de quali si otterrà un modello utile per l'attività di *Object Detection* nella guida autonoma. 
<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/steps_KD.png", width="600"/>
</p>

## Modelli Insegnante e Studenti

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/table_KD.png", width="500"/>
</p>

I primi quattro step sono riassunti nella tabella in alto.
Nello specifico, questi sono:
1. **Modelli**: tutti i modelli riportati derivano dalla rete **MobileNet-V1** con iper-parametro α* variabile;
2. **Allenamento**: i modelli Insegnante e Studnete base verranno allenati singolarmente su un totale di 1000 epoche;
3. **Distillation**: preso il modello *Studente base*, si ricavano N modelli di *Studenti distillati (Dst)* ad una *Temperatura T* variabile;
4. **Selezione**: viene scelto il miglior modello di Studente Distillato, ricavato da una temperatura T>1, avente le accuratezze comprese nel range delle accuratezze del modello Insegnante e Studente base. In questo caso, il modello ricavato da una tempertatua T=3 verrà selezionato. 

*L'iper-parametro *width-multiplier α* serve a gestire il numero di canali di input e di output in ogni layer convoluzionale del modello. Ad esempio, un modello ricavato con α=0.25 avrà una dimensione pari a 1/4 rispetto a un modello ricavato con un α=1.

## Distilled-Single-Shot-Detector DSSD

Dopo aver selezionato il modello di Studente Distillato, ricavato da una temperatura T=3, lo si va ad integrare, come rete *backbone*, all'interno dell'architettura di rete *Single-Shot-Detector (SSD)*. Tale integrazione, seguita da una modifica dei layer convoluzionali successivi, va a formare l'architettura del modello di rete proposto, che prenderà il nome di: **Distilled-Single-Shot-Detector(DSSD)** (Step 5).

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/SSD_architecture_freeze.png">
</p>

Dopo aver definito il modello proposto, segue l'applicazione di un ultimo processo (Step 6): il *Fine-Tuning*. Quest'ultimo si basa nell'eseguire un allenamento dell'intero modello andandone però a congelare i parametri della rete base (backbone), al fine di preservare la sua accuratezza.

## Architetture di riferimento

Prima di passare ai risultati sperimentali, è bene illustrare quali sono state le architetture utilizzate nel seguente elaborato. 
Nello specifico, queste sono tre, ovvero:
1. **NVidia Jetson Nano**
2. **Google Colaboratory**
3. **Macbook Pro**

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/arch_spec.png">
</p>

 Tutte le architetture sono differenti in termini di performance, dimensione e costo. È importante notare come la scheda embedded Jetson Nano abbia performance, dimensioni e costo inferiori rispetto alle altre due architetture.

## Dataset

Per quanto riguarda le immagini utilizzate per allenare il modello proposto, queste derivano tutte dal dataset [**Open-Images**](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false&c=%2Fm%2F0k4j). Al suo interno troviamo immagini utili per addestrare un modello sia nell'atività di Object Detection che nell'attività di Semantic Segmentation. 
Altri dataset, come Cityscapes, MS COCO e Pascal VOC, sono stati utilizzati per poter formare dei modelli pre-addestrati, risultati utili per poter creare un quadro globale sulle performance di ogni architettura.

## Risultati sperimentali Pruning 

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/pr_results.png">
</p>

Prendendo in riferimento i risultati derivanti dalla tecnica *Globale* di pruning, possiamo notare come fino al 60% di sparsità, la funzione di perdita resti costante. Superata tale soglia, la funzione di perdita iniza a degradare. Cosa sta a significare? Detto in altri termini, il 60% dei parametri azzerati, che in questo caso sono i pesi, sono del tutto inutili all'interno del modello. 

Non avendo la possibilità di eliminarli, possiamo solamente applicare un'utility di compressione ("*gzip*" in questo caso), per poter ridurre le dimensioni del modello. Al 60% di sparsità, gzip riesce a ridurre le dimensioni del modello di circa il 54%, facendolo passare da 28.8 MB (0%) a circa 14.1 MB.

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/pr2_results.png">
</p>

## Risultati sperimentali DSSD Dimensioni

Per quanto riguarda il risultati sperimentali provenienti dal modello proposto, come possiamo notare dalla figura in basso, vi è una riduazione totale pari all'88.73% del numero dei parametri. Tale riduzione fa diminuire il numero dei parametri da circa 8 Milioni, sul modello di partenza SSD, a circa 1 Milione sul modello proposto.

Un comportamento simile si verifica sulle dimensioni dell'intero modello. Un decremento pari all'88.60%, porta il modello ad assumere una dimensione pari a quasi 4 MB, rispetto ai 30.7 MB del modello di partenza SSD. 

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/dssd_results.png">
</p>

## Risultati sperimentali DSSD Inferenza

Oltre alle dimensioni, è stata calcolata la velocità di inferenza, che generalmente si misura in *frames-per-second*, di entrambi i modelli.
Tutti i benchmarks hanno evidenziato un netto sorpasso delle performance derivanti dal modello prosposto rispetto al modello di partenza. Come si può notare, il miglioramento avviene su tutte le architetture utilizzate, maggiormente sulla Jetson Nano. 

Tutti i test sono stati svolti utilizzando sia l'acceleratore TensorRT che il supporto delle librerie OpenCV (cuDNN) su entrambi i modelli.

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/total_speed.png">
</p>

## Esempi visivi SSD Vs DSSD

<p align="center">
    <img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/visual_example.png">
</p>

Dal punto di vista pratico, sottoponendo le stesse immagini ad entrambi i modelli, si può notare come il modello proposto DSSD riesca a generare delle probabilità di predizione che non si discostano tanto da quelle prodotte dal modello di partenza SSD. Stesso ed identico discorso per quanto riguarda la creazione e la forma dell bounding-boxes.

## Conclusioni e Sviluppi futuri
Il seguente elaborato di tesi è incentrato nello sviluppo di una metodologia innovativa che ha permesso di:
1. Combinare la tecnica di compressione **Knowledge Distillation** con **l’iper-parametro α** sulla rete **MobileNet-V1**;
2. Integrare quest’ultima, come *rete base*, nel modello **Single-Shot-Detector (SSD)**;
3. Derivare ed effettuare il *fine-tuning* del nuovo modello, nominato **"Distilled-Single-Shot-Detector (DSSD)"**, che è in grado di:
    - Avere una bassa occupazione della memoria;
    - Essere distribuito su sistemi a limitata capacità computazionale;
    - Incrementare le performance rispetto al modello di partenza;
    - Effettuare il task di object detection nella guida autonoma.

Infine, come sviluppi futuri, sarebbe interessante:
- Eseguire tale tecnica anche su modelli diversi;
- Esntedere il concetto anche per il task di *Semantic Segmentation*;
- Utilizzare una tecnica di compressione standard (es: *Pruning*), per ricavare il modello Studente nella tecnica di *Knowledge Distillation*.
* # Distilled-Single-Shot-Detector (DSSD): un nuovo modello di guida autonoma ad alta inferenza
La seguente repository contiene il progetto della mia tesi magistrale. Il tema principale è rivolto verso lo sviluppo di un nuovo modello di rete neurale utile nel contesto automotive.
Lo studio vede una sua evoluzione grazie ad una nota problematica che sta influenzando, in maniera negativa, l'intero settore industriale. La crisi, soprannominata `"Crisi dei Semiconduttori"`, ha colpito maggiormente tutte le aziende produttrici di veicoli. Quest'ultime si sono ritrovate con una perdita complessiva di circa 210 Miliardi di dollari e, purtroppo, tale numerò tenderà ad aumentare anche nel prossimo anno. 
L'assenza, o la carenza, di semiconduttori, da poter installare all'interno del comparto `ECM` (Engine Control Module), costringe alcune filiere ad interrompere la loro produzione in attesa di un nuovo lotto. 
Ad essere le più colpite, sono le nuove auto elettriche che a bordo possiedono i Sistemi Avanzati di Assistenza alla guida (aka `ADAS`).
La soluzione proposta da tale elaborato, mira a favorire il riuso di sitemi con risorse computazionali inferiori rispetto a quelli comunemente richiesti. 

### Tabella dei contenuti
* [Lavoro di tesi](#thesis-job)
* [Tecniche di Compressione/Ottimizzazione](#techniques)
* [Pruning](#pruning)



## Lavoro di tesi

Lo studio della tesi si è concentrato sulla ricerca e sull’implementazione di varie tecniche di compressione e, allo stesso tempo, di ottimizzazione, in grado di offrire supporto allo sviluppo di un nuovo modello di guida autonoma efficiente e ad alta velocità di inferenza.
Quest’ultimo, deriva dallo studio di due modelli già noti allo stato dell’arte:
1. **MobileNet-V1**: specializzato nel task di Image classification;
2. **Single-Shot-Detector (SSD)**: specializzato nel task di Object Detection.

## Tecniche di Compressione/Ottimizzazione

In letteratura esistono varie tecniche di compressione/ottimizzazione da poter applicare sulle reti neurali profonde. In questo elaborato però, si è preferito optare verso l'utilizzo di tre tecniche, due delle quali ben conosciute allo stato dell'arte. Queste sono:
1. **Pruning** *(Potatura)*: Azzeramento di determinati parametri nella rete;
2. **Knowledge Distillation** *(Conoscenza Distillata)*: Trasferimento della "Conoscenza" da un modello di grandi dimensioni, verso un modello più piccolo;
3. **Metodologia Proposta**: combinazione della tecnica di Knowledge Distillation con l’iper-parametro `width-multiplier α` per la derivazione del modello proposto.

## Pruning
<img src="https://github.com/flavioforenza/thesis_latex/blob/main/images/pruning%20no%20name.png">


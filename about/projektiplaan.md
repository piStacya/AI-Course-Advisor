# Projektiplaan

<br>
<br>


## 1. Äritegevuse mõistmine
*Fookus: mis on probleem ja milline on hea tulemus?*


### 1.1 Kasutaja kirjeldus ja eesmärgid
Kellel on probleem ja miks see lahendamist vajab? Mis on lahenduse oodatud kasu? Milline on hetkel eksisteeriv lahendus?

> Tartu ülikooli õppeinfo süsteemis on tuhandeid õppeaineid ning endale sobivate õppeainete leidmine võib olla üsna keeruline ning aeganõudev - ükshaaval ainete läbikäimine ei ole realistlik ning praegused väga konkreetsete otsingusõnadega filtrid ei ole piisavalt paindlikud. Lisaks pole võimalik näha eelmiste semestrite aineid. Kuna üks Tartu Ülikooli plussidest on just võimalus võtta väga erinevaid õppeaineid ning panna endale kokku just sulle sobiv kava, on eriti vaba- ja valikainete parem otsimine väga tore võimalus tudengite jaoks leida just endale sobivad õppeained. Kasu ongi see, et saame leida paremaid sobivusi ainete ning tudengite vahel ning teha protsessi lõbusamaks ning ehk ka just integreerida erinevate valdkondade kontakte.

### 1.2 Edukuse mõõdikud
Kuidas mõõdame rakenduse edukust? Mida peab rakendus teha suutma?

> Rakendus on edukas, kui see enam-vähem usaldusväärselt suudab leida vasteid vabatekstilistele päringutele. Rakendus peaks leidma õppeained ka siis, kui aine sisu sõna sõnalt otsingule ei vasta, aga semantiline vaste on olemas. Kui kasutaja vabasõnalises päringus on ka rangeid filtreid (näiteks semester, instituut vms) peaks rakendus suutma ka neid rakendada. Rakendus peab toimima mõistliku kiirusega, sisaldama uusimat versiooni õppeainete nimekirjast. Rakendus peaks andma asjakohaseid vasteid, mitte sisaldama ebasobivaid ainete pakkumisi, reastama need sobivuse järgi. Rakenduse edukust saab mõõta oma õppekava väliste ainete võtmise kasvu järgi. Saame ka koguda kasutaja tagasisidet. Rakenduse arendamise käigus saame selle edukust testida teststsenaariumitega.

### 1.3 Ressursid ja piirangud
Millised on ressursipiirangud (nt aeg, eelarve, tööjõud, arvutusvõimsus)? Millised on tehnilised ja juriidilised piirangud (GDPR, turvanõuded, platvorm)? Millised on piirangud tasuliste tehisintellekti mudelite kasutamisele?

> Rakendus võiks töötada avalikel ÕIS2 andmetel ning olla veebipõhine ning vabalt kättesaadav. Rakendus peaks kasutama kas vabavaralisi tehisintellekti mudeleid või kui rakenduse edukaks toimimiseks on vaja tasulisi mudeleid, siis tuleb kindlasti vaadata, et kasutamisel oleks piirang vastavalt ressursi olemasolule. Rakendus ei tohi anda kasutajale ebasobivaid ja õppeainete otsinguga mitteseotud vastuseid.

<br>
<br>


## 2. Andmete mõistmine
*Fookus: millised on meie andmed?*

### 2.1 Andmevajadus ja andmeallikad
Milliseid andmeid (ning kui palju) on lahenduse toimimiseks vaja? Kust andmed pärinevad ja kas on tagatud andmetele ligipääs?

> Vaja on infot kõikide antud hetkel Tartu Ülikoolis õpetatavate ainete kohta (vähemalt terve aasta, eelistatult viimased 2 aastat), mis sisaldaks detailset infot, näiteks õppekava/õppeastme kuuluvust, kas aine toimub kohapeal või veebis jne. Andmed saab alla laadida Tartu Ülikooli õppeinfosüsteemi API kaudu ning andmed on avalikult kättesaadavad.

### 2.2 Andmete kasutuspiirangud
Kas andmete kasutamine (sh ärilisel eesmärgil) on lubatud? Kas andmestik sisaldab tundlikku informatsiooni?

> Andmed on avalikult kättesaadavad. Kuna need sisaldavad isikuandmeid õppejõudude kohta, siis avaliku rakenduse puhul oleks siiski vaja küsida eetikakomitee luba. Kui luba ei küsi, saame isikuandmed andmestikust eemaldada.

### 2.3 Andmete kvaliteet ja maht
Millises formaadis andmeid hoiustatakse? Mis on andmete maht ja andmestiku suurus? Kas andmete kvaliteet on piisav (struktureeritus, puhtus, andmete kogus) või on vaja märkimisväärset eeltööd)?

> Andmed hoiustatakse csv formaadis. Toorandmete maht on 45.3 MB, 3031 rida, 223 veergu. Osad veerud on erinevates keeltes või duplikaatides (aine üldine kirjeldus ning konkreetse versiooni kirjeldus). Osad veerud on tekstilised või numbrilised, osad json formaadis. Eeltööd on vaja, kuid tundub, et mitte märkimisväärselt.

### 2.4 Andmete kirjeldamise vajadus
Milliseid samme on vaja teha, et kirjeldada olemasolevaid andmeid ja nende kvaliteeti.

> Vaja on analüüsida 223 veeru tähendused ning välja valida olulised veerud. Seejärel on vaja valida õige veerg info leidmiseks, puhastada json väljad, panna kokku vabatekstilised kirjeldavad tunnused keelemudelile või RAG süsteemile analüüsiks. Vaja on üle vaadata puuduvate tunnuste hulk ning otsustada, mida nendega ette võtta.

<br>
<br>


## 3. Andmete ettevalmistamine
Fookus: Toordokumentide viimine tehisintellekti jaoks sobivasse formaati.

### 3.1 Puhastamise strateegia
Milliseid samme on vaja teha andmete puhastamiseks ja standardiseerimiseks? Kui suur on ettevalmistusele kuluv aja- või rahaline ressurss?

> Andmed on vaja puhastada natukene sarnasel viisil nagu 2.4 andmete kirjelduses mainitud. Võimalik, et oleks vaja imputeerida puuduvaid andmeid või neid otsida mõnest teisest ÕIS2 APIst või järeldada muudest andmetest. 

### 3.2 Tehisintellektispetsiifiline ettevalmistus
Kuidas andmed tehisintellekti mudelile sobivaks tehakse (nt tükeldamine, vektoriseerimine, metaandmete lisamine)?

> Olenevalt erinevatest meetoditest saame anda tehisintellektile kirjelduse andmetest ning ligipääsu puhastatud andmetele, et neid vajadusel filtreerida jne. RAG süsteemi jaoks on vaja välja valida aineid kirjeldavad veerud ning teha iga aine jaoks üks kirjeldav tekst. Valitud andmed tuleb vektoresituse kujule viimise mudeliga teisendada vektoriteks. Selle abil saab RAG süsteem semantiliselt otsingu järgi valida otsingule vastavad ained.

<br>
<br>

## 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> Vektoresituse mudel, (reranker mudel), LLM. Komponente rakendatakse üksteise järel.

### 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> LLMi kasutame API kaudu. Vektoresituse mudelit kasutame lokaalselt.

### 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> Vigade analüüs. Testjuhtude koostamine ja nende põhjal rakenduse headuse hindamine.
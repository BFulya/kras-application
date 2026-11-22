# KRAS-Quantum

**Kuantum-klasik hibrit üretici model ile KRAS inhibitörü tasarımı**

Bu depo, aşağıdaki makalenin açık kaynak bir uygulamasıdır:

> Vakili M.G., Gorgulla C., Snider J., Nigam A.K. ve ark.
> **"Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors"**
> *Nature Biotechnology* **43**, 1954–1959 (2025).
> [doi:10.1038/s41587-024-02526-3](https://doi.org/10.1038/s41587-024-02526-3)

Makale, 16 qubit'lik bir IBM kuantum işlemcisinde çalışan bir **Quantum Circuit
Born Machine (QCBM)** ile klasik bir **LSTM**'i birleştirerek KRAS proteinini
hedefleyen moleküller tasarlar; 15 bileşik sentezlenmiş ve ikisi (ISM061-018-2,
ISM061-022) deneysel olarak umut vaat eden sonuçlar vermiştir.

> ⚠️ **Bu depo bir araştırma yeniden-uygulamasıdır.** Tıbbi veya klinik bir
> amaçla kullanılamaz. Ürettiği moleküller hesaplamalı adaylardır; hiçbiri
> deneysel olarak doğrulanmamıştır.

---

## İçindekiler

- [Ne yapıyor?](#ne-yapıyor)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Neden Docker?](#neden-docker)
- [Hattın yapısı](#hattın-yapısı)
- [Kuantum bileşen: QCBM](#kuantum-bileşen-qcbm)
- [Klasik bileşen: LSTM](#klasik-bileşen-lstm)
- [Skorlama ve ödül](#skorlama-ve-ödül)
- [Doğrulamalar](#doğrulamalar)
- [Makaleden sapmalar ve yorumlar](#makaleden-sapmalar-ve-yorumlar)
- [Proje yapısı](#proje-yapısı)
- [Testler](#testler)
- [Kaynaklar](#kaynaklar)

---

## Ne yapıyor?

Makalenin üç aşamalı iş akışı (Makale, Şekil 1) baştan sona uygulanmıştır:

| Aşama | Makale | Bu depo |
|---|---|---|
| **1. Eğitim verisi** | 650 inhibitör + 850K STONED analoğu + 250K VirtualFlow hiti ≈ 1,1M | PubChem'den 19 gerçek inhibitör + ölçeklenebilir STONED analogları |
| **2. Üretim** | QCBM (16 qubit, IBM Guadalupe) + LSTM + Chemistry42 ödülü | QCBM (Qiskit, Aer **veya gerçek IBM donanımı**) + LSTM + açık kaynak ödül |
| **3. Seçim** | 3×1M bileşik → 15 sentez adayı | Aynı huni, aynı eşikler, aynı 3/8/4 kota dağılımı |

Temel fikir: **QCBM molekülleri doğrudan üretmez.** LSTM'i koşullandıran
*prior* örnekleri üretir ve geri besleme döngüsünde "iyi moleküle yol açan
prior'ları" öğrenir:

```
prior bitstring → LSTM → molekül → ödül R(x)
                                       │
QCBM ← P(x)=softmax(R(x)) hedef ←──────┘
```

Bu sayede 16 qubit'lik küçük bir devre, çok daha büyük bir kimyasal uzayı
yönlendirebilir — makalenin "qubit limitations" sorununa getirdiği çözüm budur.

---

## Hızlı başlangıç

```bash
git clone https://github.com/BFulya/kras-app.git
cd kras-app

# 1) Ortamı kur (qiskit + rdkit + torch + AutoDock Vina)
docker compose build

# 2) Çekirdek inhibitör yapılarını PubChem'den indir
docker compose run --rm kras python scripts/00_fetch_seed_library.py

# 3) Docking reseptörünü hazırla (PDB 7EW9, KRAS-G12D + TH-Z816)
docker compose run --rm kras python scripts/01_prepare_receptor.py

# 4) Eğitim setini oluştur (STONED-SELFIES)
docker compose run --rm kras python scripts/03_build_training_set.py --n-analogs 20000

# 5) Kampanyayı çalıştır
docker compose run --rm kras python scripts/04_run_campaign.py \
      --arms classical quantum_sim --n-epochs 20 --n-samples 5000
```

Testler:

```bash
docker compose run --rm kras pytest tests/ -q
```

### Gerçek kuantum donanımında çalıştırma

Makale, QCBM'i IBM'in **Guadalupe** sisteminde (16 qubit, Falcon r4P)
çalıştırmıştır. O cihaz emekliye ayrıldı; kod günümüz IBM cihazlarını aynı
arayüzle kullanır:

```bash
pip install qiskit-ibm-runtime
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \
           QiskitRuntimeService.save_account(channel='ibm_quantum', token='TOKEN')"

docker compose run --rm kras python scripts/04_run_campaign.py --arms quantum_hw
```

> Her epoch donanım kuyruğuna girer; makale ölçeğinde bir koşu saatler sürebilir.

---

## Neden Docker?

Windows'ta **Smart App Control**, pip ile gelen imzasız native eklentileri
bloklar — `qiskit._accelerate` ve `rdkit.rdBase` yüklenemez:

```
ImportError: DLL load failed while importing _accelerate:
An Application Control policy has blocked this file.
```

Linux konteyneri bu politikadan etkilenmez. Docker, Windows güvenlik ayarlarına
(geri alınamaz şekilde) dokunmadan tam çalışır bir ortam sağlar. Linux/macOS'ta
doğrudan `pip install -r requirements.txt` de yeterlidir.

---

## Hattın yapısı

### 1. Eğitim verisi üretimi (Makale, Şekil 1a)

**Çekirdek kütüphane.** Makaledeki 650 molekül, Insilico Medicine'in iç
derlemesidir ve yayınla dağıtılmamıştır. Bu depo o seti **uydurmaz**: literatürde
adı geçen KRAS inhibitörlerinin yapılarını PubChem REST API'sinden isimle çeker
([`scripts/00_fetch_seed_library.py`](scripts/00_fetch_seed_library.py)).
Depoda elle yazılmış tek bir SMILES yoktur.

İndirilen 19 bileşik arasında makalenin kendi referansları da bulunur:
**MRTX1133** (makale kaynak 21, G12D seçicilik referansı) ve **TH-Z816**
(Extended Data Şekil 4'ün farmakofor referansı, PDB 7EW9).

**STONED–SELFIES.** Makalenin hit genişletme adımı
([`stoned.py`](src/kras_quantum/data/stoned.py)): SMILES randomizasyonu →
SELFIES → nokta mutasyonları (silme, değiştirme, ekleme) → SMILES. SELFIES'in
sağlamlık garantisi sayesinde mutasyonlar geçersiz molekül üretemez.

### 2. Üretim (Makale, Şekil 1b, Algorithm 1)

Üç kol karşılaştırılır (Makale, Şekil 1d):

| Kol | Prior | Makalede seçilen |
|---|---|---|
| `classical` | yok (vanilla LSTM) | 3 bileşik |
| `quantum_sim` | QCBM, simülatörde | 4 bileşik |
| `quantum_hw` | QCBM, gerçek donanımda | 8 bileşik |

### 3. Seçim (Makale, Şekil 1d)

```
geçerlilik → filtreler → farmakofor → PLI → yapısal yenilik
          → sentetik karmaşıklık → kümeleme → 15 aday
```

---

## Kuantum bileşen: QCBM

[`src/kras_quantum/quantum/`](src/kras_quantum/quantum/)

- **Born kuralı:** `p(x) = |⟨x|ψ(θ)⟩|²`
- **Ansatz:** Extended Data Şekil 5 devresi —
  `[Rx(n)][Rz(n)][Rxx(n−1), lineer]` + kapanış `[Rz(n)][Rx(n)]`
- **Hedef dağılım:** `P(x) = softmax(R(x))`
- **Kayıp:** exact NLL — `NLL(θ) = −Σ_{x∈D} log p_θ(x)`
- **Optimize edici:** COBYLA (makalenin seçimi)
- **Arka uçlar:** `AerBackend` (SIM) ve `IBMBackend` (HW), aynı arayüz arkasında

Ansatz, Şekil 5'teki devreyi birebir izler: 4 qubit için tam **19 parametre**
üretir ve şemadaki `theta_0…theta_18` ile örtüşür (testle sabitlenmiştir).

---

## Klasik bileşen: LSTM

[`src/kras_quantum/classical/`](src/kras_quantum/classical/)

Kuantum prior'u LSTM girişine katılır (Makale, Denklem 9) — makalenin
belirttiği **iki yöntem de** uygulanmıştır:

```
concat :  x'_t = X(i) ++ x_t     (giriş boyutu büyür)
add    :  x'_t = X(i) +  x_t     (prior, öğrenilebilir doğrusal katmanla
                                  gömme uzayına taşınır)
```

`prior_dim=0` verilirse model makalenin **vanilla LSTM** kontrol koluna döner —
kuantum prior'un katkısını ölçmek için gereken karşılaştırma budur.

Eğitim: Adam + NLL (Denklem 7–8), BPTT, dropout.

---

## Skorlama ve ödül

**Chemistry42 tescillidir.** Bu depo onu yeniden uygulamaz; aynı işlevleri
açık araçlarla kuran bağımsız bir vekil sunar
([`src/kras_quantum/scoring/`](src/kras_quantum/scoring/)):

| Makale bileşeni | Bu depodaki karşılığı |
|---|---|
| Yapısal + med-chem filtreleri | RDKit tanımlayıcıları + SMARTS motif taraması |
| PLI skoru (QuickVina 2 / Chemistry42) | **Gerçek AutoDock Vina docking** + hızlı vekil |
| 3B farmakofor (PDB 7EW9) | Özellik eşleşmesi + Gobbi farmakofor parmak izi |
| ReRSA sentetik erişilebilirlik | Ertl–Schuffenhauer SA Score (aynı 1–10 ölçeği) |

Makalenin **iki aşamalı ödülü** de uygulanmıştır: epoch 20'ye kadar ucuz
"local filter", sonrasında tam bileşik ödül.

Sentez adaylığı eşikleri doğrudan makaleden alınmıştır:
toplam ödül > 0.7 · PLI < −8 kcal/mol · farmakofor > 0.7 · SA < 5.

---

## Doğrulamalar

Bu depodaki sayısal seçimler tahminle değil, **ölçümle** yapılmıştır.

### Docking kurulumu — redocking kontrolü

Ko-kristal ligand kendi cebine geri yerleştirildi (altın standart kontrol):

| Molekül | Vina (kcal/mol) | Beklenen |
|---|---|---|
| **TH-Z816** (7EW9 ko-kristal ligandı) | **−11.07** | en iyi ✔ |
| MRTX1133 (G12D inhibitörü) | −9.06 | güçlü ✔ |
| sotorasib (G12C ilacı) | −7.44 | G12D yapısında daha zayıf ✔ |
| benzen / etanol (negatif kontrol) | −4.64 / −2.34 | en kötü ✔ |

### Filtre eşikleri gerçek ilaçlara kalibre edildi

İlk uygulamada halka sayısı eşiği 7'ydi ve **MRTX1133 eleniyordu** (kaynaşmış
halka sistemi 8 SSSR halkası verir). Eşik 9'a çekildi. Çekirdek kütüphanedeki
19 gerçek inhibitörün 16'sı katı filtreyi geçiyor; elenen 3'ten 2'si hidrazin
taşıyan eski araç bileşikleri (Kobe0065/2602) — elenmeleri doğru.

### Farmakofor skoru kalibre edildi

İlk uygulamada özellik-varlığı ağırlığı 0.5'ti ve gerçek inhibitörlerin
yalnızca **3/19'u** makalenin 0.7 eşiğini geçiyordu — yani makalenin kendi
referans ilaçları "farmakofor uyumsuz" sayılıyordu. Ağırlık 0.75'e çekildi,
15/19 bileşik eşiği geçiyor. Bir testle sabitlendi.

### Vekil PLI skoru — elle yazılmış formülden öğrenilen modele

Eğitim döngüsü epoch başına yüzlerce molekül ödüllendirir; gerçek docking
molekül başına ~10–18 saniye sürdüğü için bu ölçekte kullanılamaz. Dolayısıyla
hızlı bir vekil skorlayıcı zorunludur — ama **doğru olduğu gösterilmiş** bir
vekil.

İlk uygulama ligand verimliliği ve fizikokimyasal terimlerden elle kurulmuştu.
Gerçek docking'e karşı ölçüldüğünde çöktü:

| Kalibrasyon seti | Spearman | Yorum |
|---|---|---|
| 34 molekül | **−0.396** | **ters korele** — ödül sinyali modeli yanlış yöne iterdi |
| 65 molekül | **+0.032** | korelasyon yok |

Teşhis: formül molekül boyutunu ödüllendiriyordu, oysa dar Switch-II cebinde
büyük moleküller daha *kötü* skor alıyor. Sezgisel bir formül bu cebin
davranışını yakalayamıyor.

Vekil bu nedenle artık **gerçek docking verisinden öğrenilir**
([`scripts/05_fit_surrogate.py`](scripts/05_fit_surrogate.py)): standartlaştırılmış
tanımlayıcılar üzerinde ridge regresyonu, 5-kat çapraz doğrulama ile seçilmiş
düzenlileştirme.

```
CV Spearman : +0.454   (görülmemiş moleküllerde sıralama)
CV MAE      : 1.14 kcal/mol
Eğitim verisi: 65 molekül, gerçek Vina skorları
```

En güçlü katsayılar, teşhisi bağımsız olarak doğruluyor:
`aromatic_fraction` −0.945 (aromatiklik iyi), `heavy_atoms` **+0.263**
(büyüklük *kötü* — elle yazılmış formülün tam tersi).

**Uygulanabilirlik alanı.** Model yalnızca eğitim verisine benzeyen
moleküllerde geçerlidir. Örneğin benzen (6 ağır atom, aromatik fraksiyon 1.0)
model tarafından −9.07 ile ödüllendiriliyordu, oysa gerçek docking −4.64
veriyor. Bu yüzden ağır atom sayısı eğitim aralığının (±%25) dışında kalan
moleküller skorlanmayı reddeder ve ödülde etkisiz kalır.

> Vekil **sıralama** içindir, enerji hesabı değildir; ve yalnızca bu reseptöre
> (PDB 7EW9, Switch-II cebi) özgüdür. **Nihai seçim daima `--pli vina` ile
> gerçek docking üzerinden yapılmalıdır.**

---

## Makaleden sapmalar ve yorumlar

Bilimsel dürüstlük açısından, makaleyle birebir örtüşmeyen her nokta:

### 1. Veri ölçeği

Makale ~1,1M molekülle eğitir. Bu depodaki çekirdek set 19 bileşiktir ve
STONED analog sayısı ölçeklenebilir. **VirtualFlow 2.0 aşaması atlanmıştır**:
100 milyon molekülün Enamine REAL taraması bulut ölçeği bir iştir (makale AWS
kullanmıştır) ve tek makinede yeniden üretilemez. Sonuçların makaleyle sayısal
olarak örtüşmesi **beklenmemelidir**.

### 2. QCBM parametre sayısı — makale içi tutarsızlık

Makale metni *"16 qubits and four layers and 96 parameters"* der. Ancak
Extended Data Şekil 5'teki ansatz (4 qubit / `theta_0…theta_18` = 19 parametre)
aynı topolojiyle 16 qubit ve 4 katmana ölçeklendiğinde **96 vermez**. 96 sayısı
ancak parametresiz entangler kullanan bir varyantla elde edilir (2 × 16 × 3 = 96).

Bu depo her iki okumayı da sunar ve hangisinin kullanıldığını her koşuda
raporlar:

- `mode="fig5"` *(varsayılan)* — Şekil 5 topolojisi, parametreli `Rxx`
- `mode="hardware_efficient"` — parametresiz CNOT; 16 qubit/3 katmanda tam 96 parametre

### 3. Tescilli bileşenler

| Makalede | Bu depoda |
|---|---|
| Chemistry42 (Insilico) | Açık kaynak vekil skorlayıcı |
| ReRSA modeli (patent WO 2021/229454) | Ertl–Schuffenhauer SA Score |
| Schrödinger protein hazırlama | OpenBabel (hidrojen ekleme + pH 7.4 protonasyon) |
| QuickVina 2 | AutoDock Vina (aynı skorlama fonksiyonu ailesi) |
| IBM Guadalupe (emekli) | Güncel IBM cihazları, aynı arayüz |

### 4. Makalenin belirtmediği, bu depoda seçilen değerler

- **Ödül bileşen ağırlıkları** — makale bileşenleri sayar ama ağırlık vermez.
  Varsayılan: filtre 0.30, PLI 0.30, farmakofor 0.20, sentez 0.20.
- **STONED mutasyon sayısı dağılımı** — makale yalnızca üst sınır verir
  ("up to 500 times"). Düz dağılım işe yaramaz: ortalama ~250 mutasyon çekirdeği
  yok eder ve Şekil 1a'nın yakın kabuğu (>0.6) hiç dolmaz (2400 denemede 0 analog).
  Geometrik dağılıma geçildi — çoğu analog yakın, azı uzak; Şekil 1a'daki
  genişleyen çeşitlilik konisiyle tutarlı.
- **Gradyan kırpma** — makale belirtmez; BPTT ile eğitimde standart uygulama.

### 5. Doğrulanmayanlar

Makalenin deneysel kısmı (SPR, MaMTH-DS hücre tahlilleri, NMR) bu depo
kapsamında **değildir**. Üretilen moleküller hesaplamalı adaylardır.
Makaledeki Tartarus kıyaslaması (REINVENT, JANUS, MoFlow vb. ile karşılaştırma)
da uygulanmamıştır; bu depo kuantum/klasik prior karşılaştırmasını kendi
içinde yapar ([`benchmark/`](src/kras_quantum/benchmark/)).

---

## Proje yapısı

```
src/kras_quantum/
├── config.py            Makaledeki tüm sayısal parametreler, kaynak referanslı
├── data/
│   ├── seed_library.py  PubChem'den çekirdek inhibitörler
│   ├── stoned.py        STONED–SELFIES hit genişletme
│   └── tokenizer.py     SELFIES token sözlüğü
├── quantum/
│   ├── ansatz.py        Extended Data Şekil 5 devresi
│   ├── backend.py       Aer (SIM) / IBM (HW)
│   └── qcbm.py          Born kuralı, exact NLL, COBYLA
├── classical/
│   ├── lstm.py          Prior-koşullu LSTM (Denklem 9–15)
│   └── trainer.py       Adam + BPTT
├── scoring/
│   ├── filters.py       Yapısal / med-chem / fizikokimyasal
│   ├── pli.py           Gerçek Vina docking + hızlı vekil
│   ├── pharmacophore.py Extended Data Şekil 4 modeli
│   ├── synthetic.py     SA Score (ReRSA vekili)
│   └── reward.py        R(x) = birleşik ödül
├── pipeline/
│   ├── hybrid.py        Algorithm 1 (kuantum-klasik döngü)
│   └── selection.py     Şekil 1d seçim hunisi
└── benchmark/
    └── priors.py        Extended Data Şekil 1 kıyaslamaları

scripts/
├── 00_fetch_seed_library.py   PubChem'den yapılar
├── 01_prepare_receptor.py     PDB 7EW9 → PDBQT + docking kutusu
├── 02_calibrate_surrogate.py  Vekil PLI'ı gerçek docking'e kalibre et
├── 03_build_training_set.py   STONED–SELFIES eğitim seti
└── 04_run_campaign.py         Uçtan uca kampanya
```

---

## Testler

```bash
docker compose run --rm kras pytest tests/ -q
```

Testler yalnızca kodun çalıştığını değil, **bilimsel tutarlılığı** da denetler:

- Born kuralı dağılımı normalize mi? COBYLA exact NLL'i düşürüyor mu?
- Ansatz, Extended Data Şekil 5'teki parametre sayısını veriyor mu?
- Farklı prior'lar gerçekten farklı çıktı üretiyor mu?
- Üretilen her SELFIES geçerli moleküle çözülüyor mu?
- **Gerçek KRAS inhibitörleri filtreleri ve farmakofor eşiğini geçiyor mu?**
- Seçim hunisi daralıyor mu, kota dağılımı 3/8/4'e uyuyor mu?

---

## Kaynaklar

- Makale: [doi:10.1038/s41587-024-02526-3](https://doi.org/10.1038/s41587-024-02526-3)
- Makalenin veri/kod deposu: [Zenodo 11137638](https://doi.org/10.5281/zenodo.11137638)
- STONED–SELFIES: [aspuru-guzik-group/stoned-selfies](https://github.com/aspuru-guzik-group/stoned-selfies)
- SELFIES: Krenn ve ark., *Mach. Learn. Sci. Technol.* **1**, 045024 (2020)
- QCBM genelleme: Hibat-Allah ve ark., *Commun. Phys.* **7**, 68 (2024)
- AutoDock Vina: Eberhardt ve ark., *J. Chem. Inf. Model.* **61**, 3891 (2021)
- PDB 7EW9 (KRAS-G12D + TH-Z816), PDB 5US4 (makalenin tarama yapısı)

## Lisans

Kod: MIT. Makalenin kendisi CC BY-NC-ND 4.0 ile lisanslıdır; bu depo makalenin
metnini veya şekillerini içermez, yalnızca tarif edilen yöntemi uygular.

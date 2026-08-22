# Yatırım — TÜFE+%5 Portföy Yönetim Asistanı

Türkiye piyasaları (BIST, Altın, Döviz, Para Piyasası Araçları) için günlük
makro ve teknik verileri girdi alan, "TÜFE + %5" reel getiri hedefine yönelik
varlık tahsisi ve alım-satım önerileri üreten kural tabanlı bir karar motoru.

Asistanın tam rol/misyon tanımı `docs/SYSTEM_PROMPT.md` içinde saklanır ve
`yatirim/strategy.py` bu tanımdaki kuralları koda döker.

## Kurulum

Sadece Python 3.10+ standart kütüphanesi kullanılır, ek bağımlılık yoktur.
Testler için `pytest` gerekir:

```bash
pip install -r requirements-dev.txt
```

## Kullanım

Günlük girdileri bir JSON dosyasına yazıp CLI ile çalıştırabilirsiniz:

```bash
python -m yatirim.cli examples/sample_input.json
```

Çıktı, spesifikasyondaki JSON formatına uygun tek bir JSON nesnesi olarak
stdout'a yazılır.

## ML tabanlı sinyal modülü (`yatirim/ml`)

Kural tabanlı motora ek olarak, geçmiş fiyat verisinden öğrenen bir
sınıflandırıcı ile hisse/altın/döviz bazında "yükseliş bekleniyor" sinyali
üreten bir pipeline bulunur:

1. **Veri**: Bu ortamdan gerçek piyasa verisine (Yahoo Finance, BIST, TCMB)
   ağ erişimi yok; `yatirim/ml/synthetic_data.py` rejim geçişli
   (bull/bear/sideways) ve volatilite kümelenmesi olan 3 yıllık **sentetik**
   günlük fiyat serileri üretir (8 BIST30 hissesi + gram altın + USD/EUR).
   Gerçek veri elde edildiğinde aynı şemaya (`tarih, sembol, varlik_tipi,
   acilis, yuksek, dusuk, kapanis, hacim`) dönüştürülüp doğrudan bu
   pipeline'a verilebilir.
2. **Özellikler** (`features.py`): getiri, MA20/50/200 uzaklığı, RSI14,
   MACD, 20 günlük volatilite, hacim z-skoru. Etiket: sonraki 10 işlem
   günündeki getirinin pozitif olup olmadığı.
3. **Model** (`model.py`): Gradient Boosting sınıflandırıcı, **zaman bazlı**
   (walk-forward) train/test ayrımıyla eğitilir — test dönemi her zaman
   eğitim döneminden sonra gelir, gelecek sızıntısı yoktur.
4. **Durum analizi demosu** (`backtest.py`): Modelin test/backtest
   döneminde ürettiği geçmiş "AL" sinyallerini, o sinyalden sonraki 10
   günde gerçekleşen getiriyle birlikte raporlar (isabet oranı, ortalama
   getiri, kümülatif getiri) — yani önerilerden **önce** "bu model geçmişte
   nasıl performans gösterdi" gösterimi. Sonuçlar hem **brüt** hem
   **işlem maliyeti düşülmüş (net)** olarak raporlanır (bkz. aşağıdaki
   "İşlem maliyeti modeli").
5. **Canlı sinyal → öneri** (`integration.py`): En güncel tarihteki model
   çıktısını, backtest'in **net** (maliyet sonrası) performansına atıfla
   gerekçelendirilmiş `Recommendation` nesnelerine çevirir.

### İşlem maliyeti modeli

`run_backtest`, her sinyali brüt getirinin yanında round-trip (alış +
satış) işlem maliyeti düşülmüş **net getiri** ile de raporlar. Varsayılan
tek yön maliyet varsayımları (`DEFAULT_COST_BY_ASSET_TYPE`,
`yatirim/ml/backtest.py`):

| Varlık tipi | Tek yön maliyet | Round-trip |
|---|---|---|
| BORSA | %0.15 (komisyon + BSMV + borsa payı) | %0.30 |
| ALTIN | %0.20 (alış-satış makası) | %0.40 |
| DOVIZ | %0.08 (alış-satış makası) | %0.16 |

Bunlar muhafazakar birer tahmindir; gerçek aracı kurumunuzun komisyon
tarifesine göre `run_backtest(result, cost_by_asset_type={...})` ile
ezilebilir. `BacktestSummary.to_dict()` hem `*_brut_*` hem `*_net_*`
alanlarını döner — kararı **her zaman net rakamlara göre** verin, brüt
rakamlar maliyetleri gizler ve yanıltıcı olabilir.

### Canlı izleme — şu an NASIL çalışıyor, ne DEĞİL

Bu pipeline **sürekli canlı bir servis değildir**; tek seferlik/batch bir
akıştır: `--data` ile verilen CSV'nin **en son satırındaki** tarihe göre
sinyal üretir (`latest_signals`). CSV ne zaman toplandıysa sinyal o ana
aittir; kendi kendine güncellenmez.

Bu depo, güvenlik amaçlı kısıtlı bir ağ ortamında geliştirildiği/çalıştığı
için piyasa verisine gerçek zamanlı erişimi yoktur ve olamaz. Gerçek
"canlı izleme" için elinizdeki tek yol, verinin toplandığı makinede
(`scripts/collect_market_data.py`'yi çalıştırdığınız makine) periyodik bir
görev kurmaktır — örn. Windows Görev Zamanlayıcı ile her gün piyasa
kapanışından sonra:

```bash
python scripts/collect_market_data.py --years 3 --out gercek_fiyatlar.csv
python -m yatirim.ml.run_pipeline --data gercek_fiyatlar.csv > guncel_rapor.txt
```

çalıştırılıp `guncel_rapor.txt` / `artifacts/guncel_sinyaller.csv`
her gün üzerine yazılabilir.

Her gün otomatik, Telegram bildirimli bir rapor için
`scripts/daily_signal_report.py` + `scripts/run_daily_signal_report.bat`
kullanın — bkz. "Günlük sinyal raporu + Telegram bildirimi" bölümü
aşağıda.

Çalıştırmak için:

```bash
pip install -r requirements-ml.txt
python -m yatirim.ml.run_pipeline
```

Bu komut sırasıyla veri üretir, modeli eğitir, geçmiş dönem durum analizini
konsola basar ve ardından güncel sinyalleri/önerileri gösterir. Ara
çıktılar `artifacts/` klasörüne (fiyatlar, eğitilmiş model, backtest
sinyal tablosu) kaydedilir.

**Önemli sınırlama**: Varsayılan çalıştırmada sonuçlar sentetik veri
üzerinde üretildiği için gerçek piyasa tahmini değildir; amacı mimariyi
(veri → özellik → eğitim → backtest → canlı sinyal) uçtan uca çalışır ve
test edilebilir halde göstermektir.

### Gerçek veriyle çalıştırma

Bu depo, güvenlik amaçlı kısıtlı bir ağ ortamında geliştirildiği için
buradan Yahoo Finance/TradingView gibi kaynaklara erişilemiyor. Gerçek
BIST 100 + gram altın + USD/EUR verisiyle çalıştırmak için:

1. **İnternete açık kendi makinenizde** veri toplama script'ini çalıştırın:

   ```bash
   pip install yfinance pandas
   python scripts/collect_market_data.py --years 3 --out artifacts/gercek_fiyatlar.csv
   ```

   Bu script, `scripts/bist100_symbols.txt` içindeki sembol listesi için
   Yahoo Finance'ten (`SEMBOL.IS`) 3 yıllık günlük OHLCV verisi, `TRY=X`
   ile USD/TRY, `EURTRY=X` ile EUR/TRY ve `GC=F` (ons altın, USD) ×
   USD/TRY'den türetilmiş gram altın (TRY) serisini indirir; hepsini
   `yatirim/ml/synthetic_data.py`'nin ürettiğiyle **birebir aynı şemada**
   tek bir CSV'ye yazar.

   > `scripts/bist100_symbols.txt`'teki liste yaklaşıktır — BIST 100
   > bileşimi üç ayda bir değişir, gerçek kararlar için Borsa
   > İstanbul'un güncel endeks listesiyle karşılaştırıp dosyayı
   > güncelleyin.

2. Çıkan CSV'yi bu depoya (veya bu depoya erişimi olan bir ortama)
   taşıyıp pipeline'ı gerçek veriyle çalıştırın:

   ```bash
   pip install -r requirements-ml.txt
   python -m yatirim.ml.run_pipeline --data artifacts/gercek_fiyatlar.csv
   ```

Kod tarafında başka hiçbir değişiklik gerekmez; `--data` verilmezse
pipeline otomatik olarak sentetik veriye döner.

### Günlük sinyal raporu + Telegram bildirimi

`scripts/daily_signal_report.py`, hafta içi her gün **saat 18:15'te**
(BIST kapanışı 18:10'dan sonra) bir kez çalışır: izleme listesindeki her
sembolün **günün kesinleşmiş kapanışını** çeker, eğitilmiş modelden
geçirir ve tamamı için **AL / SAT / TUT** sinyalini tek bir Telegram
mesajında raporlar (AL: `P(yükseliş) >= 0.55`, SAT: `P <= 0.45`, arası TUT
— eşikler `--buy-threshold`/`--sell-threshold` ile değiştirilebilir).

**Dürüst sınırlama**: Model **günlük** bar'lar üzerinde, 10 işlem günü
ileriye dönük yön tahmini için eğitildi; bu rapor o tahminin **o günkü**
anlık görüntüsüdür, kesin bir işlem emri değildir. "İşlem maliyeti modeli"
bölümündeki net rakamlar (isabet oranı ~%50, yani pratikte yazı-turaya
yakın) burada da geçerli — bildirim almak, kârlı bir sinyal garantisi
değildir. Ayrıca Yahoo Finance'in günlük veriyi ne zaman güncellediği
garanti değildir; bir sembolün kapanışı henüz güncellenmemişse rapor bunu
"⚠️ Veri güncel değil" diye açıkça belirtir.

Kurulum:

1. Önce gerçek veriyle modeli eğitin (yukarıdaki adımlar) — `artifacts/model.joblib`
   oluşmuş olmalı.
2. Telegram bot token'ınızı ve chat ID'nizi girin:
   ```bash
   copy scripts\.env.example scripts\.env
   notepad scripts\.env
   ```
   (`scripts/.env` asla git'e commit edilmez — token'lar yalnızca kendi
   makinenizde kalır.) Mevcut bir botunuzu kullanmak istiyorsanız o botu
   yönettiğiniz Telegram sohbetinden `@BotFather` → `/mybots` → botu seçin
   → **API Token** ile token'ı alabilirsiniz; chat ID'nizi öğrenmek için
   `@userinfobot`'a mesaj atmanız yeterli.
3. Bağımlılıkları kurun ve başlatın:
   ```bash
   pip install -r requirements-ml.txt
   scripts\run_daily_signal_report.bat
   ```
   Pencereyi açık bırakın — script bir sonraki 18:15'i (hafta sonuysa
   pazartesiye kayarak) hesaplayıp o ana kadar bekler, sonra raporu
   gönderip bir sonrakini beklemeye devam eder. Arka planda/oturum
   kapansa da çalışsın isterseniz Windows Görev Zamanlayıcı'da bu `.bat`'ı
   "Oturum açılışında" tetikleyicisiyle başlatın. Durdurmak için pencerede
   Ctrl+C.

İzleme listesi varsayılan olarak `scripts/watchlist_live.txt`'teki ~20
likit BIST hissesidir; genişletmek için
`python scripts\daily_signal_report.py --symbols-file scripts\bist100_symbols.txt`
kullanabilir ya da `watchlist_live.txt`'i düzenleyebilirsiniz. Tek seferlik
test için (18:15'i beklemeden hemen çalıştırır): `python scripts\daily_signal_report.py --once`.

## Proje yapısı

- `yatirim/models.py` — girdi/çıktı veri modelleri (dataclass'lar)
- `yatirim/strategy.py` — TÜFE+%5 stratejisi: rejim tespiti, varlık tahsisi,
  stop-loss/drawdown kontrolü, işlem maliyeti eşiği
- `yatirim/advisor.py` — günlük girdilerden nihai JSON raporunu üreten motor
- `yatirim/cli.py` — komut satırı arayüzü
- `yatirim/ml/synthetic_data.py` — 3 yıllık sentetik fiyat verisi üretimi
- `yatirim/ml/features.py` — teknik gösterge/etiket üretimi
- `yatirim/ml/model.py` — walk-forward model eğitimi
- `yatirim/ml/backtest.py` — geçmiş dönem durum analizi (backtest) demosu
- `yatirim/ml/integration.py` — model sinyallerini `Recommendation`'a çevirir
- `yatirim/ml/data_loader.py` — gerçek fiyat CSV'sini yükler/doğrular
- `yatirim/ml/run_pipeline.py` — uçtan uca demo CLI (`--data` ile gerçek veri)
- `scripts/collect_market_data.py` — Yahoo Finance'ten gerçek veri toplama
  (internete açık makinede çalıştırılır, bu depodan değil)
- `scripts/bist100_symbols.txt` — BIST 100 sembol listesi (yaklaşık)
- `scripts/watchlist_live.txt` — günlük rapor için daha kısa/likit liste
- `scripts/daily_signal_report.py` — her gün 18:15'te AL/SAT/TUT + Telegram raporu
- `scripts/run_daily_signal_report.bat` — günlük rapor için Windows başlatıcı
- `scripts/.env.example` — Telegram kimlik bilgileri şablonu
- `examples/sample_input.json` — örnek günlük girdi
- `tests/` — strateji ve ML pipeline kurallarını doğrulayan birim testler

## Strateji kuralları (özet)

1. **Hedef getiri**: Yıllık tahmini TÜFE + %5 net reel kazanç.
2. **Dinamik varlık dağılımı**: Yüksek enflasyon/belirsizlik döneminde
   Altın/Döviz/Likit ağırlıklı; büyüme/ralli döneminde BIST ağırlığı
   %40–%70 arası.
3. **Likit taban**: Portföyün en az %10–%20'si her zaman BPP/Repo gibi
   anlık likit araçlarda tutulur.
4. **Maksimum drawdown**: Günlük %2.5 kayıp limiti aşılırsa otomatik
   "Stop-Loss / Güvenli Limana Geçiş" senaryosu tetiklenir.
5. **İşlem maliyeti eşiği**: Beklenen getiri farkı en az %1.5 olmadıkça
   mevcut pozisyon korunur (TUT).
6. **Hisse evreni**: Sadece BIST 30 hisseleri önerilir, düşük hacimli/
   spekülatif hisseler önerilmez.

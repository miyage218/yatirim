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
   nasıl performans gösterdi" gösterimi.
5. **Canlı sinyal → öneri** (`integration.py`): En güncel tarihteki model
   çıktısını, backtest performansına atıfla gerekçelendirilmiş
   `Recommendation` nesnelerine çevirir.

Çalıştırmak için:

```bash
pip install -r requirements-ml.txt
python -m yatirim.ml.run_pipeline
```

Bu komut sırasıyla veri üretir, modeli eğitir, geçmiş dönem durum analizini
konsola basar ve ardından güncel sinyalleri/önerileri gösterir. Ara
çıktılar `artifacts/` klasörüne (fiyatlar, eğitilmiş model, backtest
sinyal tablosu) kaydedilir.

**Önemli sınırlama**: Sonuçlar sentetik veri üzerinde üretildiği için
gerçek piyasa tahmini değildir; pipeline'ın amacı mimariyi (veri → özellik
→ eğitim → backtest → canlı sinyal) uçtan uca çalışır ve test edilebilir
halde göstermektir. Gerçek veri bağlandığında aynı kod tabanı kullanılır.

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
- `yatirim/ml/run_pipeline.py` — uçtan uca demo CLI
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

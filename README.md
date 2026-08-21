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

## Proje yapısı

- `yatirim/models.py` — girdi/çıktı veri modelleri (dataclass'lar)
- `yatirim/strategy.py` — TÜFE+%5 stratejisi: rejim tespiti, varlık tahsisi,
  stop-loss/drawdown kontrolü, işlem maliyeti eşiği
- `yatirim/advisor.py` — günlük girdilerden nihai JSON raporunu üreten motor
- `yatirim/cli.py` — komut satırı arayüzü
- `examples/sample_input.json` — örnek günlük girdi
- `tests/` — strateji kurallarını doğrulayan birim testler

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

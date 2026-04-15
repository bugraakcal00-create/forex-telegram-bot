"""
news_impact.py
--------------
Türkçe ekonomik olay etki analiz veritabanı.
Anahtar kelime eşleştirme ile olayların kapsamlı analizini döndürür.
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Olay veritabanı
# ---------------------------------------------------------------------------
_EVENTS: list[dict] = [
    {
        "_keywords": ["nonfarm", "non-farm", "non farm", "nfp", "tarım dışı istihdam", "tarim disi"],
        "name": "Tarım Dışı İstihdam (NFP)",
        "kategori": "İstihdam",
        "frekans": "Aylık (her ayın ilk Cuması)",
        "aciklama": (
            "ABD ekonomisindeki tarım sektörü dışındaki toplam istihdam değişimini ölçer. "
            "Fed'in para politikası kararlarında birincil gösterge olarak kullanılır. "
            "Piyasanın en çok beklediği makroekonomik veridir."
        ),
        "yukari_etki": (
            "Beklentinin üzerinde gelirse USD güçlenir; XAU/USD sert düşer, EUR/USD ve GBP/USD geriler. "
            "Fed faiz artırımı beklentisi canlanır, tahvil faizleri yükselir. "
            "Risk iştahı artar, hisse endeksleri prim yapabilir."
        ),
        "asagi_etki": (
            "Beklentinin altında gelirse USD zayıflar; XAU/USD fırlar, EUR/USD ve GBP/USD yükselir. "
            "Fed'in faiz indirimi beklentisi öne çekilir. "
            "Risk iştahı düşer, güvenli liman varlıkları (altın, JPY) talep görür."
        ),
        "etkilenen": ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"],
        "tipik_hareket": "150-400 pip (XAU/USD), 50-150 pip (majörler)",
        "volatilite": "⚡⚡⚡⚡⚡ Çok Yüksek",
        "gecmis": (
            "Ocak 2024: 256K beklenti, 353K gerçekleşme → XAU/USD 45 dolar düştü. "
            "Ekim 2023: 170K beklenti, 336K gerçekleşme → EUR/USD 120 pip çöküş. "
            "Temmuz 2023: 200K beklenti, 187K gerçekleşme → XAU/USD 25 dolar yükseldi."
        ),
        "tavsiye": (
            "Açıklamadan 15 dakika önce ve sonra işlem yapmaktan kaçının. "
            "İlk hareketi bekleyin, sahte kırılımlara karşı dikkatli olun. "
            "Veri açıklandıktan 5-10 dakika sonra oluşan konsolidasyon kırılımlarını takip edin."
        ),
    },
    {
        "_keywords": ["cpi", "consumer price", "tüketici fiyat", "tuketici fiyat", "enflasyon", "inflation"],
        "name": "Tüketici Fiyat Endeksi (CPI / Enflasyon)",
        "kategori": "Enflasyon",
        "frekans": "Aylık",
        "aciklama": (
            "Tüketicilerin ödediği mal ve hizmet fiyatlarındaki değişimi ölçer. "
            "Fed başta olmak üzere tüm merkez bankalarının faiz kararlarında temel referans noktasıdır. "
            "Çekirdek CPI (enerji ve gıda hariç) daha önemli kabul edilir."
        ),
        "yukari_etki": (
            "Beklentinin üzerinde gelirse enflasyon yüksek demektir; merkez bankası sıkılaştırma beklentisi artar. "
            "USD güçlenir, XAU/USD kısa vadede düşebilir. "
            "Tahvil faizleri yükselir, büyüme hisselerine baskı gelir."
        ),
        "asagi_etki": (
            "Beklentinin altında gelirse enflasyon soğuyor demektir; merkez bankası gevşeme beklentisi artar. "
            "USD zayıflar, XAU/USD yükselir, risk iştahı canlanabilir. "
            "Faiz indirimi beklentisi öne çekilir."
        ),
        "etkilenen": ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"],
        "tipik_hareket": "80-250 pip (XAU/USD), 40-100 pip (majörler)",
        "volatilite": "⚡⚡⚡⚡⚡ Çok Yüksek",
        "gecmis": (
            "Haziran 2022: ABD CPI %9.1 (beklenti %8.8) → XAU/USD 30 dolar düştü, DXY 100 puan fırladı. "
            "Kasım 2023: CPI %3.2 (beklenti %3.3) → EUR/USD 150 pip yükseldi. "
            "Şubat 2024: Çekirdek CPI %3.9 (beklenti %3.7) → XAU/USD 40 dolar geriledi."
        ),
        "tavsiye": (
            "Açıklamadan önce pozisyon almayın. İlk 2-3 dakika çok sert ve yanıltıcı olabilir. "
            "Gerçek yön genellikle 5-10 dakika sonra belli olur. "
            "Stop-loss'larınızı geniş tutun veya açıklamayı geçirin."
        ),
    },
    {
        "_keywords": ["fomc", "federal reserve", "federal open market", "fed funds rate", "fed interest rate", "faiz kararı", "interest rate decision"],
        "name": "FOMC / Fed Faiz Kararı",
        "kategori": "Merkez Bankası Kararı",
        "frekans": "Yılda 8 kez (her 6-7 haftada bir)",
        "aciklama": (
            "ABD Merkez Bankası (Fed) Federal Açık Piyasa Komitesi'nin faiz oranı kararıdır. "
            "Kararın yanı sıra Fed Başkanı'nın basın toplantısı ve politika açıklamaları da büyük önem taşır. "
            "Küresel piyasaları en çok etkileyen para politikası kararıdır."
        ),
        "yukari_etki": (
            "Faiz artırımı veya şahin açıklama gelirse USD güçlü tepki verir. "
            "XAU/USD sert satılır (yüksek faiz altının fırsat maliyetini artırır). "
            "EUR/USD, GBP/USD geriler; USD/JPY yükselir."
        ),
        "asagi_etki": (
            "Faiz indirimi veya güvercin açıklama gelirse USD zayıflar. "
            "XAU/USD güçlü yükselir, EUR/USD ve GBP/USD toparlar. "
            "Risk iştahı artar, gelişen piyasa paraları değer kazanır."
        ),
        "etkilenen": ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD"],
        "tipik_hareket": "200-600 pip (XAU/USD), 80-200 pip (majörler)",
        "volatilite": "⚡⚡⚡⚡⚡ Çok Yüksek",
        "gecmis": (
            "Mart 2022: Fed 25 baz puan artırdı, şahin sinyal → XAU/USD 50 dolar düştü. "
            "Temmuz 2023: 25 baz puan artırım, duraklatma sinyali → XAU/USD 20 dolar yükseldi. "
            "Aralık 2023: Faiz sabit, 2024 için 3 indirim sinyali → XAU/USD 80 dolar fırladı."
        ),
        "tavsiye": (
            "Karar 3 aşamada değerlendirin: İlk karar açıklaması, yazılı politika metni, Powell basın toplantısı. "
            "En sert hareket genellikle basın toplantısında olur. "
            "Karardan 1 saat önce pozisyon kapatmayı düşünün."
        ),
    },
    {
        "_keywords": ["ism", "pmi", "purchasing managers", "manufacturing", "services pmi", "imalat", "hizmet sektörü", "markit"],
        "name": "ISM PMI (Üretim & Hizmet Sektörü Endeksi)",
        "kategori": "Öncü Gösterge",
        "frekans": "Aylık (ISM Üretim: her ayın 1. iş günü, ISM Hizmet: her ayın 3. iş günü)",
        "aciklama": (
            "Satın Alma Yöneticileri Endeksi, imalat ve hizmet sektörlerindeki ekonomik aktiviteyi ölçer. "
            "50'nin üzeri genişleme, altı daralma anlamına gelir. "
            "Öncü gösterge niteliğinde olduğu için piyasa tarafından yakından takip edilir."
        ),
        "yukari_etki": (
            "50'nin üzerinde ve beklentiden yüksek gelirse ekonomik aktivite güçlü demektir. "
            "USD değer kazanır, risk iştahı artar. "
            "XAU/USD baskı altına girebilir."
        ),
        "asagi_etki": (
            "50'nin altında veya beklentiden düşük gelirse ekonomik yavaşlama sinyali verir. "
            "USD zayıflar, güvenli liman varlıkları (altın, CHF, JPY) talep görür. "
            "Stagflasyon kaygıları öne çıkabilir."
        ),
        "etkilenen": ["USD/JPY", "EUR/USD", "GBP/USD", "XAU/USD", "AUD/USD"],
        "tipik_hareket": "30-80 pip (majörler), 15-40 dolar (XAU/USD)",
        "volatilite": "⚡⚡⚡ Orta-Yüksek",
        "gecmis": (
            "Kasım 2023: ISM Üretim 46.7 (beklenti 47.7) → USD/JPY 50 pip düştü. "
            "Mayıs 2023: ISM Hizmet 50.3 (beklenti 51.5) → EUR/USD 40 pip yükseldi. "
            "Ocak 2024: ISM Üretim 49.1 (beklenti 47.0) → XAU/USD 15 dolar düştü."
        ),
        "tavsiye": (
            "NFP veya FOMC kadar sert değil ancak pozisyonları tehlikeye atabilir. "
            "Haber öncesi stop-loss sıkılaştırın veya pozisyonu küçültün. "
            "50 eşiğinin üstünde/altında geçişlere özellikle dikkat edin."
        ),
    },
    {
        "_keywords": ["gdp", "gross domestic", "gsyh", "büyüme", "buyume", "economic growth"],
        "name": "GSYİH / GDP (Gayri Safi Yurt İçi Hasıla)",
        "kategori": "Büyüme",
        "frekans": "Üç ayda bir (çeyreklik); ön tahmin, revize ve nihai olmak üzere 3 aşamada yayımlanır",
        "aciklama": (
            "Bir ekonominin belirli bir dönemde ürettiği toplam mal ve hizmet değerini ölçer. "
            "Ekonomik büyümenin temel göstergesidir. "
            "Ön tahmin en çok hareket yaratırken nihai veri genellikle daha az etki yapar."
        ),
        "yukari_etki": (
            "Beklentinin üzerinde gelirse ekonomi güçlü büyüyor demektir. "
            "Para birimi değer kazanır, faiz artırımı beklentisi güçlenebilir. "
            "Risk iştahı artabilir."
        ),
        "asagi_etki": (
            "Beklentinin altında gelirse ekonomik yavaşlama kaygıları artar. "
            "Para birimi zayıflar, güvenli liman varlıkları talep görür. "
            "İki çeyreklik negatif büyüme teknik resesyon anlamına gelir."
        ),
        "etkilenen": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD", "AUD/USD"],
        "tipik_hareket": "30-100 pip (majörler), 20-50 dolar (XAU/USD)",
        "volatilite": "⚡⚡⚡ Orta-Yüksek",
        "gecmis": (
            "Q3 2023 ABD GDP %4.9 (beklenti %4.3) → USD/JPY 80 pip yükseldi. "
            "Q1 2023 ABD GDP %1.1 (beklenti %2.0) → XAU/USD 25 dolar fırladı. "
            "Q2 2022: -0.9% (negatif) → Resesyon tartışmaları, XAU/USD 30 dolar düştü."
        ),
        "tavsiye": (
            "GDP açıklamalarında ilk reaksiyon zaman zaman tersine döner. "
            "Kümülatif tablo ve revizyon rakamlarını da değerlendirin. "
            "Çeyreklik değil, yıllık bazda karşılaştırma yapın."
        ),
    },
    {
        "_keywords": ["retail sales", "perakende", "tüketici harcama", "consumer spending", "core retail"],
        "name": "Perakende Satışlar (Retail Sales)",
        "kategori": "Tüketim",
        "frekans": "Aylık",
        "aciklama": (
            "Perakende sektöründeki satışlardaki aylık değişimi ölçer ve tüketici harcamalarının göstergesidir. "
            "ABD ekonomisinin %70'i tüketime dayalı olduğu için çok önemlidir. "
            "Çekirdek perakende satışlar (otomobil hariç) daha istikrarlı bir sinyal verir."
        ),
        "yukari_etki": (
            "Güçlü gelen veri tüketici harcamalarının canlı olduğunu gösterir. "
            "USD değer kazanır, büyüme hikayesi güçlenir. "
            "Fed'in faiz artırımını ertelemesine gerek kalmayabilir."
        ),
        "asagi_etki": (
            "Zayıf gelen veri tüketim gerilemesine işaret eder. "
            "USD zayıflar, ekonomik yavaşlama kaygıları artar. "
            "Altın ve güvenli liman varlıkları talep görür."
        ),
        "etkilenen": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "tipik_hareket": "20-60 pip (majörler), 10-30 dolar (XAU/USD)",
        "volatilite": "⚡⚡⚡ Orta",
        "gecmis": (
            "Ocak 2024: %0.6 (beklenti %0.4) → USD/JPY 40 pip yükseldi. "
            "Kasım 2023: -%0.1 (beklenti +%0.2) → EUR/USD 35 pip yükseldi. "
            "Eylül 2023: %0.7 (beklenti %0.3) → XAU/USD 20 dolar geriledi."
        ),
        "tavsiye": (
            "Genel rakam ile çekirdek rakamı karşılaştırın; ikisi aynı yönde değilse dikkatli olun. "
            "Revizyon rakamları da piyasayı etkileyebilir. "
            "NFP haftasında değilse piyasa etkisi daha belirgin olur."
        ),
    },
    {
        "_keywords": ["unemployment", "jobless claims", "initial claims", "işsizlik", "işsizlik maaşı", "initial jobless"],
        "name": "İşsizlik / Haftalık İşsizlik Başvuruları",
        "kategori": "İstihdam",
        "frekans": "Haftalık (Perşembe); aylık işsizlik oranı ayrıca aylık yayımlanır",
        "aciklama": (
            "İlk kez işsizlik ödeneği başvurusu yapan kişi sayısını ölçer. "
            "Hızlı frekanslı bir gösterge olduğu için işgücü piyasasının anlık nabzını verir. "
            "4 haftalık hareketli ortalama daha güvenilir bir trend göstergesidir."
        ),
        "yukari_etki": (
            "Beklentinin altında işsizlik başvurusu (iyi haber) → USD değer kazanır. "
            "İşgücü piyasası sağlıklı demektir, Fed sıkılaştırma sürdürebilir. "
            "XAU/USD üzerinde aşağı yönlü baskı oluşabilir."
        ),
        "asagi_etki": (
            "Beklentinin üzerinde işsizlik başvurusu (kötü haber) → USD zayıflar. "
            "İşgücü piyasası zayıflıyor demektir, Fed gevşeme baskısı artar. "
            "XAU/USD yükselir, güvenli liman talebi artar."
        ),
        "etkilenen": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "tipik_hareket": "15-50 pip (majörler), 8-25 dolar (XAU/USD)",
        "volatilite": "⚡⚡ Orta-Düşük",
        "gecmis": (
            "Şubat 2024: 201K (beklenti 218K) → USD/JPY 30 pip yükseldi. "
            "Ekim 2023: 220K (beklenti 210K) → EUR/USD 25 pip yükseldi. "
            "Ağustos 2023: 248K (beklenti 230K) → XAU/USD 12 dolar fırladı."
        ),
        "tavsiye": (
            "Tek haftalık veriyi değil, 4 haftalık trendi takip edin. "
            "NFP'den önce gelen son haftalık veri önem kazanır. "
            "200K altı güçlü, 250K üzeri zayıf işgücü piyasası anlamına gelir."
        ),
    },
    {
        "_keywords": ["ecb", "european central bank", "avrupa merkez", "lagarde", "euro zone rate", "eurozone rate"],
        "name": "ECB / Avrupa Merkez Bankası Faiz Kararı",
        "kategori": "Merkez Bankası Kararı",
        "frekans": "Yılda 8 kez (her 6-7 haftada bir)",
        "aciklama": (
            "Avrupa Merkez Bankası'nın ana refinansman faiz oranı kararıdır. "
            "Kararın yanı sıra Lagarde'ın basın toplantısı kritik sinyal verir. "
            "Euro'nun yönünü belirleyen en önemli para politikası olayıdır."
        ),
        "yukari_etki": (
            "Faiz artırımı veya şahin sinyal → EUR değer kazanır, EUR/USD yükselir. "
            "GBP/USD da genellikle olumlu etkilenir. "
            "XAU/USD baskı altına girebilir (USD'nin tepkisine bağlı)."
        ),
        "asagi_etki": (
            "Faiz indirimi veya güvercin sinyal → EUR değer kaybeder, EUR/USD düşer. "
            "GBP/EUR etkilenebilir. "
            "Risk iştahı azalırsa XAU/USD yükselişe geçebilir."
        ),
        "etkilenen": ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/CHF", "GBP/USD"],
        "tipik_hareket": "60-180 pip (EUR çiftleri)",
        "volatilite": "⚡⚡⚡⚡ Yüksek",
        "gecmis": (
            "Eylül 2023: ECB 25 baz puan artırdı ama duraklatma sinyali → EUR/USD 80 pip düştü. "
            "Haziran 2024: İlk indirim (25 baz puan) beklentilere uygun → EUR/USD 30 pip düştü. "
            "Mart 2023: Faiz artırımı süreci devam, şahin ton → EUR/USD 70 pip yükseldi."
        ),
        "tavsiye": (
            "Karar metni ve Lagarde'ın ilk açıklaması arasında 45-60 dakika olur; her ikisini de bekleyin. "
            "Soru-cevap bölümünde söylenen kelimeler EUR'da sert hareketlere neden olabilir. "
            "Karardan önceki ECB üyesi açıklamalarını (hawk/dove dengesini) takip edin."
        ),
    },
    {
        "_keywords": ["boe", "bank of england", "bailey", "İngiltere merkez", "ingiltere merkez", "mpc decision", "sterling rate"],
        "name": "BOE / İngiltere Merkez Bankası Faiz Kararı",
        "kategori": "Merkez Bankası Kararı",
        "frekans": "Yılda 8 kez (her 6-7 haftada bir)",
        "aciklama": (
            "İngiltere Merkez Bankası Para Politikası Komitesi'nin faiz kararıdır. "
            "9 üyenin oyu açıklanır; oy dağılımı (ör: 7-2 artırım) piyasa için önemli sinyaldir. "
            "Beraberinde yayımlanan Enflasyon Raporu da kritik bilgiler içerir."
        ),
        "yukari_etki": (
            "Faiz artırımı veya şahin oy dağılımı → GBP değer kazanır. "
            "GBP/USD, GBP/EUR, GBP/JPY yükselir. "
            "Bailey'nin şahin açıklamaları ek güç verir."
        ),
        "asagi_etki": (
            "Faiz indirimi veya güvercin oy dağılımı → GBP değer kaybeder. "
            "GBP/USD düşer, EUR/GBP yükselir. "
            "Ekonomik zayıflık vurgusu piyasayı olumsuz etkiler."
        ),
        "etkilenen": ["GBP/USD", "EUR/GBP", "GBP/JPY", "GBP/CHF", "GBP/AUD"],
        "tipik_hareket": "60-160 pip (GBP çiftleri)",
        "volatilite": "⚡⚡⚡⚡ Yüksek",
        "gecmis": (
            "Ağustos 2023: 25 baz puan artırım, 6-3 oy → GBP/USD 50 pip yükseldi. "
            "Kasım 2023: Faiz sabit, iki üye indirim istedi → GBP/USD 70 pip düştü. "
            "Şubat 2024: Sabit, güvercin eğilim belirginleşti → GBP/USD 80 pip düştü."
        ),
        "tavsiye": (
            "Oy dağılımına ve İngiliz enflasyon raporuna odaklanın. "
            "Bailey'nin basın toplantısı karar kadar önemlidir. "
            "İngiltere'nin Brezilya'ya yakın yüksek enflasyon ortamında BOE kararları daha kritiktir."
        ),
    },
    {
        "_keywords": ["boj", "bank of japan", "japonya merkez", "ueda", "ycc", "yield curve control", "yen rate", "tankan"],
        "name": "BOJ / Japonya Merkez Bankası Faiz Kararı",
        "kategori": "Merkez Bankası Kararı",
        "frekans": "Yılda 8 kez",
        "aciklama": (
            "Japonya Merkez Bankası para politikası kararıdır. "
            "Negatif faiz döneminden normalleşme sürecinde BOJ kararları büyük önem taşıdı. "
            "YCC (getiri eğrisi kontrolü) politikası ve müdahale açıklamaları JPY'yi sert etkiler."
        ),
        "yukari_etki": (
            "Faiz artırımı veya hawkish sinyal → JPY güçlenir, USD/JPY düşer. "
            "YCC üst limitinin genişletilmesi ya da kaldırılması da JPY'ye güç katar. "
            "EUR/JPY, GBP/JPY gibi çapraz çiftler geriler."
        ),
        "asagi_etki": (
            "Faiz sabit veya YCC genişletme → JPY değer kaybeder, USD/JPY yükselir. "
            "Carry trade pozisyonları yeniden oluşur. "
            "Olağandışı güvercin açıklama JPY'yi sert düşürür."
        ),
        "etkilenen": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "XAU/USD"],
        "tipik_hareket": "80-300 pip (JPY çiftleri)",
        "volatilite": "⚡⚡⚡⚡⚡ Çok Yüksek (müdahale riski nedeniyle)",
        "gecmis": (
            "Aralık 2022: YCC bandı genişletildi → USD/JPY 400 pip düştü (şok etki). "
            "Temmuz 2023: YCC referans noktası değiştirildi → USD/JPY 200 pip düştü. "
            "Mart 2024: Negatif faizden çıkış (+%0.1) → USD/JPY önce 150 pip düştü, sonra toparladı."
        ),
        "tavsiye": (
            "BOJ kararları genellikle sabahın erken saatlerinde gelir; uyku saatlerinizi düzenleyin. "
            "Beklenmedik politika değişikliklerine karşı stop-loss şarttır. "
            "Japonya Maliye Bakanlığı'nın kur müdahalesi tehditlerine de dikkat edin."
        ),
    },
    {
        "_keywords": ["rbnz", "reserve bank of new zealand", "yeni zelanda", "orr", "nzd rate"],
        "name": "RBNZ / Yeni Zelanda Merkez Bankası Faiz Kararı",
        "kategori": "Merkez Bankası Kararı",
        "frekans": "Yılda 7 kez",
        "aciklama": (
            "Yeni Zelanda Merkez Bankası'nın Resmi Nakit Oranı (OCR) kararıdır. "
            "Küçük ve açık bir ekonomi olan Yeni Zelanda, emtia fiyatlarına ve Çin ekonomisine duyarlıdır. "
            "NZD/USD'yi doğrudan etkiler; AUD/NZD çapraz çiftinde de büyük hareketler görülebilir."
        ),
        "yukari_etki": (
            "Faiz artırımı veya şahin ton → NZD güçlenir, NZD/USD yükselir. "
            "AUD/NZD düşer. "
            "Olumlu büyüme görünümü de NZD'ye destek verir."
        ),
        "asagi_etki": (
            "Faiz indirimi veya güvercin ton → NZD değer kaybeder, NZD/USD düşer. "
            "AUD/NZD yükselir. "
            "Resesyon endişesi vurgulanırsa NZD üzerindeki baskı artar."
        ),
        "etkilenen": ["NZD/USD", "AUD/NZD", "NZD/JPY", "EUR/NZD"],
        "tipik_hareket": "50-120 pip (NZD çiftleri)",
        "volatilite": "⚡⚡⚡ Orta-Yüksek",
        "gecmis": (
            "Mayıs 2023: RBNZ 25 baz puan artırdı ve döngünün sonuna yaklaştığını belirtti → NZD/USD 60 pip düştü. "
            "Ağustos 2023: Faiz sabit, endişeli ton → NZD/USD 40 pip geriledi. "
            "Şubat 2024: Şahin sürpriz, artırım sinyali → NZD/USD 80 pip yükseldi."
        ),
        "tavsiye": (
            "RBNZ kararları sabahın erken saatlerinde (Türkiye saatiyle gece veya sabah erken) gelir. "
            "Likidite düşük olduğundan slippage riski yüksektir. "
            "Karar öncesi NZD pozisyonu taşıyorsanız stop-loss sıkılaştırın."
        ),
    },
    {
        "_keywords": ["ppi", "producer price", "üretici fiyat", "uretici fiyat", "wholesale price"],
        "name": "PPI (Üretici Fiyat Endeksi)",
        "kategori": "Enflasyon (Öncü)",
        "frekans": "Aylık",
        "aciklama": (
            "Üreticilerin mallarını sattıkları fiyatlardaki değişimi ölçer. "
            "CPI'dan önce yayımlanan öncü bir enflasyon göstergesidir. "
            "Üretici maliyetlerinin tüketici fiyatlarına yansımasının habercisi sayılır."
        ),
        "yukari_etki": (
            "Yüksek PPI enflasyonun üretim zincirinde güçlendiğine işaret eder. "
            "USD değer kazanabilir; tahvil faizleri yükselir. "
            "CPI öncesinde yüksek gelirse piyasa dikkatli olur."
        ),
        "asagi_etki": (
            "Düşük PPI enflasyonun soğuduğunun habercisidir. "
            "USD zayıflar, faiz indirimi beklentileri güçlenir. "
            "XAU/USD yükselişe geçebilir."
        ),
        "etkilenen": ["EUR/USD", "USD/JPY", "XAU/USD"],
        "tipik_hareket": "20-60 pip (majörler), 10-25 dolar (XAU/USD)",
        "volatilite": "⚡⚡ Orta",
        "gecmis": (
            "Şubat 2024: PPI aylık %0.6 (beklenti %0.3) → USD güçlendi, EUR/USD 40 pip düştü. "
            "Kasım 2023: PPI aylık -%0.1 (beklenti %0.0) → EUR/USD 30 pip yükseldi. "
            "Haziran 2022: PPI yıllık %11.3 zirve → USD sert güçlendi, XAU/USD baskı altı."
        ),
        "tavsiye": (
            "PPI tek başına büyük hareket yaratmaz; asıl önemi CPI açıklaması öncesinde beklenti şekillendirmesidir. "
            "PPI ve CPI aynı yönde gelirse trend güçlenir. "
            "Çekirdek PPI'ya (gıda ve enerji hariç) dikkat edin."
        ),
    },
    {
        "_keywords": ["pce", "personal consumption", "kişisel tüketim", "kisise tuketim", "core pce", "deflator"],
        "name": "PCE (Kişisel Tüketim Harcamaları Fiyat Endeksi)",
        "kategori": "Enflasyon",
        "frekans": "Aylık",
        "aciklama": (
            "Fed'in tercih ettiği enflasyon ölçütüdür; CPI'dan daha geniş bir sepeti kapsar. "
            "Çekirdek PCE (enerji ve gıda hariç) Fed'in 2% hedefiyle karşılaştırılan temel göstergedir. "
            "Her ay kişisel gelir ve harcama raporu ile birlikte yayımlanır."
        ),
        "yukari_etki": (
            "Çekirdek PCE beklentinin üzerinde gelirse Fed'in faiz indirmekte acele etmeyeceği anlaşılır. "
            "USD güçlenir, tahvil faizleri yükselir. "
            "XAU/USD satış baskısıyla karşılaşabilir."
        ),
        "asagi_etki": (
            "Çekirdek PCE beklentinin altında gelirse Fed faiz indirimine yaklaştı sinyali verir. "
            "USD zayıflar, XAU/USD yükselir. "
            "Risk iştahı artar, hisse endeksleri prim yapabilir."
        ),
        "etkilenen": ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY"],
        "tipik_hareket": "30-100 pip (majörler), 15-50 dolar (XAU/USD)",
        "volatilite": "⚡⚡⚡⚡ Yüksek",
        "gecmis": (
            "Ocak 2024: Çekirdek PCE %2.9 (beklenti %3.0) → EUR/USD 60 pip yükseldi, XAU/USD 20 dolar fırladı. "
            "Kasım 2023: Çekirdek PCE %3.2 (beklenti %3.5) → Risk iştahı arttı, XAU/USD güçlendi. "
            "Şubat 2024: Çekirdek PCE %2.8 (beklenti %2.8) → Etkisiz; beklentiye uygun."
        ),
        "tavsiye": (
            "CPI'dan daha az dikkat çeker ama Fed için daha önemlidir; bu nedenle sürpriz gelirse güçlü tepki verir. "
            "Cuma günleri yayımlandığı için likidite düşük olabilir. "
            "CPI ile karşılaştırarak tutarlılığı değerlendirin."
        ),
    },
    {
        "_keywords": ["opec", "petroleum", "crude oil", "petrol", "oil production", "oil meeting"],
        "name": "OPEC Toplantısı (Petrol Üretim Kararı)",
        "kategori": "Emtia & Jeopolitik",
        "frekans": "Yılda birkaç kez (resmi toplantılar + olağanüstü toplantılar)",
        "aciklama": (
            "OPEC ve OPEC+ üyelerinin ham petrol üretim kotalarını belirlemek için düzenlediği toplantıdır. "
            "Üretim kesintisi petrol fiyatlarını yükseltirken artış baskı yaratır. "
            "Petrol ihracatçısı ülkelerin para birimlerini (CAD, NOK, RUB) doğrudan etkiler."
        ),
        "yukari_etki": (
            "Üretim kesintisi kararı → petrol fiyatları yükselir. "
            "USD/CAD düşer (CAD güçlenir). "
            "Enflasyon kaygısı nedeniyle XAU/USD pozitif etkilenebilir."
        ),
        "asagi_etki": (
            "Üretim artışı kararı → petrol fiyatları düşer. "
            "USD/CAD yükselir (CAD zayıflar). "
            "Risk iştahı azalabilir."
        ),
        "etkilenen": ["USD/CAD", "XAU/USD", "USD/NOK", "USD/RUB"],
        "tipik_hareket": "50-200 pip (USD/CAD), 10-40 dolar (XAU/USD)",
        "volatilite": "⚡⚡⚡ Orta-Yüksek",
        "gecmis": (
            "Kasım 2023: OPEC+ üretim kesintisini Mart 2024'e kadar uzattı → WTI 2 dolar yükseldi. "
            "Haziran 2023: Suudi Arabistan gönüllü kesinti açıkladı → WTI 4 dolar fırladı. "
            "Mart 2020: Rusya ile anlaşmazlık, üretim savaşı → WTI %25 çöküş."
        ),
        "tavsiye": (
            "Toplantı sonuçları bazen sızıntıyla önceden fiyatlanır. "
            "Karar açıklandıktan sonra ilk hareket geri alınabilir. "
            "CAD ve NOK çiftlerine odaklanın; USD/CAD en direkt etkilenen çifttir."
        ),
    },
    {
        "_keywords": ["trade balance", "current account", "ticaret dengesi", "ihracat", "ithalat", "trade deficit", "trade surplus"],
        "name": "Ticaret Dengesi (Trade Balance)",
        "kategori": "Dış Ticaret",
        "frekans": "Aylık",
        "aciklama": (
            "Bir ülkenin ihracat ve ithalat arasındaki farkı ölçer. "
            "Ticaret açığı (ithalat > ihracat) kısa vadede para birimini zayıflatabilir. "
            "Ticaret fazlası ise ülke parasına talep yaratır."
        ),
        "yukari_etki": (
            "Beklentinden az açık (veya daha fazla fazla) → para birimi değer kazanır. "
            "İhracat artışı ekonomik güce işaret eder. "
            "USD için pozitif, EUR, GBP için karışık etki olabilir."
        ),
        "asagi_etki": (
            "Beklentiden fazla açık → para birimi değer kaybeder. "
            "İthalat artışı enflasyona katkıda bulunabilir. "
            "Dış denge sorununun derinleştiğini gösterir."
        ),
        "etkilenen": ["EUR/USD", "USD/JPY", "AUD/USD", "USD/CAD"],
        "tipik_hareket": "10-40 pip (majörler)",
        "volatilite": "⚡⚡ Düşük-Orta",
        "gecmis": (
            "Kasım 2023 ABD Ticaret Dengesi: -$64.3B (beklenti -$64.8B) → USD hafif güçlendi. "
            "Ağustos 2023 Almanya: Tahminlerin üzerinde ihracat artışı → EUR/USD 20 pip yükseldi. "
            "Japonya ticaret açıkları 2022-2023 boyunca JPY zayıflığına katkıda bulundu."
        ),
        "tavsiye": (
            "Tek başına büyük hareket yaratmaz; daha büyük ekonomik tablonun parçası olarak değerlendirin. "
            "Büyük sürprizler beklentide anlamlı fark varsa etkili olabilir. "
            "Çin'in ticaret verileri hem AUD hem de JPY için kritiktir."
        ),
    },
    {
        "_keywords": ["housing", "building permits", "new home sales", "existing home", "konut", "mortgage", "pending home"],
        "name": "Konut Verileri (Housing Starts / Permits / Sales)",
        "kategori": "Konut Sektörü",
        "frekans": "Aylık",
        "aciklama": (
            "Konut başlangıçları, izinleri ve satışlarını kapsayan göstergeler ekonomik aktivite ve faiz duyarlılığını ölçer. "
            "Yüksek faiz ortamında konut sektörü ilk daralan sektör olduğundan öncü gösterge özelliği taşır. "
            "ABD ekonomisinin %15-18'ini temsil eder."
        ),
        "yukari_etki": (
            "Beklentinin üzerinde konut verisi ekonominin dayanıklı olduğunu gösterir. "
            "USD hafif güçlenebilir. "
            "Faiz hassasiyeti nedeniyle güçlü konut verisi bazen Fed'in daha az indirim yapacağını çağrıştırır."
        ),
        "asagi_etki": (
            "Zayıf konut verisi ekonomik yavaşlamaya işaret eder. "
            "USD hafif zayıflar. "
            "Fed'in faiz indirmesine gerekçe oluşturur."
        ),
        "etkilenen": ["EUR/USD", "USD/JPY", "XAU/USD"],
        "tipik_hareket": "10-30 pip (majörler)",
        "volatilite": "⚡ Düşük-Orta",
        "gecmis": (
            "Şubat 2024: Konut başlangıçları 1.42M (beklenti 1.45M) → USD hafif zayıfladı. "
            "Ekim 2023: İzinler 1.50M (beklenti 1.45M) → USD/JPY 15 pip yükseldi. "
            "2022 yılı: Yükselen faizlerle konut sektöründe %30+ daralma, USD'ye kısmî destek."
        ),
        "tavsiye": (
            "Konut verisi tek başına büyük pozisyon almak için yeterli değildir. "
            "Mevcut faiz ortamıyla birlikte değerlendirin. "
            "Mortgage başvuru verileriyle karşılaştırarak bağlamı anlayın."
        ),
    },
    {
        "_keywords": ["adp", "adp employment", "adp nonfarm", "private payrolls", "private employment", "automatic data processing"],
        "name": "ADP Özel Sektör İstihdamı",
        "kategori": "İstihdam (Öncü)",
        "frekans": "Aylık (genellikle NFP'den 2 gün önce, Çarşamba)",
        "aciklama": (
            "ADP firması tarafından hazırlanan ve özel sektördeki istihdam değişimini ölçen rapordur. "
            "NFP'nin habercisi olarak değerlendirilir, ancak ikisi her zaman aynı yönde gitmez. "
            "Sector bazında (büyük/küçük işletme) ayrıntılar da analiz edilir."
        ),
        "yukari_etki": (
            "Güçlü ADP → NFP beklentileri yükselir, USD değer kazanır. "
            "İşgücü piyasasının sağlıklı olduğu teyit edilir. "
            "XAU/USD baskı altına girebilir."
        ),
        "asagi_etki": (
            "Zayıf ADP → NFP beklentileri düşer, USD zayıflar. "
            "İşgücü piyasasında yavaşlama sinyali verir. "
            "XAU/USD ve güvenli liman varlıkları yükselir."
        ),
        "etkilenen": ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"],
        "tipik_hareket": "20-60 pip (majörler), 10-25 dolar (XAU/USD)",
        "volatilite": "⚡⚡ Orta",
        "gecmis": (
            "Ocak 2024: ADP 107K (beklenti 145K) → EUR/USD 30 pip yükseldi. "
            "Eylül 2023: ADP 177K (beklenti 195K) → USD/JPY 20 pip düştü. "
            "Şubat 2024: ADP 140K (beklenti 150K) → Sınırlı etki, piyasa NFP'yi bekledi."
        ),
        "tavsiye": (
            "ADP ile NFP arasındaki korelasyon düşüktür; ADP'ye körce güvenmemelisiniz. "
            "ADP, piyasa beklentilerini kalibre etmek için kullanışlıdır. "
            "NFP günü aynı yönde gelirse trend güçlenir."
        ),
    },
]


# ---------------------------------------------------------------------------
# Eşleştirme fonksiyonu
# ---------------------------------------------------------------------------

def get_event_analysis(event_title: str) -> Optional[dict]:
    """
    Olay başlığına göre Türkçe etki analizini döndürür.
    Büyük/küçük harf duyarsız anahtar kelime eşleştirmesi kullanır.

    Parametreler
    ------------
    event_title : str
        Takvimden gelen olay adı (örn: "Nonfarm Payrolls", "ECB Rate Decision")

    Döndürür
    --------
    dict | None
        Analiz sözlüğü veya eşleşme bulunamazsa None
    """
    if not event_title:
        return None

    lower_title = event_title.lower()

    for event in _EVENTS:
        keywords: list[str] = event.get("_keywords", [])
        for kw in keywords:
            if kw.lower() in lower_title:
                # _keywords anahtarını temizleyerek döndür
                result = {k: v for k, v in event.items() if k != "_keywords"}
                return result

    return None

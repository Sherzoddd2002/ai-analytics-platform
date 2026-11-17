# AI Analytics Platform

Julius.ai uslubidagi professional AI analitik platforma. Fayl yuklang va avtomatik analiz, tozalash, grafiklar, ML prediction — hammasi bir joyda.

## Imkoniyatlar

| Bo'lim | Funksiya |
|--------|----------|
| Fayl Yuklash | CSV, Excel (.xlsx, .xls), JSON, TSV |
| Dashboard | Avtomatik statistik kartalar va avto-grafiklar |
| Ma'lumot Tozalash | Dublikatlar, yetishmayotgan qiymatlar, anomal qiymatlar (IQR/Z-score), normalizatsiya, kategoriyali kodlash |
| Grafiklar | Scatter, Line, Bar, Histogram, Box, Violin, Pie, Area, Heatmap, 3D Scatter, Sunburst, Treemap (12 tur) |
| Machine Learning | Regression (Linear, Random Forest, Gradient Boosting, Decision Tree), Classification, Auto-model selection, Feature Importance |
| Klasterlash | K-Means, Elbow Method, PCA 3D vizualizatsiya |
| Statistik Tahlil | Korrelyatsiya matrisasi, T-Test, Taqsimot (Shapiro-Wilk, Skewness, Kurtosis) |
| AI Insights | Avtomatik ma'lumot tahlili va tavsiyalar |
| Yuklab Olish | Tozalangan ma'lumotni CSV/Excel/JSON formatda |

## Talab qilingan dasturlar

- **Python 3.10+** (tavsiya: 3.12 yoki 3.13)
- **pip** (Python bilan birga keladi)

## O'rnatish (qadam-baqadam)

### 1. Loyihani yuklab olish

```bash
git clone https://github.com/Sherzoddd2002/ai-analytics-platform.git
cd ai-analytics-platform
```

Yoki ZIP sifatida yuklab oling va arxivdan chiqaring.

### 2. Virtual muhit yaratish (tavsiya etiladi)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Kutubxonalarni o'rnatish

```bash
pip install -r requirements.txt
```

### 4. Ishga tushirish

```bash
python app.py
```

### 5. Brauzerda oching

```
http://localhost:5000
```

## Foydalanish

1. **Fayl yuklang** — CSV yoki Excel faylni drag-and-drop qiling
2. **Dashboard** — avtomatik grafiklar va statistikani ko'ring
3. **Tozalash** — ma'lumotni tozalang (dublikatlar, NaN, outliers, encoding)
4. **Grafiklar** — o'zingiz xohlagan grafikni yarating (12 tur)
5. **ML / Prediction** — target ustunni tanlab bashorat qiling (auto-model)
6. **Klasterlash** — K-Means bilan guruhlang
7. **Statistika** — korrelyatsiya, T-test, taqsimot tahlili
8. **AI Insights** — avtomatik tavsiyalar oling
9. **Yuklab olish** — tozalangan ma'lumotni eksport qiling

## Texnologiyalar

- **Backend**: Flask 3.1, Pandas 2.2, NumPy 2.1, SciPy 1.15, Scikit-learn 1.6, Statsmodels 0.14
- **Frontend**: HTML5, CSS3, JavaScript (vanilla), Plotly.js 2.27
- **Vizualizatsiya**: Plotly

## Loyiha tuzilishi

```
ai-analytics-platform/
|-- app.py                 # Flask backend (barcha API endpointlar)
|-- requirements.txt       # Kutubxonalar
|-- README.md              # Hujjat
|-- .gitignore
|-- templates/
|   `-- index.html         # Asosiy sahifa
|-- static/
|   |-- css/
|   |   `-- style.css      # Stil
|   `-- js/
|       `-- app.js         # Frontend logika
`-- uploads/               # Yuklang fayllar (avtomatik yaratiladi)
```

## Xatolik yuzaga kelsa

```bash
# pip ni yangilang
python -m pip install --upgrade pip

# Kutubxonalarni qayta o'rnating
pip install -r requirements.txt --force-reinstall
```

Agar port 5000 band bo'lsa, portni `PORT` muhit o'zgaruvchisi bilan o'zgartiring:

```bash
# Windows
set PORT=5055 && python app.py
# macOS / Linux
PORT=5055 python app.py
```

## Optimallashtirishlar (tezkor versiya)

- Profil, grafiklar, ustun ma'lumotlari va AI Insights **bir marta hisoblanib session ichida keshlanadi**; tab almashtirishda qayta hisoblanmaydi.
- Grafiklar uchun **2 000 qator** namunasi ishlatiladi (100 000 qatorlik scatter endi 22 MB emas, ~0.24 MB gzip).
- Login/ML: model 10 000 qator namunasida o'rgatiladi, cross-validation 3 000 qator va 3 katlamada; eng yaxshi model **ikkinchi marta qayta o'rgatilmaydi**.
- Klasterlash: 10 000 qator namunasi, katta ma'lumotda `MiniBatchKMeans`, silhouette 2 000 nuqtada, Elbow eng ko'pi 8 ta K.
- `/clean` endi faqat natija + kichik profil/preview qaytaradi; grafiklar `/charts/auto` orqali alohida olinadi.
- JSON javoblar **gzip** bilan siqiladi; `debug=False` va sessiyalar 6 soat / 20 sessiya bilan cheklangan.
- Fayl nomi va foydalanuvchi ma'lumotlari HTML'ga chiqarishda escape qilinadi; xato traceback'i brauzerga yuborilmaydi; 250 000 qator / 200 ustun chegarasi mavjud.

(grafiklar, AI Insights va ML bashoratlari namuna asosida hisoblanadi — ya'ni taxminiy ko'rsatkich; asl ma'lumot va tozalash operatsiyalari to'liq qatorlarda bajariladi).

## Litsenziya

MIT - erkin foydalaning.
